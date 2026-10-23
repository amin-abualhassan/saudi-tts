"""Export the exact selected LoRAs in the original S2 Pro HF layout, on CPU."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import sys
import uuid

from common import BASE, HERE, UPSTREAM, atomic_json, file_hash, identity, read_json

EXPORT_VERSION = 1


def native_key(key):
    if key.startswith("fast_"):
        return "audio_decoder." + key.removeprefix("fast_")
    if key.startswith("codebook_embeddings."):
        return "audio_decoder." + key
    return "text_model.model." + key


def merge_weight(weight, a, b, alpha, embedding=False):
    """Match loralib's eval merge after the serving model is cast to BF16."""
    import torch
    a, b = a.to(weight.dtype), b.to(weight.dtype)
    delta = b @ a
    if embedding:
        delta = delta.T
    if delta.shape != weight.shape or not torch.isfinite(delta).all():
        raise ValueError("Invalid adapter delta")
    merged = weight + delta * (alpha / a.shape[0])
    if not torch.isfinite(merged).all():
        raise ValueError("Nonfinite merged weight")
    return merged.contiguous()


def load_adapters(entry, base=BASE):
    import torch
    import yaml
    sys.path.insert(0, str(UPSTREAM))
    from fish_speech.models.text2semantic.llama import DualARTransformer
    from fish_speech.models.text2semantic.lora import LoraConfig

    for field in ("checkpoint", "config"):
        if file_hash(Path(entry[field])) != entry[field + "_sha256"]:
            raise ValueError(f"Selected {field} changed")
    settings = dict(yaml.safe_load(Path(entry["config"]).read_text())["model"]["model"]["lora_config"])
    settings.pop("_target_", None)
    # Shape/target validation uses the real training implementation, with no base allocation.
    with torch.device("meta"):
        model = DualARTransformer.from_pretrained(str(base), load_weights=False,
            max_length=4096, lora_config=LoraConfig(**settings))
    expected = {n: tuple(p.shape) for n, p in model.named_parameters() if "lora_" in n}
    del model
    saved = torch.load(entry["checkpoint"], map_location="cpu", weights_only=False)
    adapters = {n.removeprefix("model."): t for n, t in saved["state_dict"].items()}
    if set(adapters) != set(expected) or any(tuple(t.shape) != expected[n] or
            not torch.isfinite(t).all() for n, t in adapters.items()):
        raise ValueError("Adapter tensors do not match the exact saved configuration")
    return adapters, settings, saved.get("global_step")


def verify_export(path, source):
    receipt = read_json(Path(path) / "export.json", {})
    if receipt.get("source") != source or not receipt.get("files"):
        return False
    return all((Path(path) / n).is_file() and file_hash(Path(path) / n) == digest
               for n, digest in receipt["files"].items())


def export_model(entry, base_identity, root, base=BASE):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    source = {"version": EXPORT_VERSION, "checkpoint": entry["checkpoint_sha256"],
              "config": entry["config_sha256"], "base": base_identity,
              "merge_dtype": "bfloat16", "merge": "loralib_eval"}
    destination = Path(root) / entry["name"] / identity(source)
    if verify_export(destination, source):
        return str(destination)
    if destination.exists():
        raise ValueError(f"Incomplete or changed export: {destination}; move it aside before retrying")
    adapters, settings, step = load_adapters(entry, base)
    index = read_json(base / "model.safetensors.index.json")
    updates = {}
    for name in adapters:
        if not name.endswith(".lora_A"):
            continue
        module = name.removesuffix(".lora_A")
        key = native_key(module + ".weight")
        if key not in index["weight_map"]:
            raise ValueError(f"Missing native target: {key}")
        updates[key] = (adapters[name], adapters[module + ".lora_B"],
                        module in ("embeddings", "fast_embeddings", "codebook_embeddings"))
    if len(updates) * 2 != len(adapters):
        raise ValueError("Unpaired adapter tensors")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / (".export-" + uuid.uuid4().hex)
    temporary.mkdir()
    try:
        files = {}
        for name, digest in base_identity.items():
            src = base / name
            if src.name != name or file_hash(src) != digest:
                raise ValueError(f"Base artifact changed: {name}")
            dst = temporary / name
            targets = {key for key in updates if index["weight_map"][key] == name}
            if targets:
                print(f"Merging {entry['name']}: {len(targets)} weights in {name}", flush=True)
                with safe_open(src, framework="pt", device="cpu") as shard:
                    expected_keys = {k for k, value in index["weight_map"].items() if value == name}
                    if set(shard.keys()) != expected_keys:
                        raise ValueError(f"Native shard/index mismatch: {name}")
                    tensors = {}
                    metadata = shard.metadata()
                    for key in shard.keys():
                        weight = shard.get_tensor(key)
                        if key in targets:
                            a, b, embedding = updates[key]
                            weight = merge_weight(weight.to(torch.bfloat16), a, b,
                                                  settings["lora_alpha"], embedding)
                        tensors[key] = weight
                    save_file(tensors, str(dst), metadata=metadata)
                del tensors, weight
                gc.collect()
            elif name.endswith(".safetensors") or name == "codec.pth":
                dst.symlink_to(src.resolve())
            else:
                shutil.copyfile(src, dst)
            files[name] = file_hash(dst)
        atomic_json(temporary / "export.json", {"source": source, "files": files,
            "global_step": step, "settings": settings, "merged_weights": sorted(updates),
            "adapter_tensors": len(adapters), "parameters": sum(t.numel() for t in adapters.values())})
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return str(destination)


def prepare_exports(bundle, root=None):
    import torch
    torch.set_num_threads(8)
    for name, expected in bundle["base"].items():
        if Path(name).name != name or file_hash(BASE / name) != expected:
            raise ValueError(f"Original base artifact changed: {name}")
    return {name: export_model(entry, bundle["base"], root or HERE / "sglang/models")
            for name, entry in bundle["models"].items()}


if __name__ == "__main__":
    from voice_server import selected_bundle
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare_exports(selected_bundle(args.campaign)), indent=2))
