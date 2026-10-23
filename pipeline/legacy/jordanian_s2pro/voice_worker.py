"""Persistent, serial CUDA worker for the exact listening-selected adapters."""
import argparse
from collections import OrderedDict
from pathlib import Path
import sys
import time
import traceback

from common import BASE, UPSTREAM, file_hash, read_json, talha_pipeline
from voice_server import PREFIX


def emit(kind, **values):
    import json
    print(PREFIX + json.dumps({"type": kind, **values}, allow_nan=False), flush=True)


def load_adapter_model(entry, device, precision):
    import torch
    import yaml
    from fish_speech.models.text2semantic.llama import DualARTransformer
    from fish_speech.models.text2semantic.lora import LoraConfig

    checkpoint, config_path = Path(entry["checkpoint"]), Path(entry["config"])
    if file_hash(checkpoint) != entry["checkpoint_sha256"] or file_hash(config_path) != entry["config_sha256"]:
        raise ValueError("Selected checkpoint or configuration changed")
    config = yaml.safe_load(config_path.read_text())
    settings = dict(config["model"]["model"]["lora_config"])
    settings.pop("_target_", None)
    model = DualARTransformer.from_pretrained(str(BASE), load_weights=True, max_length=4096,
                                             lora_config=LoraConfig(**settings))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    adapters = {name.removeprefix("model."): tensor for name, tensor in saved["state_dict"].items()}
    state = model.state_dict()
    expected = {name for name in state if "lora_" in name}
    if set(adapters) != expected or any(t.shape != state[n].shape or not torch.isfinite(t).all()
                                       for n, t in adapters.items()):
        raise ValueError("Selected adapter does not match its saved LoRA configuration")
    model.load_state_dict(state | adapters, strict=True)
    del saved, adapters, state
    model = model.to(device=device, dtype=precision).eval()
    if model.tokenizer is None or model.config.num_codebooks != 10:
        raise ValueError("Invalid S2 Pro tokenizer or codebook configuration")
    return model


def reference_tokens(reference, codec, device):
    import numpy as np
    import torch
    from fish_speech.models.text2semantic.inference import encode_audio

    wav = Path(reference["prepared_audio"])
    if file_hash(wav) != reference["wav_sha256"]:
        raise ValueError("Reference audio changed")
    tokens = wav.with_suffix(".npy")
    if not reference.get("uploaded") and tokens.is_file():
        receipt = read_json(Path(reference["dataset"]) / "token-receipts" / (reference["clip_id"] + ".json"), {})
        if receipt.get("tokens") != file_hash(tokens) or receipt.get("wav") != reference["wav_sha256"]:
            raise ValueError("Reference codec tokens changed")
        talha_pipeline.validate_tokens(tokens, reference["duration"])
        return torch.from_numpy(np.load(tokens, allow_pickle=False)).long().cpu()
    return encode_audio(wav, codec, device).long().cpu()


def run_worker(selection, gpu, model_cache):
    import gc
    import json
    sys.path.insert(0, str(UPSTREAM))
    import numpy as np
    import soundfile as sf
    import torch
    from fish_speech.models.text2semantic.inference import (
        decode_one_token_ar, generate_long, load_codec_model, decode_to_audio)
    from compare import audio_metrics

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run this server in the fish-speech environment on the H100 host")
    torch.cuda.set_device(gpu)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The voice server requires a GPU with BF16 support")
    device, precision = f"cuda:{gpu}", torch.bfloat16
    bundle = read_json(selection)
    emit("loading", message="Verifying the original S2 Pro model and codec")
    if not bundle.get("base"):
        raise ValueError("Missing original-model integrity receipt")
    for name, expected in bundle["base"].items():
        if Path(name).name != name or file_hash(BASE / name) != expected:
            raise ValueError(f"Original S2 Pro artifact changed: {name}")
    if model_cache == 2 and torch.cuda.mem_get_info(gpu)[0] < 32 * 2**30:
        model_cache = 1
        emit("loading", message="Using one resident model; models load when you switch")
    torch.set_float32_matmul_precision("high")
    codec = load_codec_model(BASE / "codec.pth", device, precision)
    models, references = OrderedDict(), OrderedDict()

    def get_model(name):
        if name in models:
            models.move_to_end(name)
            return models[name]
        if len(models) >= model_cache:
            old_name, old = models.popitem(last=False)
            del old
            gc.collect()
            torch.cuda.empty_cache()
        emit("loading", message="Loading " + name)
        models[name] = load_adapter_model(bundle["models"][name], device, precision)
        emit("model_loaded", models=list(models))
        return models[name]

    for name in list(bundle["models"])[:model_cache]:
        get_model(name)
    emit("ready", models=list(models))
    for line in sys.stdin:
        item = json.loads(line)
        request = item["request"]
        started = time.monotonic()
        try:
            # Release the previous request's reference before a one-model cache eviction.
            model = None
            model = get_model(request["model"])
            ref = item["reference"]
            key = ref["wav_sha256"]
            if key not in references:
                references[key] = reference_tokens(ref, codec, device)
                if len(references) > 32:
                    references.popitem(last=False)
            references.move_to_end(key)
            tokens = references[key]
            torch.manual_seed(request["seed"])
            torch.cuda.manual_seed_all(request["seed"])
            codes = []
            with torch.inference_mode():
                for response in generate_long(model=model, device=device, decode_one_token=decode_one_token_ar,
                        text="<|speaker:0|>" + request["text"], num_samples=1,
                        max_new_tokens=request["max_new_tokens"], temperature=request["temperature"],
                        top_p=request["top_p"], top_k=30, compile=False, iterative_prompt=True,
                        chunk_length=200, prompt_tokens=[tokens], prompt_text=[ref["text"]]):
                    if response.action == "sample":
                        codes.append(response.codes)
                if not codes:
                    raise ValueError("No generated audio tokens")
                generated = torch.cat(codes, dim=1)
                audio = decode_to_audio(generated.to(device), codec).float().cpu().numpy()
            metrics = audio_metrics(audio, codec.sample_rate)
            if metrics["rms"] < .000001:
                raise ValueError("Generated audio is silent; try a shorter prompt or another reference")
            audio = audio / max(1.0, float(np.max(np.abs(audio))) / .98)
            output = Path(item["output"])
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(".tmp.wav")
            sf.write(temporary, audio, codec.sample_rate, subtype="PCM_16")
            temporary.replace(output)
            emit("result", id=item["id"], duration=metrics["duration"], warning=metrics["warning"],
                 sample_rate=codec.sample_rate, elapsed=round(time.monotonic() - started, 2))
        except Exception as error:
            traceback.print_exc()
            emit("job_error", id=item["id"], error=str(error))
            if isinstance(error, torch.cuda.OutOfMemoryError):
                raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model-cache", type=int, choices=(1, 2), default=2)
    parser.add_argument("--optimization", choices=("eager", "compiled"), default="eager")
    parser.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune-no-cudagraphs"), default="reduce-overhead")
    args = parser.parse_args()
    # Accept former optimization flags for workers restarted by an older server.
    # They are deliberately ignored: serving always uses the original decoder.
    run_worker(args.selection, args.gpu, args.model_cache)
