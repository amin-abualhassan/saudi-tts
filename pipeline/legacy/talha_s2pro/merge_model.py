"""Merge a future adapter using its saved run settings. Not executed during setup."""
import argparse
import json
from pathlib import Path
import sys

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fish-speech-upstream"))


def lora_settings(config_path):
    config = yaml.safe_load(config_path.read_text())
    lora = dict(config["model"]["model"]["lora_config"])
    lora.pop("_target_", None)
    if not {"r", "lora_alpha", "target_modules"} <= lora.keys():
        raise ValueError("Run config must explicitly contain rank, alpha and target modules")
    base = config["pretrained_ckpt_path"]
    if "${" in base:
        raise ValueError("Use a resolved saved run config for merge")
    return Path(base), lora


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite an existing merged model")
    base, settings = lora_settings(args.experiment_config)
    if not base.is_absolute():
        raise ValueError("Resolve the base model path to an absolute path")
    if not (base / "codec.pth").exists():
        raise ValueError("S2 Pro codec is missing from the base directory")
    import torch
    from fish_speech.models.text2semantic.llama import BaseTransformer
    from fish_speech.models.text2semantic.lora import LoraConfig

    model = BaseTransformer.from_pretrained(str(base), load_weights=True,
                                           lora_config=LoraConfig(**settings))
    # Own Lightning checkpoint includes optimizer/config objects; not a tensor-only file.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    adapters = {name.removeprefix("model."): value for name, value in state.items() if "lora_" in name}
    expected = {name for name in model.state_dict() if "lora_" in name}
    if not adapters or set(adapters) != expected:
        raise ValueError("Adapter names do not match the saved experiment configuration")
    for name, value in adapters.items():
        if not torch.isfinite(value).all() or value.shape != model.state_dict()[name].shape:
            raise ValueError(f"Invalid adapter tensor: {name}")
    model.load_state_dict(model.state_dict() | adapters, strict=True)
    model.eval()
    model.save_pretrained(str(args.output), drop_lora=True)
    # Upstream inference looks for codec.pth inside the model directory.
    # Link the original codec rather than duplicating its 1.8 GB of unchanged weights.
    (args.output / "codec.pth").symlink_to((base / "codec.pth").resolve())
    (args.output / "merge_provenance.json").write_text(json.dumps({
        "base": str(base), "checkpoint": str(args.checkpoint.resolve()),
        "experiment_config": str(args.experiment_config.resolve()), "lora": settings,
        "codec": str(base / "codec.pth"),
    }, indent=2) + "\n")
    print(f"Merged model saved to {args.output}; use the original S2 Pro codec for decoding.")


if __name__ == "__main__":
    main()
