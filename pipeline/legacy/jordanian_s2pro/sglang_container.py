"""Executed inside the pinned container: validate CUDA and bound the S2 engine."""
import argparse
import importlib.metadata
import inspect
import json
from pathlib import Path
import subprocess

OVERRIDES = {
    "dtype": "bfloat16",
    "max_running_requests": 1, "max_total_tokens": 8192,
    "mem_fraction_static": 0.32, "chunked_prefill_size": 2048,
    "cuda_graph_bs_decode": [1], "cuda_graph_max_bs_decode": 1,
    "disable_prefill_cuda_graph": True,
    "torch_compile_max_bs": 1, "enable_torch_compile": True,
    "disable_cuda_graph": False, "attention_backend": "fa3",
}


def config_for(model_path):
    from sglang_omni.models.fishaudio_s2_pro.config import S2ProPipelineConfig
    from sglang_omni.models.fishaudio_s2_pro.stages import create_sglang_tts_engine_executor
    from sglang.srt.server_args import ServerArgs
    from sglang_omni.models.fishaudio_s2_pro.engine_builder import FishS2ProEngineBuilder
    if FishS2ProEngineBuilder.context_length != 4096:
        raise RuntimeError("Unexpected S2 Pro context length")

    parameters = inspect.signature(create_sglang_tts_engine_executor).parameters
    if "server_args_overrides" not in parameters:
        raise RuntimeError("Pinned S2 engine does not support bounded ServerArgs; refusing unbounded startup")
    unknown = set(OVERRIDES) - set(inspect.signature(ServerArgs).parameters)
    if unknown:
        raise RuntimeError(f"Unknown engine settings: {sorted(unknown)}")
    config = S2ProPipelineConfig(model_path=str(model_path))
    data = config.model_dump(mode="json", exclude_none=True)
    engines = [s for s in data["stages"] if s["name"] == "tts_engine"]
    if len(engines) != 1:
        raise RuntimeError("Unexpected S2 Pro pipeline stages")
    stage = engines[0]
    stage["factory"] = "sglang_telemetry.create_timed_tts_engine_executor"
    fields = type(next(s for s in config.stages if s.name == "tts_engine")).model_fields
    if "engine" in fields:
        stage["engine"] = OVERRIDES
        stage.setdefault("factory", {})["max_new_tokens"] = 2048
    elif "factory_args" in fields:
        stage.setdefault("factory_args", {}).update(max_new_tokens=2048,
            server_args_overrides=OVERRIDES)
    else:
        raise RuntimeError("Unexpected engine configuration schema")
    # Validate the complete config with the installed release, not a local facsimile.
    checked = S2ProPipelineConfig.model_validate(data)
    engine = next(s for s in checked.stages if s.name == "tts_engine")
    effective = (engine.engine.overrides() if "engine" in fields else
                 engine.factory_args["server_args_overrides"])
    if any(effective.get(k) != v for k, v in OVERRIDES.items()):
        raise RuntimeError("Engine bounds were dropped during configuration validation")
    result = checked.model_dump(mode="json", exclude_none=True)
    result["config_cls"] = "S2ProPipelineConfig"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import torchaudio
    from sglang_omni.serve.protocol import CreateSpeechRequest
    from sglang_telemetry import validate_runtime
    validate_runtime()
    for package, expected in (("sglang", "0.5.16"), ("sglang-omni", "0.1.3")):
        if importlib.metadata.version(package) != expected:
            raise RuntimeError(f"Unexpected {package} version")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (9, 0):
        raise RuntimeError("This launcher requires the selected H100 GPU and a CUDA-13 compatible NVIDIA container runtime/driver")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is unavailable")
    # Exercise driver/runtime interop before stopping the current server.
    x = torch.ones((16, 16), device="cuda", dtype=torch.bfloat16)
    assert (x @ x).float().mean().item() == 16
    torch.cuda.synchronize()
    versions = {p: importlib.metadata.version(p) for p in
                ("sglang", "sglang-omni", "transformers", "flashinfer-python")}
    manifest = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for name, path in manifest["exports"].items():
        (args.output / (name + ".json")).write_text(json.dumps(config_for(path), indent=2))
    help_text = subprocess.check_output(["sgl-omni", "serve", "--help"], text=True)
    if not all(flag in help_text for flag in ("--host", "--port", "--config", "--model-path")):
        raise RuntimeError("Unexpected serving CLI")
    receipt = {"versions": versions, "torch": torch.__version__,
               "token_timing": {"factory": "sglang_telemetry.create_timed_tts_engine_executor",
                                "clock": "monotonic", "tagged_streams_only": True},
               "speech_fields": sorted(CreateSpeechRequest.model_fields),
               "torchaudio": torchaudio.__version__, "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(0),
               "total_memory": torch.cuda.get_device_properties(0).total_memory,
               "settings": OVERRIDES}
    (args.output / "preflight.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()

