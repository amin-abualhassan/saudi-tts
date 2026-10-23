"""First generated audio-token timestamps for explicitly tagged stream requests.

Loaded inside the pinned container through its configured engine factory. The
normal model runner is retained; timing is recorded after its first real output.
"""
import inspect
import json
import os
from pathlib import Path
import re
import time


def validate_runtime():
    from sglang_omni.models.fishaudio_s2_pro.model_runner import FishS2ProModelRunner
    method = FishS2ProModelRunner._collect_step_outputs
    if tuple(inspect.signature(method).parameters) != ("self", "result", "requests"):
        raise RuntimeError("Unsupported S2-Pro token timing hook")
    return FishS2ProModelRunner


def install_timing_hook(runner_cls, directory):
    if getattr(runner_cls, "_studio_timing_installed", False):
        return
    original = runner_cls._collect_step_outputs

    def collect(self, result, requests):
        value = original(self, result, requests)
        for request in requests:
            data = request.data
            if len(data.output_codes) != 1 or getattr(data, "_studio_token_reported", False):
                continue
            payload = getattr(data, "stage_payload", None)
            if payload is None:
                continue
            params = payload.request.params
            tag = params.get("studio_request_id") or params.get("stage_params", {}).get(
                "tts_engine", {}).get("studio_request_id")
            if not isinstance(tag, str) or not re.fullmatch(r"[a-f0-9]{32}", tag):
                continue
            token_time = time.monotonic()
            data._studio_token_reported = True
            receipt = {"first_token_monotonic": token_time,
                       "engine_time_to_first_token_seconds":
                           time.perf_counter() - data.engine_start_s if data.engine_start_s else None,
                       "native_request_id": payload.request_id,
                       "definition": "First generated audio frame collected after Slow-AR and Fast-AR"}
            try:
                directory.mkdir(parents=True, exist_ok=True)
                target = directory / (tag + ".json")
                temporary = directory / (tag + ".tmp")
                temporary.write_text(json.dumps(receipt))
                temporary.replace(target)
            except OSError:
                # A timing failure must not affect model output or normal jobs.
                pass
        return value

    runner_cls._collect_step_outputs = collect
    runner_cls._studio_timing_installed = True


def create_timed_tts_engine_executor(model_path, **kwargs):
    from sglang_omni.models.fishaudio_s2_pro.stages import create_sglang_tts_engine_executor
    install_timing_hook(validate_runtime(), Path(os.environ["JORDANIAN_TOKEN_TIMING_DIR"]))
    return create_sglang_tts_engine_executor(model_path, **kwargs)
