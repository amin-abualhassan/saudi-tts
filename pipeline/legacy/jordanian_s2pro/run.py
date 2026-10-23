#!/usr/bin/env python3
"""Sequential S2 Pro experiments; default invocation performs real training."""
import argparse
from contextlib import ExitStack
import fcntl
import json
import math
from pathlib import Path
import re
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

from common import BASE, HERE, TALHA, atomic_json, file_hash, identity, read_json, read_rows, shared, talha_pipeline, talha_validation, write_rows
import corpus
from compare import cases_for


def load_config(path):
    c = json.loads(Path(path).read_text(encoding="utf-8"))
    families = {"talha", "podcast", "single"}
    if set(c["hours"]) != families or set(c["family_weights"]) != families:
        raise ValueError("hours and family_weights must contain talha, podcast, single")
    for values in (c["hours"], c["family_weights"]):
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in values.values()):
            raise ValueError("Budgets and sampling weights must be positive and finite")
    if sum(c["hours"].values()) > 20:
        raise ValueError("Hard safety cap: at most 20 total hours per suite; use separate bounded suites")
    for key in ("steps", "max_length", "validation_steps", "accumulation", "seed"):
        if type(c[key]) is not int or c[key] < (0 if key == "seed" else 1):
            raise ValueError(f"{key} must be a positive integer (seed may be zero)")
    if not 1024 <= c["max_length"] <= 4096 or c["steps"] > 10000 or c["accumulation"] > 16:
        raise ValueError("Use context 1024–4096, at most 10000 steps and accumulation <=16")
    experiments = c["experiments"]
    if not 1 <= len(experiments) <= 8 or len({e["name"] for e in experiments}) != len(experiments):
        raise ValueError("Supply 1–8 uniquely named experiments")
    allowed = {"attention", "mlp", "embeddings", "output", "fast_attention", "fast_mlp", "fast_embeddings", "fast_output"}
    for e in experiments:
        if not re.fullmatch(r"[a-zA-Z0-9_]+", e["name"]) or type(e["r"]) is not int or not 1 <= e["r"] <= 64:
            raise ValueError("Experiment names must be safe and ranks between 1 and 64")
        if (not math.isfinite(e["lr"]) or not 0 < e["lr"] <= 5e-5 or
                not math.isfinite(e["alpha"]) or not 0 < e["alpha"] <= 128 or
                not e["targets"] or not set(e["targets"]) <= allowed):
            raise ValueError("Invalid LoRA targets, alpha or learning rate (maximum 5e-5)")
    evaluation = c["evaluation"]
    if not 1 <= len(evaluation["prompts"]) <= 20 or any(not isinstance(t, str) or not t.strip() or len(t) > 1500 for t in evaluation["prompts"]):
        raise ValueError("Supply 1–20 evaluation prompts of at most 1500 characters")
    if not 0 < evaluation["temperature"] <= 1.5 or not 0 < evaluation["top_p"] <= 1:
        raise ValueError("Invalid evaluation sampling settings")
    if type(evaluation["seed"]) is not int or evaluation["seed"] < 0 or type(evaluation["max_new_tokens"]) is not int or not 128 <= evaluation["max_new_tokens"] <= 2048:
        raise ValueError("Invalid evaluation seed or token limit")
    return c


def resolved_config(dataset, directory, suite, experiment, pilot=False):
    args = SimpleNamespace(seed=suite["seed"], max_length=suite["max_length"],
                           steps=suite["steps"], validation_steps=suite["validation_steps"])
    config = shared.composed_config(dataset, directory, args, pilot)
    config["project"] = "jordanian_" + experiment["name"]
    config["trainer"]["default_root_dir"] = str(directory)
    config["trainer"]["accumulate_grad_batches"] = suite["accumulation"]
    config["trainer"]["val_check_interval"] = suite["validation_steps"]*suite["accumulation"]
    config["trainer"]["num_sanity_val_steps"] = 2
    config["trainer"]["strategy"] = "auto"
    config["data"]["_target_"] = "balanced_dataset.BalancedDataModule"
    config["data"]["num_workers"] = 0  # all speaker groups available; no worker shard omissions
    config["data"]["batch_size"] = 1
    config["train_dataset"].update(_target_="balanced_dataset.BalancedDataset", family_weights=suite["family_weights"],
                                     causal=False, skip_text_prob=0.0)
    # OmegaConf resolve already copied train_dataset into data; update that copy too.
    config["data"]["train_dataset"] = config["train_dataset"]
    config["model"]["model"]["lora_config"] = {"_target_": "fish_speech.models.text2semantic.lora.LoraConfig",
        "r": experiment["r"], "lora_alpha": experiment["alpha"], "lora_dropout": 0.0,
        "target_modules": experiment["targets"]}
    config["model"]["optimizer"]["lr"] = experiment["lr"]
    config["model"]["lr_scheduler"]["lr_lambda"]["num_warmup_steps"] = min(100, max(1, suite["steps"]//10))
    config["callbacks"]["model_checkpoint"]["save_top_k"] = 2
    config["callbacks"]["guard"] = {"_target_": "training_guard.TrainingGuard", "early_step": min(200, suite["steps"])}
    if pilot:
        config["trainer"].update(max_steps=2, val_check_interval=suite["accumulation"], limit_val_batches=2)
        config["callbacks"]["model_checkpoint"]["every_n_train_steps"] = 1
        config["callbacks"]["guard"]["early_step"] = 1
        config["model"]["lr_scheduler"]["lr_lambda"]["num_warmup_steps"] = 1
    return config


def save_config(path, config):
    import yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")


def failure_reason(error, monitor):
    message = str(error)
    logs = monitor.snapshot().get("logs", [])
    if message.startswith("Stage failed with exit code") and logs:
        # A worker traceback ends with its concrete exception. Surface that in
        # the experiment table as well as preserving the full launcher log.
        match = re.search(r"((?:\w+Error|\w+Exception): .+)$", logs[-1])
        if match:
            return match.group(1) + "; " + message
    return message


def preview_rows(rows):
    # Up to three reserves per family/split; prefer long clips to exercise memory.
    selected = []
    for family in ("talha", "podcast", "single"):
        for split in corpus.SPLITS:
            pool = [r for r in rows if r["family"] == family and r["split"] == split]
            buckets = [[r for r in pool if r["speaker"] == "wesam_ali"],
                       [r for r in pool if r["speaker"] != "wesam_ali"]] if family == "podcast" else [pool]
            for bucket in buckets:
                selected.extend(sorted(bucket, key=lambda r: (-r["duration"], r["clip_id"]))[:3])
    return selected


def prepare_pilot(rows, dataset, monitor):
    prepared = corpus.prepare(preview_rows(rows), dataset, monitor, pilot=True)
    selected = []
    for family in ("talha", "podcast", "single"):
        for split in corpus.SPLITS:
            pool = [r for r in prepared if r["family"] == family and r["split"] == split]
            if not pool:
                raise ValueError(f"No clean pilot clip for {family}/{split}; inspect QC")
            if family == "podcast":
                wesam = [r for r in pool if r["speaker"] == "wesam_ali"]
                if not wesam:
                    raise ValueError(f"No clean Wesam pilot clip in {split}")
                selected.append(wesam[0])
                guests = [r for r in pool if r["speaker"] != "wesam_ali"]
                if guests:
                    selected.append(guests[0])
            else:
                selected.append(pool[0])
    keep = {Path(r["prepared_audio"]) for r in selected}
    for wav in (dataset / "audio").rglob("*.wav"):
        if wav not in keep:
            for suffix in (".wav", ".lab", ".npy"):
                wav.with_suffix(suffix).unlink(missing_ok=True)
    write_rows(dataset / "prepared.jsonl", selected)
    return selected


def encode(dataset, rows, monitor, codec_key):
    # Compatibility bridge: only this launcher process uses multi-speaker validation.
    # Talha's source code and single-speaker behavior remain intact.
    original = talha_validation.validate_packed
    try:
        talha_validation.validate_packed = corpus.validate_packed
        shared.encode_and_pack(dataset, rows, monitor, codec_key)
    finally:
        talha_validation.validate_packed = original


def gpu_pilot(dataset, suite, experiment, monitor, base):
    config = resolved_config(dataset, Path("PILOT"), suite, experiment, True)
    pilot_data = shared.artifact_hashes(dataset, read_rows(dataset / "prepared.jsonl"))
    key = identity({"config": config, "base": base, "code": code_identity(), "data": pilot_data,
                    "versions": monitor.snapshot().get("versions")})[:20]
    directory = dataset / "gpu-pilots" / key
    receipt = read_json(directory / "passed.json")
    checkpoint = directory / "checkpoints/final.ckpt"
    if receipt and checkpoint.is_file() and receipt.get("checkpoint_sha256") == file_hash(checkpoint):
        monitor.log(f"Reusing train/save/resume pilot: {experiment['name']}")
        return
    finished = read_json(directory / "completed.json", {})
    updates = read_json(directory / "adapter_updates.json", {})
    if finished.get("step") == 3 and updates.get("changed_tensors") and checkpoint.is_file():
        atomic_json(directory / "passed.json", {"step": 3, "checkpoint_sha256": file_hash(checkpoint), "updates": updates})
        return
    config = resolved_config(dataset, directory, suite, experiment, True)
    save_config(directory / "config.yaml", config)
    monitor.update(stage="GPU pilot", experiment=experiment["name"], history=[], step=0,
                   message="Full-context forward/backward, validation, checkpoint and resume test")
    shared.process([sys.executable, HERE / "run.py", "--worker", directory / "config.yaml", "--pilot-worker"], monitor)
    config["trainer"]["max_steps"] = 3
    save_config(directory / "config.yaml", config)
    shared.process([sys.executable, HERE / "run.py", "--worker", directory / "config.yaml", "--pilot-worker"], monitor)
    finished = read_json(directory / "completed.json", {})
    updates = read_json(directory / "adapter_updates.json", {})
    if finished.get("step") != 3 or not updates.get("changed_tensors"):
        raise RuntimeError("Pilot failed save/resume or adapter-update validation")
    atomic_json(directory / "passed.json", {"step": 3, "checkpoint_sha256": file_hash(checkpoint), "updates": updates})


def code_identity():
    paths = [*HERE.glob("*.py"), TALHA / "run.py", TALHA / "pipeline.py", TALHA / "validation.py"]
    return {str(p.relative_to(HERE.parent)): file_hash(p) for p in paths}


def train(dataset, directory, suite, experiment, monitor):
    config = resolved_config(dataset, directory, suite, experiment)
    config_path = directory / "config.yaml"
    save_config(config_path, config)
    finished = read_json(directory / "completed.json")
    valid = bool(finished and finished.get("step", 0) >= suite["steps"]
                 and Path(finished["chosen_checkpoint"]).is_file() and Path(finished["final_checkpoint"]).is_file())
    hashes = read_json(directory / "completed_hashes.json", {})
    if valid:
        try:
            valid = bool(hashes and all(file_hash(p) == sha for p, sha in hashes.items()))
        except OSError:
            valid = False
    monitor.update(stage="training", experiment=experiment["name"], run_dir=str(directory), history=[], step=0,
                   max_steps=suite["steps"], message="Training balanced Jordanian adapter")
    if valid:
        monitor.log(f"Completed experiment reused: {experiment['name']}")
    else:
        shared.process([sys.executable, HERE / "run.py", "--worker", config_path], monitor)
    finished = read_json(directory / "completed.json", {})
    if finished.get("step", 0) < suite["steps"] or not Path(finished.get("final_checkpoint", "MISSING")).is_file():
        raise RuntimeError("Experiment did not produce its completion checkpoint")
    atomic_json(directory / "completed_hashes.json", {p: file_hash(p) for p in
                {finished["chosen_checkpoint"], finished["final_checkpoint"]}})
    candidates = [("early", directory / "checkpoints/early.ckpt"),
                  ("val_best", Path(finished["chosen_checkpoint"])),
                  ("final", Path(finished["final_checkpoint"]))]
    # Compare distinct weight states once; retain aliases for best==final.
    seen, models = {}, []
    for label, checkpoint in candidates:
        if not checkpoint.is_file():
            continue
        sha = file_hash(checkpoint)
        if sha in seen:
            seen[sha]["aliases"].append(label)
            continue
        model = {"name": experiment["name"] + "_" + label, "experiment": experiment["name"],
                 "checkpoint": str(checkpoint), "checkpoint_sha256": sha, "config": str(config_path), "aliases": [label]}
        models.append(model)
        seen[sha] = model
    return finished, models


def compare_models(campaign, models, monitor):
    comparison = campaign / "comparison"
    manifest = read_json(comparison / "models.json", [])
    for model in models:
        monitor.update(stage="comparison", experiment=model["name"],
                       message="Generating matched references, prompts and seed")
        output = comparison / model["name"]
        command = [sys.executable, HERE / "compare.py", "--cases", comparison / "cases.jsonl", "--output", output]
        if model.get("checkpoint"):
            command += ["--config", model["config"], "--checkpoint", model["checkpoint"]]
        try:
            shared.process(command, monitor)
            results = read_rows(output / "results.jsonl")
            item = {**model, "results": results, "status": "complete" if all(r["status"] == "complete" for r in results) else "sample failures"}
        except RuntimeError as error:
            item = {**model, "status": "failed", "error": str(error),
                    "results": read_rows(output / "results.jsonl") if (output / "results.jsonl").exists() else []}
            monitor.log(f"Comparison failed for {model['name']}; preserving report and continuing")
        manifest = [m for m in manifest if m["name"] != model["name"]] + [item]
        atomic_json(comparison / "models.json", manifest)
        monitor.update(models=manifest)
    return manifest


def check_resources(rows, suite, monitor):
    hours = sum(r["duration"] for r in rows)/3600
    # PCM audio + integer codec arrays + group Python objects are bounded by the selection.
    estimated_data_gb = hours*.5
    estimated_disk_gb = 15 + hours*.6 + len(suite["experiments"])*2
    free = shutil.disk_usage(HERE).free/2**30
    if free < estimated_disk_gb + 20:
        raise RuntimeError(f"Need {estimated_disk_gb+20:.1f} GiB free including reserve; have {free:.1f}")
    meminfo = Path("/proc/meminfo").read_text()
    available = int(re.search(r"MemAvailable:\s+(\d+)", meminfo).group(1))*1024/2**30
    if available < 20 + estimated_data_gb:
        raise RuntimeError(f"Available host RAM {available:.1f} GiB below estimated requirement {20+estimated_data_gb:.1f}")
    monitor.update(resource_estimates={"selected_hours": hours, "data_ram_gb": estimated_data_gb,
                   "disk_gb": estimated_disk_gb, "free_disk_gb": free, "available_ram_gb": available})


def run_suite(args, monitor):
    suite = load_config(args.config)
    if args.steps:
        suite["steps"] = args.steps
        # Revalidate overrides with the same rules.
        if not 1 <= args.steps <= 10000:
            raise ValueError("--steps must be 1–10000")
    if args.only:
        names = set(args.only.split(","))
        available = {e["name"] for e in suite["experiments"]}
        if not names <= available:
            raise ValueError(f"Unknown experiments: {names-available}")
        suite["experiments"] = [e for e in suite["experiments"] if e["name"] in names]
    if args.install:
        shared.install_dependencies(monitor)
    if args.check:
        shared.preflight(monitor)
        monitor.update(status="checked", message="Prerequisites passed; no training started")
        return
    monitor.update(stage="planning", status="running", message="Inventory and bounded recording-disjoint selection")
    dataset, rows, summary = corpus.plan(suite, args.review, args.accepted_only)
    monitor.update(dataset=str(dataset), selected=len(rows), splits=summary, suite=suite)
    check_resources(rows, suite, monitor)
    if args.plan:
        monitor.update(status="planned", message="Plan written; no audio preparation, codec extraction or training started")
        return
    if args.preview:
        prepared = prepare_pilot(rows, dataset / "preview", monitor)
        monitor.update(status="preview complete", references=prepared, message="Real CPU previews including each source and split; no GPU work started")
        return
    shared.preflight(monitor)
    base = shared.base_signature()
    codec_key = identity({"codec_sha256": base["codec.pth"], "upstream": talha_pipeline.PIN})
    # Campaign excludes experiment subset so --only can fill the same comparison.
    settings = {k: v for k, v in suite.items() if k not in {"experiments", "evaluation"}}
    key = identity({"dataset": dataset.name, "base": base, "settings": settings,
                    "code": code_identity(), "versions": monitor.snapshot().get("versions")})[:24]
    campaign = HERE / "suites" / key
    campaign.mkdir(parents=True, exist_ok=True)
    monitor.update(campaign=str(campaign))
    atomic_json(campaign / "settings.json", {"settings": settings, "evaluation": suite["evaluation"],
                                           "base": base, "dataset": str(dataset)})
    existing = read_json(campaign / "status.json", {"experiments": {}})
    states = existing["experiments"]
    pilots = []
    pilot_dataset = dataset / "pilot-data"
    if not args.compare_only:
        pilot_rows = prepare_pilot(rows, pilot_dataset, monitor)
        encode(pilot_dataset, pilot_rows, monitor, codec_key)
        # Codec round-trip precedes every experiment; caches are small and shared.
        reconstruction = pilot_dataset / "codec-reconstruction.wav"
        reference = Path(pilot_rows[0]["prepared_audio"])
        shared.process([sys.executable, shared.UPSTREAM / "fish_speech/models/dac/inference.py",
                        "--input-path", reference.with_suffix(".npy"), "--output-path", reconstruction,
                        "--config-name", "modded_dac_vq", "--checkpoint-path", BASE / "codec.pth"], monitor)
        import numpy as np
        import soundfile as sf
        decoded, rate = sf.read(reconstruction)
        if rate != 44100 or decoded.ndim != 1 or not np.isfinite(decoded).all() or np.sqrt(np.mean(decoded**2)) < 1e-5 or abs(len(decoded)/rate-pilot_rows[0]["duration"]) > .1:
            raise ValueError("Codec round-trip failed")
        monitor.update(references=pilot_rows, codec_audio=str(reconstruction))
        for experiment in suite["experiments"]:
            try:
                gpu_pilot(pilot_dataset, suite, experiment, monitor, base)
                pilots.append(experiment)
            except (RuntimeError, ValueError, OSError) as error:
                states[experiment["name"]] = {"status": "pilot failed", "error": failure_reason(error, monitor)}
                atomic_json(campaign / "status.json", {"experiments": states})
                monitor.update(experiments=states)
                monitor.log(f"Pilot failed: {experiment['name']}; remaining experiments continue")
        if not pilots:
            raise RuntimeError("All GPU pilots failed; full dataset preparation was skipped")
        prepared = corpus.prepare(rows, dataset, monitor)
        encode(dataset, prepared, monitor, codec_key)
    else:
        prepared = read_rows(dataset / "prepared.jsonl")
        corpus.validate_packed(dataset)
        pilots = suite["experiments"]
    # Identity includes accepted/QC-filtered audio and token/proto contents. Repaired
    # or rejected data never silently resume an optimizer trained on another corpus.
    data_identity = identity({"rows": prepared, "artifacts": shared.artifact_hashes(dataset, prepared),
                              "wav_hashes": [file_hash(r["prepared_audio"]) for r in prepared]})
    comparison = campaign / "comparison"
    comparison.mkdir(exist_ok=True)
    atomic_json(comparison / "base_identity.json", base)
    write_rows(comparison / "cases.jsonl", cases_for(dataset, suite["evaluation"]))
    monitor.update(references=list({c["speaker"]: c["reference"] for c in cases_for(dataset, suite["evaluation"])}.values()))
    compare_models(campaign, [{"name": "base", "aliases": ["original S2 Pro"]}], monitor)
    for experiment in pilots:
        run_key = identity({"data": data_identity, "experiment": experiment})[:16]
        directory = campaign / "runs" / (experiment["name"] + "_" + run_key)
        try:
            if args.compare_only and not (directory / "completed.json").is_file():
                raise RuntimeError("No completed experiment exists for these exact settings and data")
            finished, models = train(dataset, directory, suite, experiment, monitor) if not args.compare_only else completed_models(directory, experiment)
            states[experiment["name"]] = {"status": "trained", "run_dir": str(directory), **finished}
            atomic_json(campaign / "status.json", {"experiments": states})
            monitor.update(experiments=states)
            compare_models(campaign, models, monitor)
        except (RuntimeError, ValueError, OSError) as error:
            states[experiment["name"]] = {"status": "failed", "run_dir": str(directory), "error": failure_reason(error, monitor)}
            monitor.log(f"Experiment failed: {experiment['name']}; the next experiment will run")
        atomic_json(campaign / "status.json", {"experiments": states})
        monitor.update(experiments=states)
    models = read_json(comparison / "models.json", [])
    complete = all(states.get(e["name"], {}).get("status") == "trained" for e in suite["experiments"])
    samples_ok = all(m["status"] == "complete" for m in models)
    status = "complete" if complete and samples_ok else "complete with failures"
    atomic_json(campaign / "status.json", {"status": status, "experiments": states,
                "comparison": str(comparison), "automatic_quality_ranking": False})
    monitor.update(status=status, stage="finished", message="Matched samples ready; listen and select a checkpoint. Waveform metrics do not rank accent quality.")


def completed_models(directory, experiment):
    finished = read_json(directory / "completed.json", {})
    checkpoints = [("early", directory / "checkpoints/early.ckpt"),
                   ("val_best", Path(finished.get("chosen_checkpoint", "MISSING"))),
                   ("final", Path(finished.get("final_checkpoint", "MISSING")))]
    if not finished or not checkpoints[-1][1].is_file():
        raise RuntimeError("Missing completion checkpoint")
    models = [{"name": experiment["name"]+"_"+label, "experiment": experiment["name"],
               "checkpoint": str(path), "checkpoint_sha256": file_hash(path),
               "config": str(directory / "config.yaml"), "aliases": [label]}
              for label, path in checkpoints if path.is_file()]
    return finished, models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "experiments.json")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--only", help="Comma-separated experiment names")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--review", type=Path, help="Downloaded Wesam review decisions")
    parser.add_argument("--accepted-only", action="store_true", help="Only explicitly accepted podcast clips")
    parser.add_argument("--install", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    for mode in ("plan", "preview", "check", "compare-only", "dashboard-only"):
        modes.add_argument("--"+mode, action="store_true")
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--exit-on-finish", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--pilot-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        # The shared worker provides fixed validation, trusted full-state resume and metrics.
        original = shared.emit
        def emit(**values):
            if values.get("message"):
                values["message"] = values["message"].replace("Talha", "Jordanian").replace("single-speaker", "balanced multi-speaker")
            original(**values)
        shared.emit = emit
        from training_compat import non_reentrant_checkpointing
        with non_reentrant_checkpointing():
            original(message="Using non-reentrant activation checkpointing for frozen-embedding LoRA")
            shared.worker(args.worker, args.pilot_worker)
        return 0
    if args.accepted_only and not args.review:
        parser.error("--accepted-only requires --review")
    runtime = HERE / "monitor"
    runtime.mkdir(exist_ok=True)
    with ExitStack() as stack:
        if not args.dashboard_only:
            for lock_path in (runtime / "run.lock", TALHA / "monitor/run.lock"):
                lock = stack.enter_context(lock_path.open("a"))
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    print(f"A training launcher already holds {lock_path}; stop it before this suite", file=sys.stderr)
                    return 1
        saved = read_json(runtime / "status.json", {})
        monitor = shared.Monitor(runtime)
        server = None
        if args.dashboard_only:
            monitor.state.update(saved)
        if not args.no_dashboard:
            from dashboard import serve
            try:
                server = serve(monitor, args.host, args.port)
            except OSError as error:
                monitor.log(f"Cannot bind dashboard {args.port}: {error}; use --port for another port")
                monitor.logfile.close()
                return 1
            if not args.dashboard_only:
                monitor.update(dashboard_port=server.server_port)
            monitor.log(f"Dashboard: http://<server-ip>:{server.server_port}")
        code = 0
        try:
            if not args.dashboard_only:
                run_suite(args, monitor)
                if monitor.snapshot().get("status") == "complete with failures":
                    code = 1
        except KeyboardInterrupt:
            monitor.update(status="interrupted", message="Stopped; rerun the same command to resume")
            code = 130
        except Exception as error:
            traceback.print_exc()
            monitor.update(status="failed", error=str(error), message=str(error))
            code = 1
        try:
            if server and not args.exit_on_finish:
                monitor.log("Dashboard remains available. Ctrl-C exits; checkpoints and samples are preserved.")
                while True:
                    time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            if server:
                server.shutdown()
                server.server_close()
            monitor.logfile.close()
        return code


if __name__ == "__main__":
    raise SystemExit(main())
