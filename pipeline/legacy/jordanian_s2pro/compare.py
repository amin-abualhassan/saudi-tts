"""Matched reference-conditioned synthesis, one model per isolated GPU process."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

from common import BASE, UPSTREAM, atomic_json, file_hash, identity, read_json, read_rows, talha_pipeline


def audio_metrics(audio, rate):
    import numpy as np
    audio = np.asarray(audio)
    if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("Empty, non-mono or nonfinite synthesis")
    rms = float(np.sqrt(np.mean(audio**2)))
    width = max(1, int(.02*rate))
    frames = [float(np.sqrt(np.mean(audio[i:i+width]**2))) for i in range(0, len(audio), width)]
    threshold = max(.001, rms*.05)
    quiet = [f < threshold for f in frames]
    trailing = 0
    for value in reversed(quiet):
        if not value:
            break
        trailing += 1
    return {"duration": len(audio)/rate, "rms": rms, "peak": float(np.max(np.abs(audio))),
            "quiet_fraction": sum(quiet)/len(quiet), "trailing_quiet_seconds": trailing*.02,
            "warning": "near-silent output" if rms < .001 else "mostly quiet output" if sum(quiet)/len(quiet) > .8 else None}


def cases_for(dataset, evaluation):
    rows = read_rows(dataset / "prepared.jsonl")
    cases = []
    for speaker in ("talha_ahmad", "wesam_ali", "single_speaker"):
        references = [r for r in rows if r["split"] == "train" and r["speaker"] == speaker]
        if not references:
            raise ValueError(f"No training reference for {speaker}")
        ref = min(references, key=lambda r: (abs(r["duration"]-12), r["clip_id"]))
        tests = [r for r in rows if r["split"] == "test" and r["speaker"] == speaker]
        prompts = list(evaluation["prompts"])
        if tests:
            prompts.append(min(tests, key=lambda r: (abs(r["duration"]-10), r["clip_id"]))["text"])
        for i, text in enumerate(prompts):
            cases.append({"id": f"{speaker}_{i}", "speaker": speaker, "text": text,
                          "held_out_transcript": i >= len(evaluation["prompts"]), "reference": ref,
                          "generation": {k: v for k, v in evaluation.items() if k != "prompts"}})
    return cases


def synthesize(config_path, checkpoint, cases_path, output):
    sys.path.insert(0, str(UPSTREAM))
    import numpy as np
    import soundfile as sf
    import torch
    import yaml
    from fish_speech.models.text2semantic.llama import DualARTransformer
    from fish_speech.models.text2semantic.lora import LoraConfig
    from fish_speech.models.text2semantic.inference import decode_one_token_ar, generate_long, load_codec_model, decode_to_audio

    cases = read_rows(cases_path)
    checkpoint = Path(checkpoint) if checkpoint else None
    config = yaml.safe_load(Path(config_path).read_text()) if config_path else None
    checkpoint_hash = file_hash(checkpoint) if checkpoint else None
    model_identity = {"checkpoint": checkpoint_hash, "base": read_json(output.parent / "base_identity.json"),
                      "lora": config["model"]["model"]["lora_config"] if config else None,
                      "implementation": file_hash(Path(__file__))}
    output.mkdir(parents=True, exist_ok=True)
    previous = {r["id"]: r for r in read_rows(output / "results.jsonl")} if (output / "results.jsonl").exists() else {}
    expected = {c["id"]: identity({"model": model_identity, "case": c,
                "reference_audio": file_hash(c["reference"]["prepared_audio"]),
                "reference_tokens": file_hash(Path(c["reference"]["prepared_audio"]).with_suffix(".npy"))}) for c in cases}
    results = []
    pending = []
    for c in cases:
        old = previous.get(c["id"])
        wav = output / (c["id"] + ".wav")
        if (old and old.get("identity") == expected[c["id"]] and old.get("wav_sha256")
                and wav.is_file() and old["wav_sha256"] == file_hash(wav)):
            results.append(old)
        else:
            pending.append(c)
    if not pending:
        print("All matched samples already verified", flush=True)
        return
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Comparison requires CUDA and BF16")
    kwargs = {}
    if config:
        settings = dict(config["model"]["model"]["lora_config"])
        settings.pop("_target_", None)
        kwargs["lora_config"] = LoraConfig(**settings)
    model = DualARTransformer.from_pretrained(str(BASE), load_weights=True, max_length=4096, **kwargs)
    if checkpoint:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        adapters = {n.removeprefix("model."): t for n, t in saved["state_dict"].items()}
        state = model.state_dict()
        expected_names = {n for n in state if "lora_" in n}
        if set(adapters) != expected_names or any(t.shape != state[n].shape or not torch.isfinite(t).all() for n, t in adapters.items()):
            raise ValueError("Comparison checkpoint does not match its saved configuration")
        model.load_state_dict(state | adapters, strict=True)
        del saved, state, adapters
    model = model.to(device="cuda", dtype=torch.bfloat16).eval()
    codec = load_codec_model(BASE / "codec.pth", "cuda", torch.bfloat16)
    for case in pending:
        started = time.monotonic()
        record = {"id": case["id"], "speaker": case["speaker"], "text": case["text"],
                  "identity": expected[case["id"]], "reference_clip": case["reference"]["clip_id"]}
        try:
            ref = case["reference"]
            ref_audio = Path(ref["prepared_audio"])
            receipt = read_json(ref_audio.parents[3] / "receipts" / (ref["clip_id"] + ".json"), {})
            if receipt.get("wav_sha256") != file_hash(ref_audio) or receipt.get("text") != ref["text"]:
                raise ValueError("Reference provenance changed")
            talha_pipeline.validate_tokens(ref_audio.with_suffix(".npy"), ref["duration"])
            tokens = torch.from_numpy(np.load(ref_audio.with_suffix(".npy"), allow_pickle=False)).long().cpu()
            options = case["generation"]
            torch.manual_seed(options["seed"])
            torch.cuda.manual_seed_all(options["seed"])
            codes = []
            with torch.inference_mode():
                for response in generate_long(model=model, device="cuda", decode_one_token=decode_one_token_ar,
                        text="<|speaker:0|>"+case["text"], num_samples=1, max_new_tokens=options["max_new_tokens"],
                        temperature=options["temperature"], top_p=options["top_p"], top_k=30,
                        compile=False, iterative_prompt=True, chunk_length=200,
                        prompt_tokens=[tokens], prompt_text=[ref["text"]]):
                    if response.action == "sample":
                        codes.append(response.codes)
                if not codes:
                    raise ValueError("No generated audio tokens")
                generated = torch.cat(codes, dim=1)
                audio = decode_to_audio(generated.to("cuda"), codec).float().cpu().numpy()
            record.update(audio_metrics(audio, codec.sample_rate))
            record["generated_frames"] = generated.shape[1]
            # Metrics describe raw output; limit only the playback peak.
            audio = audio/max(1, float(np.max(np.abs(audio)))/.98)
            wav = output / (case["id"] + ".wav")
            temp = wav.with_suffix(".tmp.wav")
            sf.write(temp, audio, codec.sample_rate, subtype="PCM_16")
            temp.replace(wav)
            record.update(wav=str(wav), wav_sha256=file_hash(wav), status="complete")
        except Exception as error:
            traceback.print_exc()
            record.update(status="failed", error=str(error))
            if isinstance(error, torch.cuda.OutOfMemoryError):
                atomic_json(output / "fatal.json", record)
                raise
        record["elapsed"] = time.monotonic()-started
        results.append(record)
        from common import write_rows
        write_rows(output / "results.jsonl", results)
        print(json.dumps({k: v for k, v in record.items() if k != "text"}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.config) != bool(args.checkpoint):
        parser.error("Pass both --config and --checkpoint, or neither for the base model")
    synthesize(args.config, args.checkpoint, args.cases, args.output)


if __name__ == "__main__":
    main()
