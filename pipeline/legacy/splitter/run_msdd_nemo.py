#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from nemo.collections.asr.models.msdd_models import NeuralDiarizer


def build_manifest(audio_path: Path, manifest_path: Path, num_speakers: int) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "audio_filepath": str(audio_path.resolve()),
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "num_speakers": num_speakers,
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def build_cfg(
    manifest_path: Path,
    out_dir: Path,
    batch_size: int,
    infer_batch_size: int,
    sigmoid_threshold: float,
    num_speakers: int,
):
    cfg = OmegaConf.create(
        {
            "name": "NeuralDiarizer",
            "num_workers": 1,
            "sample_rate": 16000,
            "batch_size": batch_size,
            "device": "cuda",
            "verbose": True,
            "diarizer": {
                "manifest_filepath": str(manifest_path.resolve()),
                "out_dir": str(out_dir.resolve()),
                "oracle_vad": False,
                "collar": 0.25,
                "ignore_overlap": False,
                "vad": {
                    "model_path": "vad_multilingual_marblenet",
                    "external_vad_manifest": None,
                    "parameters": {
                        "window_length_in_sec": 0.15,
                        "shift_length_in_sec": 0.01,
                        "smoothing": "median",
                        "overlap": 0.5,
                        "onset": 0.1,
                        "offset": 0.1,
                        "pad_onset": 0.1,
                        "pad_offset": 0.0,
                        "min_duration_on": 0.0,
                        "min_duration_off": 0.2,
                        "filter_speech_first": True,
                    },
                },
                "speaker_embeddings": {
                    "model_path": "titanet_large",
                    "parameters": {
                        "window_length_in_sec": [1.5, 1.25, 1.0, 0.75, 0.5],
                        "shift_length_in_sec": [0.75, 0.625, 0.5, 0.375, 0.25],
                        "multiscale_weights": [1, 1, 1, 1, 1],
                        "save_embeddings": True,
                    },
                },
                "clustering": {
                    "parameters": {
                        "oracle_num_speakers": True,
                        "max_num_speakers": num_speakers,
                        "enhanced_count_thres": 80,
                        "max_rp_threshold": 0.25,
                        "sparse_search_volume": 30,
                        "maj_vote_spk_count": False,
                        "chunk_cluster_count": 50,
                        "embeddings_per_chunk": 10000,
                    },
                },
                "msdd_model": {
                    "model_path": "diar_msdd_telephonic",
                    "parameters": {
                        "use_speaker_model_from_ckpt": True,
                        "infer_batch_size": infer_batch_size,
                        "sigmoid_threshold": [sigmoid_threshold],
                        "seq_eval_mode": False,
                        "split_infer": True,
                        "diar_window_length": 50,
                        "overlap_infer_spk_limit": 8,
                    },
                },
            },
        }
    )
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to mono 16k wav")
    parser.add_argument("--out_dir", default="./output")
    parser.add_argument("--num_speakers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--infer_batch_size", type=int, default=64)
    parser.add_argument("--sigmoid_threshold", type=float, default=0.7)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in PyTorch. Fix your CUDA/PyTorch install first.")

    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "input_manifest.json"
    build_manifest(audio_path, manifest_path, args.num_speakers)

    cfg = build_cfg(
        manifest_path=manifest_path,
        out_dir=out_dir,
        batch_size=args.batch_size,
        infer_batch_size=args.infer_batch_size,
        sigmoid_threshold=args.sigmoid_threshold,
        num_speakers=args.num_speakers,
    )

    print("Using device:", cfg.device)
    print("Output dir:", out_dir)
    print("Manifest:", manifest_path)

    diarizer_model = NeuralDiarizer(cfg=cfg).to(cfg.device)
    diarizer_model.diarize()

    print("\nDone.")
    print("RTTM folder:", out_dir / "pred_rttms")
    print("Speaker-label JSONs / intermediate files are under:", out_dir)


if __name__ == "__main__":
    main()