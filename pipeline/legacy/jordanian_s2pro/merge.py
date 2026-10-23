#!/usr/bin/env python3
"""Export a selected comparison checkpoint using its exact saved LoRA settings."""
import argparse
from pathlib import Path
import sys
from common import TALHA, read_json, file_hash, shared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Comparison model name, e.g. fast_r32_early")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    models = read_json(args.suite / "comparison/models.json", [])
    selected = next((m for m in models if m["name"] == args.model), None)
    if not selected or not selected.get("checkpoint"):
        parser.error("Select a trained model name from the comparison")
    checkpoint = Path(selected["checkpoint"])
    if file_hash(checkpoint) != selected["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed since comparison; regenerate samples first")
    if args.output.exists():
        raise ValueError("Choose a new output directory; existing models are preserved")
    monitor = shared.Monitor(args.suite / "export-log")
    try:
        shared.process([sys.executable, TALHA / "merge_model.py", "--experiment-config", selected["config"],
                        "--checkpoint", checkpoint, "--output", args.output.resolve()], monitor)
    finally:
        monitor.logfile.close()


if __name__ == "__main__":
    main()
