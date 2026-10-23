"""Reuse the pinned Talha utilities without changing its launcher or artifacts."""
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TALHA = HERE.parent / "talha_s2pro"
UPSTREAM = HERE.parent / "fish-speech-upstream"
BASE = HERE.parent / "checkpoints/s2-pro"

# Talha's dependency-light utility modules use adjacent absolute imports.
sys.path.insert(0, str(TALHA))
import pipeline as talha_pipeline
import validation as talha_validation

spec = importlib.util.spec_from_file_location("talha_runner_shared", TALHA / "run.py")
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
sys.path.insert(0, str(HERE))

atomic_json = shared.atomic_json
file_hash = shared.file_hash
identity = shared.identity
read_json = shared.read_json


def read_rows(path):
    import json
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_rows(path, rows):
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    temp.replace(path)
