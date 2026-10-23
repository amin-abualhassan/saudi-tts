#!/usr/bin/env python3
"""Regression tests for ops/guard.py. Run: python3 ops/test_guard.py  (prints n/n passed)."""
import json
import subprocess
import sys
from pathlib import Path

GUARD = Path(__file__).with_name("guard.py")

BLOCK = [
    "ssh Majed-AWS rm -rf /opt/dlami/nvme/training",
    "ssh Majed-AWS 'ls /opt; touch x'",
    "ssh Majed-AWS",
    "rsync -av local/ Majed-AWS:/opt/dlami/nvme/",
    "rsync -av --remove-source-files Majed-AWS:/opt/x /tmp/x",
    "aws ec2 stop-instances --instance-ids i-05a3bb80bb4f85759",
    "sudo shutdown -h now",
    "rm -rf /opt/dlami/nvme",
    "rm -rf ~/saudi-tts/",
    "rm -r /opt/dlami/nvme/data/saudi_eval_v1",
    "aws s3 rm --recursive s3://aljawab-asr-data/Casablanca/",
    "aws s3 sync /x s3://aljawab-asr-data/Jordan/ --delete",
    "tmux kill-session -t agent",
    "pkill -f claude",
    "cat ~/.secrets/api-keys.txt",
    "grep HF ~/.secrets/api-keys.txt",
    "source ~/.secrets/fallback.env",
    "curl -s https://api.elevenlabs.io/v1/voices",
    "hf download some-org/podcast-dataset --repo-type dataset",
    "huggingface-cli download musabalosimi/saudi_dialect_asrv1.0",
    "wget https://example.com/emilia.tar",
    "curl -L https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0/resolve/main/x.tar",
    "yt-dlp https://www.youtube.com/watch?v=abc",
    "pkill -f train.py",
]
ALLOW = [
    "ssh Majed-AWS ls -la /opt/dlami/nvme",
    "ssh Majed-AWS cat /opt/dlami/nvme/AGENTS.md",
    "rsync -av Majed-AWS:/opt/dlami/nvme/Splitter/ pipeline/legacy/Splitter/",
    "aws s3 sync reports/ s3://aljawab-asr-data/saudi-tts/reports/",
    "aws s3 sync /x s3://aljawab-asr-data/saudi-tts/ckpt/ --delete",
    "ls -l ~/.secrets/",
    "python3 pipeline/fetch.py omnilingual",
    "hf download fishaudio/s2-pro --local-dir /opt/dlami/nvme/models/s2-pro",
    "huggingface-cli download facebook/omnilingual-asr-corpus --repo-type dataset",
    "curl -LsSf https://astral.sh/uv/install.sh",
    "git clone https://github.com/fishaudio/fish-speech.git",
    "wget https://huggingface.co/k2-fsa/OmniVoice/resolve/main/config.json",
    "kill 999999",
    "nvidia-smi",
    "git status",
]


def run(cmd: str) -> int:
    p = subprocess.run([sys.executable, str(GUARD)], input=json.dumps({"tool_input": {"command": cmd}}),
                       capture_output=True, text=True)
    return p.returncode


def main() -> int:
    fails = []
    for cmd in BLOCK:
        if run(cmd) != 2:
            fails.append(f"should BLOCK: {cmd}")
    for cmd in ALLOW:
        if run(cmd) != 0:
            fails.append(f"should ALLOW: {cmd}")
    n = len(BLOCK) + len(ALLOW)
    for f in fails:
        print("FAIL", f)
    print(f"{n - len(fails)}/{n} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
