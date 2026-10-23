#!/usr/bin/env python3
"""Starter PreToolUse guard for the Saudi TTS agent (Bash tool).

Exit 0 = allow, exit 2 = block (the message on stderr is shown to the agent).
The agent may EXTEND this file (ElevenLabs cap, dataset allowlist, training-PID
protection) but must never remove or loosen the rules below.
"""
import json
import re
import sys

try:
    cmd = json.load(sys.stdin).get("tool_input", {}).get("command", "") or ""
except Exception:
    sys.exit(0)  # never break the agent on malformed input

c = " ".join(cmd.split())
# Treat any direct address of the TTS-FT box like the alias.
c = re.sub(r"(ubuntu@)?(ec2-98-83-42-166\.compute-1\.amazonaws\.com|98\.83\.42\.166)", "Majed-AWS", c)


def block(why: str) -> None:
    print(f"BLOCKED by ops/guard.py: {why}", file=sys.stderr)
    sys.exit(2)


# 1. The TTS-FT server (Majed-AWS) is strictly read-only.
READ_ONLY = {"ls", "cat", "find", "du", "df", "head", "tail", "stat", "wc", "file", "tree", "nvidia-smi", "grep", "md5sum", "sha256sum"}
FORBIDDEN_REMOTE = re.compile(
    r"[;&>`]|\$\(|\b(rm|mv|cp|kill|pkill|sudo|pip|uv|conda|python3?|bash|sh|zsh|tee|dd|chmod|chown|truncate|git|docker|tmux|nohup|systemctl|apt|wget|curl|rsync|scp|touch|mkdir|ln)\b"
)
for m in re.finditer(r"\bssh\b[^|;&]*?\bMajed-AWS\b\s*(.*)", c):
    remote = m.group(1).strip()
    words = remote.strip("'\"").split()
    if not words:
        block("interactive SSH to Majed-AWS is not allowed; run one read-only command (ls/cat/find/du/head/tail/stat/grep)")
    if words[0] not in READ_ONLY or FORBIDDEN_REMOTE.search(remote):
        block("Majed-AWS is read-only: only ls/cat/find/du/df/head/tail/stat/wc/grep, no redirects or chaining; pull files with rsync Majed-AWS:<path> <local>")
for m in re.finditer(r"\b(rsync|scp)\b([^|;&]*)", c):
    args = m.group(2)
    toks = args.split()
    if toks and toks[-1].startswith("Majed-AWS:"):
        block("never copy files TO Majed-AWS; only pull from it")
    if "Majed-AWS:" in args and "--remove-source-files" in args:
        block("--remove-source-files on Majed-AWS would delete its data")

# 2. Never touch the instances' lifecycle.
if re.search(r"\baws\b.*\bec2\b.*\b(stop|terminate|reboot|modify)-instance", c):
    block("instance lifecycle changes are Amin's call")
if re.search(r"\b(sudo\s+)?(shutdown|poweroff|halt)\b|\bsudo\s+reboot\b|\bsystemctl\s+(poweroff|reboot|halt)\b|\binit\s+[06]\b", c):
    block("no shutdown/reboot of the server")

# 3. Never delete the roots of data, code, secrets, or the eval set.
ROOTS = r"(/|/opt/dlami/nvme|/opt/dlami/nvme/data|~|\$HOME|/home/ubuntu|~/saudi-tts|/home/ubuntu/saudi-tts|~/\.secrets|/home/ubuntu/\.secrets|~/\.ssh|/home/ubuntu/\.ssh|~/\.claude|/home/ubuntu/\.claude)/?"
if re.search(r"\brm\b(?=[^|;&]*\s-\S*[rR])[^|;&]*\s" + ROOTS + r"(\s|$)", c):
    block("refusing recursive delete of a root directory (data, code, secrets, ssh, claude config)")
if re.search(r"\brm\b[^|;&]*saudi_eval_v1", c) or re.search(r"\bmv\b[^|;&]*saudi_eval_v1", c):
    block("saudi_eval_v1 is frozen; never delete or move it")
if re.search(r"\baws\b.*\bs3\b.*\b(rm|rb|mv)\b", c) and "s3://aljawab-asr-data/saudi-tts/" not in c:
    block("S3 deletes/moves are only allowed under s3://aljawab-asr-data/saudi-tts/")
if re.search(r"\baws\b.*\bs3\b.*\bsync\b.*--delete", c) and not re.search(r"\s\S+\s+s3://aljawab-asr-data/saudi-tts/\S*\s.*--delete|--delete.*\s\S+\s+s3://aljawab-asr-data/saudi-tts/", c):
    block("aws s3 sync --delete is only allowed with a destination under s3://aljawab-asr-data/saudi-tts/")

# 4. Don't kill the agent's own harness.
if re.search(r"\btmux\s+kill-server\b|\btmux\s+kill-session\b[^|;&]*-t\s*(agent|watchdog)\b", c):
    block("the agent and watchdog tmux sessions must stay up")
if re.search(r"\b(pkill|killall)\b[^|;&]*\bclaude\b", c):
    block("do not kill Claude Code processes")

# 5. Secrets stay on this box.
if re.search(r"\.secrets/", c) and re.search(r"\b(curl|wget|scp|rsync|aws\s+s3|git\s+add|gh)\b", c):
    block("secrets must not be uploaded, committed or sent anywhere")

# ---------------------------------------------------------------------------
# Extensions added by the agent on 2026-10-23 (see DECISIONS.md). Additive only.
# ---------------------------------------------------------------------------
import os
from pathlib import Path

REPO = Path(os.path.expanduser("~/saudi-tts"))

# 6. Secrets are never printed: no content-reading tool may touch ~/.secrets.
if re.search(r"(^|[\s/'\"=:])\.secrets\b", c) and re.search(
    r"\b(cat|head|tail|less|more|grep|egrep|rg|sed|awk|cut|strings|xxd|od|hexdump|base64|echo|printf|source|env|printenv|diff|cmp|tee|cp|mv|vim?|nano)\b|(^|[;&|]\s*)\.\s", c
):
    block("~/.secrets is never read in a shell; Python code reads keys via pipeline/keys.py without printing them")

# 7. ElevenLabs: only through pipeline/elevenlabs_client.py (it logs spend), and never past the cap.
if re.search(r"elevenlabs\.io", c, re.I) and re.search(r"\b(curl|wget|http|httpie)\b", c):
    block("call ElevenLabs only through pipeline/elevenlabs_client.py (spend logging + cap)")
try:
    spend = json.load(open(REPO / "elevenlabs_spend.json"))
    over = float(spend.get("spent_usd", 0)) >= float(spend.get("cap_usd", 50)) or (
        "credits_remaining_at_start" in spend
        and float(spend.get("credits_used", 0)) >= float(spend["credits_remaining_at_start"]) - 500)
except Exception:
    over = False
if over and re.search(r"elevenlabs", c, re.I) and not re.search(r"--dry-run|--report-only", c):
    block("ElevenLabs spend has reached the cap in elevenlabs_spend.json; ask Amin in QUESTIONS.md + Notion before going further")

# 8. Downloads: hosts and Hugging Face repos must be on ops/allowlist.txt.
def _allow():
    hosts, repos = set(), []
    try:
        for line in open(REPO / "ops" / "allowlist.txt"):
            line = line.split("#", 1)[0].strip()
            if line.startswith("host "):
                hosts.add(line[5:].strip().lower())
            elif line.startswith("hf "):
                repos.append(line[3:].strip())
    except Exception:
        pass
    return hosts, repos

def _repo_ok(repo, repos):
    for pat in repos:
        if pat.endswith("/*") and repo.lower().startswith(pat[:-1].lower()):
            return True
        if repo.lower() == pat.lower():
            return True
    return False

HOSTS, REPOS = _allow()
for m in re.finditer(r"\b(?:huggingface-cli|hf)\s+download\s+(?:--\S+\s+\S+\s+)*([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", c):
    if not _repo_ok(m.group(1), REPOS):
        block(f"Hugging Face repo {m.group(1)} is not on ops/allowlist.txt (ledger); add it only if the ledger allows it, else PERMISSION_REQUESTS.md")
for seg in re.split(r"&&|\|\||[;|\n]", cmd):
    if not re.search(r"\b(curl|wget|aria2c|yt-dlp|git\s+clone|kaggle)\b", seg):
        continue
    for url in re.findall(r"https?://[^\s'\"]+|git@[^\s'\"]+", seg):
        host = re.sub(r"^(https?://|git@)", "", url).split("/")[0].split(":")[0].lower()
        if host in ("localhost", "127.0.0.1"):
            continue
        if host in ("huggingface.co", "hf.co"):
            mm = re.match(r"https?://(?:huggingface\.co|hf\.co)/(?:(datasets|spaces)/)?([^/]+/[^/?#]+)", url)
            if mm and not _repo_ok(mm.group(2), REPOS):
                block(f"Hugging Face repo {mm.group(2)} is not on ops/allowlist.txt")
            continue
        if not any(host == h or host.endswith("." + h) for h in HOSTS):
            block(f"download host {host} is not on ops/allowlist.txt; datasets must be in the ledger (else PERMISSION_REQUESTS.md)")
    if re.match(r"\s*(\S+=\S+\s+)*(nohup\s+|sudo\s+|uvx?\s+(run\s+)?|python3?\s+-m\s+)?yt[-_]dlp\b", seg) and "--download-archive" not in seg:
        block("yt-dlp only through the pipeline (needs --download-archive) and only once the gap analysis asks for YouTube")

# 9. Training jobs listed in PROGRESS.md are protected from kill unless the line carries --approved.
if re.search(r"\b(kill|pkill|killall)\b", c) and "--approved" not in c:
    protected = set()
    try:
        ended = set(re.findall(r"\bENDED pid=(\d+)", open(REPO / "PROGRESS.md").read()))
        for pid in re.findall(r"\bJOB pid=(\d+) kind=train\b", open(REPO / "PROGRESS.md").read()):
            if pid not in ended:
                protected.add(pid)
    except Exception:
        pass
    for m in re.finditer(r"\bkill\b([^|;&]*)", c):
        for pid in re.findall(r"\b(\d+)\b", m.group(1)):
            if pid in protected:
                block(f"PID {pid} is a training job listed in PROGRESS.md; append '# --approved' with a reason, and log it")
    if re.search(r"\b(pkill|killall)\b[^|;&]*\b(python|train|torchrun|accelerate|deepspeed)\b", c):
        block("pkill/killall on python/train processes can hit a protected training job; kill by PID instead")

sys.exit(0)
