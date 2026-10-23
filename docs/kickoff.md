# Kickoff notes — 23 Oct 2026

Read this right after `docs/plan-v2.md`. Where the two differ, **this file wins**. Import it from `CLAUDE.md` (`@docs/kickoff.md`) so it survives compaction.

## 1. Facts that changed since the plan was written

- **Repo**: `github.com/amin-abualhassan/saudi-tts` (origin already set, deploy key with write access). Ignore every mention of `aljawab-ai/saudi-tts`.
- **GPU window**: the instance runs on EC2 Capacity Block `cr-0beea95f3aeb16558`, which ends **Sun 1 Nov 11:30 London**. UK clocks go back on Sun 25 Oct, so that is 11:30 UTC, and AWS starts **terminating** instances at **11:00 UTC**. This is termination, not a stop: assume the EBS root (`/home/ubuntu`, including `~/saudi-tts` and `~/.secrets`) is deleted along with the NVMe. The only durable stores are `s3://aljawab-asr-data/saudi-tts/` and GitHub.
  - Every checkpoint you keep, every manifest, `reports/`, `configs/` and the dashboard data are synced to S3 as you go.
  - Final full sync done and verified by **Sun 1 Nov 09:00 London**.
  - On Sat 31 Oct, create a Notion task asking Amin to snapshot the root EBS volume from the console.
- **Calendar** (the plan's weekday labels are off by one; the stages and their order stand):
  - Day 0 Fri 23 Oct (evening)
  - Day 1 Sat 24
  - Day 2 Sun 25
  - Day 3 Mon 26
  - Day 4 Tue 27 (panel 1)
  - Day 5 Wed 28
  - Day 6 Thu 29
  - Day 7 Fri 30 (panel 2, Jordanian dry run)
  - Day 8 Sat 31 (buffer, export)
  - Sun 1 Nov morning: final sync only.
- **Extension is possible**: Capacity Reservations → Actions → Extend Capacity Block, in 1-day steps, paid up front, only if capacity exists. By Thu 29 Oct, write a `FUNDING_ASKS.md` entry plus a Notion task that recommends extending (and by how many days) or not, with the evidence.
- **Permissions**: every source in the plan's ledger is allowed for all uses. That includes Fish S2 Pro and Drama 3 (Drama 3 may be used as a teacher), Higgs, SADA (commercial derivatives included), ElevenLabs, Saudi ASWAT, MASC, Meta Omnilingual, SCC22, and everything in `s3://aljawab-asr-data/`. Anything not in the ledger still goes to `PERMISSION_REQUESTS.md`.
- **Saudi ASWAT**: Amin is requesting the download from falak.ksaa.gov.sa, and it may land in S3 later. Start with SADA, Omnilingual, SCC22 and MASC, and pick up ASWAT when it appears.
- **Panel**: Amin plus listeners he invites, rating blind pairs through the dashboard's Test page. Keep Amin's votes in their own column.
- **Voices**: the flagship is Riyadh Najdi and Maryam-like (calm, warm, female). The library also gets a Hijazi voice.
- **Second dialect for the dry run**: Jordanian.
- **The pipeline** is internal-only, but must stay reusable for any future Aljawab TTS work.

## 2. Secrets (never print, log, commit, sync or send them)

- `~/.secrets/api-keys.txt`: `ELEVENLABS_API_KEY`, `HF_TOKEN`. Parse it whether lines are `KEY=value` or `key: "value"`.
- `~/.secrets/dashboard.env`: `DASHBOARD_PASSWORD`.
- `~/.secrets/claude.env`: `CLAUDE_CODE_OAUTH_TOKEN`, which is the subscription.
- `~/.secrets/fallback.env` (may appear later): `ANTHROPIC_API_KEY` for rung 4, sourced only by `ops/watchdog.sh` when it switches the agent to API billing. Never source it yourself; an API key in the environment silently overrides the Max plan.

## 3. Network exposure

The security group is shared with the TTS-FT box and opens these ports to the whole internet: 22, 80, 443, 3000, 4001, 7860, 8000, 8001, 8002, 8080, 8081, 8888.

- Bind every model server (SGLang, vLLM), Jupyter, TensorBoard, Gradio and debug server to **127.0.0.1 only**.
- Only Caddy listens publicly, on 80/443, for `training.aljawab.ai`. The A record already points at 100.53.203.107.
- The dashboard login is the password from `dashboard.env` plus TOTP. Show the TOTP QR once, on first login, then never again.
- The TTS API stays behind the dashboard login, or on localhost, until Amin says to open it.

## 4. Guard rails already in place

- `ops/guard.py` is the PreToolUse hook for Bash, a starter written by the planner. It blocks:
  - anything but read-only commands on Majed-AWS, and pushes to it
  - instance lifecycle changes and shutdown/reboot
  - recursive deletes of root folders, and anything touching `saudi_eval_v1`
  - S3 deletes outside `saudi-tts/`
  - killing the tmux sessions or Claude
  - moving secrets anywhere
- **Extend it, never loosen it**: add the ElevenLabs spend cap, the dataset allowlist and training-PID protection. Log the change in `DECISIONS.md`.
- `ops/watchdog.sh` runs in tmux session `watchdog`. It:
  - restarts the agent session if it exits
  - nudges it if `PROGRESS.md` is quiet for 90 minutes
  - switches to Sonnet on an Opus cap
  - switches to the fallback API key on a weekly cap, and back after 12 h
  - records the auth mode in `ops/AUTH_MODE` and logs to `ops/watchdog.log`

  Do not edit it without a `DECISIONS.md` entry. While waiting on a long job, append a heartbeat line to `PROGRESS.md` at least every 60 minutes.
- Show `ops/AUTH_MODE` and the tail of `ops/watchdog.log` on the dashboard Status page. When the mode flips to `api`, create a Notion task.

## 5. The TTS-FT server (`ssh Majed-AWS`, read-only)

This box was used earlier to fine-tune Fish S2 Pro. Inventory it first and write `reports/ttsft_inventory.md`:

- what was run and how: `AGENTS.md`, `PROJECT_STATUS.md`, `README.md`, `fish-speech/` configs, logs and checkpoints, `training/`, `Splitter/`, `chatter/`, `OmniVoice/`, `speech-agent/`, `speaker_embedings.ipynb`, `refrence.txt`
- what data exists: `full-audios-transcript/`, `splitted-audio-transcript/`, `youtube-jordan-48khz/`, `youtube-jordan-telephony/`
- what can save install time: `wheelhouse/`, `uv_cache/`, `sglang_docker_storage/`
- which settings and results to adopt, what it reveals that the plan missed, and whether anything still has to be pulled. Amin will stop that box once nothing more is needed from it.

Pull with `rsync -av Majed-AWS:<path> <local>` into `/opt/dlami/nvme/` (data) or `pipeline/legacy/` (code), and credit what you adopt.

## 6. Notion

The Notion MCP server is connected at user scope.

- Find Amin's **Life OS** Tasks database. Create tasks for him: open questions, panel sessions, funding asks, HF gated-model acceptances, AUTH_MODE flips, and the Sat 31 Oct snapshot reminder.
- Close each task when the matching file says it is resolved.
- Do not edit any other Notion content.

## 7. Hugging Face gated models

If a download returns 401 or 403, do not wait. Put the exact repo IDs in **one** Notion task so Amin can accept them all from his phone. Likely candidates:

- `pyannote/speaker-diarization-community-1`
- `neuphonic/neucodec`
- any Fish or Boson repo that asks for terms

Continue with other work and retry after he ticks the task.

## 8. Money

- ElevenLabs cap: **$50** to start, tracked in `elevenlabs_spend.json`. Ask in `QUESTIONS.md` and Notion before going past it; a ceiling of $120 is under discussion.
- No other spend.

## 9. The goal (re-set it with `/goal` if it is ever cleared)

PLAN.md shows every stage PASS with gate numbers in PROGRESS.md, MODEL_CARD.md and the latency report are in reports/, and the final checkpoint, voice library and reports are synced to s3://aljawab-asr-data/saudi-tts/ and verified before Sun 1 Nov 09:00 London.
