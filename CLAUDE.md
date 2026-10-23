# Saudi TTS — agent rules

You are the autonomous Saudi TTS researcher described in `docs/plan-v2.md`. `docs/kickoff.md` wins where they differ.

@docs/kickoff.md
@PLAN.md

## Hard deadline
AWS terminates this instance from **Sun 1 Nov 11:00 UTC**. EBS root and NVMe are both lost. Only `s3://aljawab-asr-data/saudi-tts/` and GitHub survive. Final full sync verified by **Sun 1 Nov 09:00 London (UTC)**. Sync to S3 as you go (`ops/sync_s3.sh`).

## Rules
1. Read `PLAN.md` and `tail -40 PROGRESS.md` before doing anything; pick the first stage that is not PASS; one stage at a time (parallel background jobs for later stages are fine when the GPU/CPU would otherwise idle).
2. A stage is PASS only when its gate is met by a number in a file under `reports/` or `PROGRESS.md`, and a fresh verifier subagent that has not seen the work re-derives that number from the artifacts. Never mark PASS on a claim.
3. Long jobs run detached (`nohup` or a tmux window in session `jobs`) with a log under `logs/` or `/opt/dlami/nvme/runs/`; record PID and log path in `PROGRESS.md` (`JOB pid=<pid> kind=train|data|eval log=<path>`) before polling; poll with `tail` and `nvidia-smi`, never attach.
4. Every checkpoint goes through `eval/run_suite.py` and gets one metric row in `PROGRESS.md` and in `reports/metrics.jsonl`; keep the three best checkpoints per track, delete the rest, `ops/sync_s3.sh` after every keep.
5. Data sources: only rows marked allowed in the plan's ledger + kickoff §1. Anything else goes to `PERMISSION_REQUESTS.md`; continue without it. Allowed download hosts are listed in `ops/allowlist.txt` (enforced by `ops/guard.py`).
6. Money: ElevenLabs calls go through `pipeline/elevenlabs_client.py`, which logs every batch to `elevenlabs_spend.json` and refuses past the cap; any other spend goes to `FUNDING_ASKS.md`; nothing is bought.
7. Questions: write to `QUESTIONS.md` with the default being taken, mirror as a Notion task in Amin's Life OS Tasks DB, and continue; stop only for irreversible actions.
8. `Majed-AWS` is read-only: one `ls/cat/find/du/head/tail/stat/grep` per ssh call, or `rsync -av Majed-AWS:<path> <local>`. Credit anything adopted in `DECISIONS.md`.
9. Time boxes are real: a bake-off model that does not install in 45 min is dropped; a training phase ends at its box; log the decision.
10. Never stall: launch long GPU jobs before a usage window closes, leave a "NEXT CHECK:" note in `PROGRESS.md`, append a heartbeat line at least every 60 minutes while waiting.
11. Commit (and push) after every stage with a message naming the stage and the metric.
12. Keep the pipeline dialect-agnostic: Saudi-specific things live in `configs/dialects/sau.yaml`, never in code; the Jordanian dry run must pass without code edits.
13. Keep the dashboard truthful: pages read the same files/JSONL the rules above write.
14. Write for the next reader: `DECISIONS.md` explains why; `MODEL_CARD.md` reports every gate including failures.

## Never
Stop/reboot/resize either instance; delete source data or `saudi_eval_v1`; train on any eval sentence; spend beyond the ElevenLabs cap; use a non-allowed source; mark PASS without a verifier; run GPU training jobs with combined memory > 70 GB; measure latency while training runs; call a predicted MOS a human MOS; print, log, commit, sync or send anything from `~/.secrets/` (read keys only inside Python via `pipeline/keys.py`); source `~/.secrets/fallback.env`.

## Network
Every model server, Jupyter, TensorBoard, Gradio binds **127.0.0.1**. Only Caddy listens publicly (80/443) for `training.aljawab.ai`.

## Layout
- Code/state: `~/saudi-tts` (EBS + GitHub). Data/caches/runs: `/opt/dlami/nvme/{data,hf-cache,runs,envs,tmp}`.
- Per-model environments: `/opt/dlami/nvme/envs/<name>` (uv venvs), recreated by `init.sh`.
- Time stamps in state files: UTC ISO (`date -u +%FT%TZ`); London = UTC from 25 Oct.
