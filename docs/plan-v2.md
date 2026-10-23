# Saudi TTS autonomous pipeline — plan v2

Oct 23, 2026 · @Amin Abu Alhassan

## What changed from v1

The new notes turn an open-ended search into a selection problem with fixed inputs: twelve permitted base models, three permitted data sources that already contain Saudi speech, a server whose clock is already running, and a budget made of credits rather than cash. Everything below is rebuilt around those facts; the bar itself (parity with ElevenLabs V4 Turbo + Maryam E, fast streaming, easy cloning) does not move.

| Area | v1 assumption | What the notes change | Effect on the plan |
| --- | --- | --- | --- |
| Base model | Agent searches for candidates and vets licences | 12 named models with written permission or open licences (Fish S2 Pro, Fish Drama 3, Higgs Audio v3, Voxtral 4B TTS, OmniVoice, F5-TTS, Habibi-TTS, Qwen3-TTS 1.7B Base + VoiceDesign, MOSS-TTS Realtime 1.7B / Local 4B, VoxCPM2, Chatterbox Multilingual V3, Audar-TTS-V1-Flash) | Model choice becomes a scripted day-1 bake-off over a fixed list; licence vetting leaves the agent's job |
| Real data | Agent discovers datasets and negotiates one by one | SADA unrestricted, Casablanca in full (incl. non-public), YouTube allowed with re-alignment | Acquisition is settled; day 0 starts with downloads, not requests; YouTube is a top-up for coverage, not the backbone |
| Synthetic data | ElevenLabs use undefined | Written permission to train on ElevenLabs output; per-character cost applies | Distillation from the Maryam E voice is a first-class track, budgeted and used for coverage and target-voice register, not bulk hours |
| Compute | "an H100 for about 5 days" | p5.4xlarge, 1× H100 80 GB, Ubuntu DLAMI, CUDA 13.2 driver, NVMe at /opt/dlami/nvme, clock running since 21 Oct, stops about 31 Oct | Schedule runs backwards from the stop date; every stage checkpoints to storage that survives the instance |
| Agent runtime | "$250 cloud credits, maybe" | Credits expire 5 Nov 07:59 GMT; Max $100 plan already paid | Harness decided explicitly (section 4); credits are spent where they cannot stall the GPU |
| Scope | Best Saudi TTS | Pipeline-as-product: a reusable dialect fine-tuning pipeline, Saudi first | Repo is config-first (a dialect = one config + data manifests) with its own acceptance test; two deliverables, not one |
| Other public data | — | Not listed = not used until Amin confirms | Agent keeps a permission-requests file instead of silently pulling data |
| Monitoring | Amin reads logs over SSH | Must be followable from a phone; Notion Life OS available to the agent | A login-protected dashboard at training.aljawab.ai (status, runs, eval, test, questions, logs) is a day-0/1 deliverable; questions and events become Notion tasks |
| Prior work | Start from scratch | A second p5.4xlarge (TTS-FT) holds an earlier Fish S2 Pro fine-tuning setup, splitting/transcription pipelines and Jordanian YouTube corpora; S3 already holds Casablanca and other dialect corpora | Day 0 begins with a read-only inventory of both; tested code is adopted, credited and wrapped in the config-driven pipeline rather than rewritten |
| Data permissions | SADA, Casablanca, ElevenLabs, YouTube | ASWAT (2,500 h Saudi), MASC, Meta Omnilingual and SCC22 added, all uses | Real Saudi hours stop being the constraint; pipeline throughput and the two-ASR transcript gate become the bottleneck, so YouTube drops to optional |

## Mission, targets and success criteria

"Done" means a served model that a blind Saudi listening panel does not reliably prefer ElevenLabs over, that streams first audio in under 100 ms on the H100, clones a new voice from a 10-second clip, and takes gender, age and style from a text description. Every gate below is measured side by side with ElevenLabs V4 Turbo using the Maryam E voice (id aMmeBf0lzDYlouyfqNjh) on the same sentences, in the same harness, so the comparison is fair rather than quoted from marketing pages.

| Gate | What is measured | Must pass | Stretch | How |
| --- | --- | --- | --- | --- |
| Dialect fidelity | Panel rates "this is Saudi, not MSA or another dialect" on 100 held-out sentences; automatic dialect-ID classifier agreement | Panel score ≥ ElevenLabs on the same set; MSA-leakage flagged on < 5% of items | Panel score ≥ 4.5/5 | 5+ native Saudi listeners, randomised, blind; classifier run on every checkpoint |
| Naturalness | CMOS vs ElevenLabs, pairwise, same sentences | CMOS ≥ −0.1 (statistical parity) | CMOS > 0 with p < 0.05 | Same panel; 100 pairs; bootstrap CI |
| Intelligibility | WER/CER from an Arabic ASR after normalisation (strip diacritics, unify alef/hamza/taa marbuta, verbalise numerals) | WER ≤ ElevenLabs on the same set | WER ≤ 0.8 × ElevenLabs | Automatic, every checkpoint |
| Voice cloning | Speaker-verification cosine between output and a 10 s reference, on 20 unseen Saudi speakers | ≥ ElevenLabs instant clone on the same references | Within 0.02 of a real same-speaker pair | Automatic; absolute threshold calibrated on real same/different-speaker pairs from SADA on day 1 |
| Latency | Time to first audio (TTFA) and RTF, streaming, p50/p95, at 1 and 8 concurrent streams | Model-side p50 TTFA ≤ 100 ms at 1 stream, ≤ 200 ms p95 at 8 streams; RTF < 0.1 | Model-side p50 ≤ 79 ms; end-to-end over the API ≤ 150 ms | Harness in section 8; ElevenLabs measured over its WebSocket API from the same box |
| Controllability | Gender, age band, 3+ styles (calm/warm, energetic, formal) from a text prompt; designed voice stays consistent | Gender and age correct on 95% of 100 prompts; style distinguishable by panel | Emotion tags usable without quality loss | Gender/age classifier + panel spot check |
| Robustness | 200-item stress set: numbers, dates, currency (ريال), phone numbers, English code-switch, names, long paragraphs, very short inputs | 0 hallucinated or dropped content; numbers read in Saudi form | Handles 2-minute inputs without drift | Automatic ASR round-trip + rules |
| Pipeline reusability | Re-run the whole pipeline on a second dialect config | Completes with config and manifest changes only, no code edits | Second dialect reaches a usable model in one unattended day | Dry run on a Casablanca non-Saudi dialect subset |

Ship list, in priority order: (1) fine-tuned weights plus a frozen "Aljawab Saudi voice library" (3–5 designed voices, each with its reference prompt), (2) a streaming server container with an OpenAI-compatible /v1/audio/speech endpoint and a LiveKit-ready plugin, (3) the evaluation report with every gate above and the ElevenLabs side-by-side numbers, (4) the dialect fine-tuning pipeline repo with a one-page "add a dialect" guide, (5) the decision journal the agent kept.

## Resources, constraints and permissions ledger

The binding constraint is the calendar, not the GPU: the H100 stops around 31 Oct, so from a 23 Oct kickoff there are roughly 8 days, of which the last one is reserved for export and hand-off. Cash spend is to be minimised; anything that would clearly move the result (more ElevenLabs minutes, a second GPU for a day, a listening panel) is flagged for funding rather than bought.

| Resource | Detail | Limit or rule |
| --- | --- | --- |
| GPU server "Morphological Variability" (`Amin-AWS`, i-05a3bb80bb4f85759) | AWS p5.4xlarge, 1× H100 80 GB HBM3, driver 595.91.07, CUDA 13.2, Ubuntu DLAMI with conda base, 3.84 TB NVMe at /opt/dlami/nvme, SSH as ubuntu with the Amin-AWS-01 key | Running since 21 Oct 17:03 BST; stops about 31 Oct; the NVMe is instance store and is erased on stop, so it holds only data, caches and in-flight runs |
| Persistent disk on the same server | The 100 GB EBS root volume (mounted at `/`, so `/home/ubuntu` lives on it) survives stops and reboots | Code (`~/saudi-tts`), secrets (`~/.secrets`), configs, logs and the state files live here; nothing larger than a few GB |
| S3 bucket `aljawab-asr-data` | Existing bucket with `Casablanca/`, `Egypt/`, `Jordan/`, `jordanian/`, `Levant/`, `multidialectal/`, `multidialectal-parquet/`, `finance/`, `tiktok/` and an `S3 directory sizes.txt` inventory; IAM role `EC2-S3-Access` with `AmazonS3FullAccess` exists | Checkpoints, manifests and reports go under `s3://aljawab-asr-data/saudi-tts/`; the agent inventories the existing folders on day 0 and treats them as allowed data; the role must be attached to the instance (day-0 step 7) |
| Second server "TTS-FT" (`Majed-AWS`, i-0bf8e6a113fff4d51) | p5.4xlarge, running since 9 Oct, used earlier for Fish S2 Pro fine-tuning; its NVMe holds `fish-speech`, `OmniVoice`, `chatter`, `Splitter`, `speech-agent`, `training`, `full-audios-transcript`, `splitted-audio-transcript`, `youtube-jordan-48khz`, `youtube-jordan-telephony`, `speaker_embedings.ipynb`, `AGENTS.md`, `PROJECT_STATUS.md`, `README.md` | Read-only for the agent: it may `ls`, `cat` and `rsync` from it, never write, run or delete there (hook-enforced); first job on day 0 is a written inventory of what is reusable |
| Secrets | `api-keys.txt` on the NVMe with the ElevenLabs and Hugging Face keys | Moved to `~/.secrets/` on the persistent disk with mode 600 on day 0 (the NVMe copy disappears on stop and is readable by every process) |
| Agent runtime | Claude Max $100 plan (paid); $250 cloud session credits | Credits expire 5 Nov 07:59 GMT, i.e. after the GPU; see section 4 for where each is spent |
| ElevenLabs API | Generation permitted for training, all uses; per-character billing | Initial cap $50 until Amin raises it; every batch logged with character count and cost |
| Monitoring | `training.aljawab.ai` (DNS at Namecheap) pointing at the server; Notion Life OS tasks | Dashboard with password + authenticator login; see section 4 |
| Cash | None budgeted | Agent proposes, Amin approves, in a funding-asks file |

Permissions ledger — the agent reads this file at start and may only use rows marked allowed.

| Resource | Kind | Status | Notes |
| --- | --- | --- | --- |
| Fish Audio S2 Pro, Fish Drama 3 | Models | Allowed, full written permission, all uses (double-checked with Fish Audio) | Fine-tune, distil and serve; Drama 3 may be a teacher |
| Higgs Audio v3 (Higgs TTS 3) | Model | Allowed, full written permission, all uses (Boson AI) | — |
| Voxtral 4B TTS | Model | Allowed, written permission | Not fine-tunable (codec encoder withheld); API reference only |
| OmniVoice, F5-TTS | Models | Allowed, written permission, all uses | Covers the CC-BY-NC weights |
| Habibi-TTS (github.com/SWivid/Habibi-TTS) | Model + recipe | Allowed, "use in any way" | Reuse its data recipe, benchmark and text front-end |
| Qwen3-TTS-12Hz-1.7B-Base + VoiceDesign | Model | Apache-2.0 | Excluded as a base (no Arabic in pretraining) |
| MOSS-TTS Realtime 1.7B, MOSS-TTS Local 4B | Models | Apache-2.0 | Alternate streaming base |
| VoxCPM2 | Model | Apache-2.0 | Alternate design base |
| Chatterbox Multilingual V3 | Model | MIT | Excluded (low Arabic ceiling) |
| Audar-TTS-V1-Flash | Model | AudarAI Open License v1.0 (commercial and fine-tuning allowed) | Bake-off tier B |
| SADA | Dataset | Allowed, unrestricted, all uses incl. commercial derivatives (SDAIA, double-checked) | Primary real Saudi source |
| Casablanca (full, incl. non-public) | Dataset | Allowed | Already in `s3://aljawab-asr-data/Casablanca/` |
| ElevenLabs API output | Synthetic data | Allowed, full written permission, all uses | Costs money; budget above |
| YouTube audio | Data | Allowed by platform | Captions unreliable: agent re-transcribes and force-aligns everything |
| Saudi ASWAT (KSGAAL) | Dataset | Allowed, written permission for everything publicly accessible | Request the download from falak.ksaa.gov.sa; the agent uses whatever is public |
| MASC | Dataset | Allowed, written permission for the public release | \~452 h Saudi-labelled YouTube; expect 20–30% to survive dialect-ID and quality filters |
| Meta Omnilingual ASR Corpus | Dataset | Allowed (CC-BY-4.0 and written permission) | \~2.8k speaker-labelled clips each for Najdi, Hijazi, Gulf |
| SCC22 (SDAIA code-switch corpus) | Dataset | Allowed, written permission | 4.5 h Saudi code-switching; code-switch training and test items |
| Existing S3 corpora (`Jordan/`, `jordanian/`, `Levant/`, `Egypt/`, `multidialectal/`, `tiktok/`, `finance/`) | Data | Allowed (Aljawab's own holdings) | Jordanian folders feed the second-dialect dry run; the agent reads `S3 directory sizes.txt` first |
| TTS-FT server contents (`youtube-jordan-*`, transcripts, pipelines) | Data + code | Allowed, read-only | Pulled by `rsync` into the NVMe when useful |
| Any other public corpus (podcast-derived HF sets, Mixat, ADI-17, QASR, MGB, Common Voice, Emilia…) | Dataset | Not allowed yet | Agent writes the request (name, owner, licence, why) to `PERMISSION_REQUESTS.md` and continues without it |

Every written permission Amin holds includes all types of use (research, commercial derivatives, serving), as long as the data is what the owner has made publicly accessible; the agent still records each licence's text in `reports/LICENCES.md` so the model card is auditable.

## Where and how the agent runs

Run Claude Code on the H100 itself, inside tmux, signed in to the Max plan; do not host the loop in a cloud session and do not drive the server from the laptop. A cloud session is a 4 vCPU / 16 GB / 30 GB VM that pauses after minutes of inactivity, does not restore background processes, and routes all egress through an HTTP(S) proxy with no documented SSH path (a [user report](https://github.com/anthropics/claude-code/issues/84967) says port 22 times out even to GitHub). The $250 is still spent — on cloud sessions doing the GPU-free work.

| Option | Verdict | Why |
| --- | --- | --- |
| Claude Code CLI on the server in tmux | Chosen | GPU is local; survives SSH drops; interactive sessions auto-wait for 5-hour resets; `/goal` plus a watchdog loop keep it going; steerable from the phone with Remote Control |
| Cloud session driving the H100 over SSH | Rejected | HTTP(S)-proxy egress only, SSH reported blocked, VM pauses when idle, bypass mode unavailable |
| Laptop Claude Code over SSH | Rejected as primary | Ties the week to Windows staying awake; fine as a manual console |
| Claude Agent SDK | Fallback only | API-key billing only (subscription auth disallowed), more code to maintain |

&#91;embedded content: runtime · agent and watchdog on the H100, state in git, checkpoints in S3, credits on cloud sessions\]

The agent session does the thinking, detached jobs do the hours, files and git hold the state, S3 holds what must survive the instance, and the cloud sessions never touch the GPU.

Budget split. The Max plan pays for the on-server loop. The $250 cloud credit is plan-exempt until 4 Nov 23:59 PT and pays for three GPU-free jobs run as `claude --cloud` against the GitHub repo: writing and refactoring pipeline code, the daily log-and-metrics review (the server syncs run logs to S3, and `*.amazonaws.com` is allowlisted), and literature and model-card digests merged as pull requests. Overflow valve: enable usage credits with a monthly cap (standard API rates, Sonnet 5.5 $2/$10 and Opus 5.5 $4/$20 per MTok) so the loop never idles the $6.88/h GPU; if the weekly cap lands mid-week, the watchdog switches to `ANTHROPIC_API_KEY`. Model choice: Opus 5.5 for the interactive loop (planning, debugging, decisions), Sonnet 5.5 for the watchdog's `-p` runs.

Keep-alive mechanics, in the order they are set up:

1. tmux session `agent` runs `claude --permission-mode auto` (the server user is `ubuntu`, non-root, so `--dangerously-skip-permissions` is also allowed once its dialog has been accepted interactively). The work is started with `/goal "<verifiable end state>"`: a model re-checks the condition after every turn, pauses at usage limits and resumes when the session auto-waits (Claude Code 2.1.234+; it re-arms at most twice in a row).
2. tmux session `watchdog` runs a Ralph-style loop: while no `DONE` file exists, if the agent process is gone, run `claude -p "$(cat PROMPT.md)" --continue --permission-mode auto --permission-prompts none --max-budget-usd 15`, then sleep five minutes. `CLAUDE_CODE_RETRY_WATCHDOG=1` makes `-p` runs wait out limit windows instead of dying.
3. State lives outside the model: `PLAN.md` (stages with PASS/FAIL), `PROGRESS.md` (append-only), `DECISIONS.md`, `QUESTIONS.md`, `PERMISSION_REQUESTS.md`, `FUNDING_ASKS.md`, `init.sh`, and one git commit per stage. A `SessionStart` hook with `matcher: compact` re-injects the tail of `PROGRESS.md` and `git log -5` after every compaction.
4. Training never runs in the Bash foreground: every job is launched detached (`nohup` or its own tmux window) with a log file and PID recorded in `PROGRESS.md`; the agent polls `tail` and `nvidia-smi`. `BASH_DEFAULT_TIMEOUT_MS=600000`.
5. Hard guards that hold in every permission mode: `permissions.deny` plus a `PreToolUse` hook that blocks `aws ec2 stop-instances`, `shutdown`, `rm -rf` on the NVMe, killing a training PID not listed in `PROGRESS.md`, any ElevenLabs call once `elevenlabs_spend.json` exceeds the cap, any download whose URL is not in the permissions ledger, and any `ssh Majed-AWS` command that is not `ls`, `cat`, `find`, `du`, `head` or an `rsync` pull.
6. Auth that cannot expire mid-week: `claude setup-token` once, exported as `CLAUDE_CODE_OAUTH_TOKEN` in the tmux environment (one-year token).
7. Human gates: the agent writes questions to `QUESTIONS.md`, states the default it will take, and continues; it stops only for irreversible spend (ElevenLabs above cap, any new paid service) or deletion. Amin answers from the phone via the dashboard, `/remote-control`, or by editing the file.
8. Durability: the NVMe is cryptographically erased when the instance stops, so `aws s3 sync` to `s3://aljawab-asr-data/saudi-tts/` runs at every checkpoint, code and state live on the 100 GB EBS root (`~/saudi-tts`) and in GitHub, and a final sync plus an EBS snapshot run before 31 Oct.

Never-stall ladder. The GPU jobs are detached, so a paused agent never pauses training; the ladder below only decides who resumes the thinking. Each rung is automatic and the watchdog logs which rung it is on to the dashboard.

| Rung | Trigger | What happens |
| --- | --- | --- |
| 1. Plan around the window | Always | The agent checks `/usage` at the start of every stage, launches long GPU jobs before a 5-hour window closes, and leaves itself a note in `PROGRESS.md` on what to check first after a pause |
| 2. Auto-wait | 5-hour limit, reset under 24 h away | Interactive session waits and resumes; `/goal` is restored; `CLAUDE_CODE_RETRY_WATCHDOG=1` does the same for `-p` runs |
| 3. Usage credits | Any limit with usage credits enabled | Work continues on usage credits at API rates under the monthly cap Amin sets (Settings → Usage) |
| 4. API key | `weekly limit` seen in `loop.log`, or auth failure, or cap reached | The watchdog restarts the loop with `ANTHROPIC_API_KEY` from `~/.secrets/claude.env` and `--max-budget-usd 25` per run, Sonnet 5.5 by default, and posts a Notion task plus a dashboard banner saying the plan limit was hit and what the API spend is |
| 5. Nobody can think | No model reachable for 30 minutes | The watchdog keeps polling the detached jobs, syncs checkpoints to S3, and pages Amin through the dashboard and a Notion task; training is unaffected |

Monitoring and control from the phone. The agent builds a small dashboard as one of its first deliverables (day 0–1, before training starts) and serves it at `training.aljawab.ai`:

- Stack: FastAPI + HTMX pages on the server, Caddy in front with automatic HTTPS, systemd units for both; login with a password plus a TOTP code from an authenticator app (pyotp), session cookies, five-attempt lockout, and nothing else exposed on the public ports (the TTS API stays on localhost until Amin opens it).
- Pages: Status (current stage from `PLAN.md`, last 40 lines of `PROGRESS.md`, which never-stall rung the loop is on, GPU utilisation and memory, disk, ElevenLabs spend vs cap); Runs (every training run with loss curves and the metric row per checkpoint, from the JSONL the eval suite writes); Eval (latest gate table vs ElevenLabs, latency p50/p95, the dialect battery results, audio samples per checkpoint with the ElevenLabs render next to them); Test (type Saudi text, pick a voice from the library or upload a 10-second reference, hear the streaming result from any served checkpoint, and rate it, which writes into the panel log); Questions (open items from `QUESTIONS.md` with the default and a reply box that writes the answer back to the file); Logs (tail of any job log).
- Notion: the server's Claude Code gets the Notion MCP connector (day-0 step 11); the agent adds a task to the Life OS for every item in `QUESTIONS.md`, every panel session, every funding ask, and every rung-4/5 event, and closes it when the file says it is resolved.
- Phone: the dashboard is the primary view; `/remote-control` in the agent session is the steering channel when a decision needs a conversation.

### Day 0 setup, step by step

Amin does the steps below from his laptop (PowerShell or Windows Terminal) and on the server over SSH; each step says what to type and what should come back. The agent takes over at step 14. Budget about 60–90 minutes.

1. Connect. On the laptop: `ssh Amin-AWS`. You should land at `(base) ubuntu@ip-172-31-78-245:~$`. Keep this window open for every server step below.
2. Check which disk is permanent. Run `df -h / /home/ubuntu /opt/dlami/nvme`. The first two lines should show the same \~100 GB device (the EBS root; survives stops); the last should show \~3.5 TB (instance store; erased on stop). If `/home/ubuntu` shows a different, larger device, tell me before going on.
3. Secure the keys. `mkdir -p ~/.secrets && mv /opt/dlami/nvme/api-keys.txt ~/.secrets/api-keys.txt && chmod 700 ~/.secrets && chmod 600 ~/.secrets/api-keys.txt`. Then `cat ~/.secrets/api-keys.txt` once to confirm both keys are there.
4. Install Claude Code. `curl -fsSL https://claude.ai/install.sh | bash`, then `source ~/.bashrc`, then `claude --version` (expect 2.1.3xx).
5. Sign in. Run `claude`, type `/login`, choose the Claude account option, and open the printed URL on your laptop. Because this is an SSH session the browser cannot call back, so copy the code it shows into the terminal at "Paste code here if prompted". Then `/status` should show your Max plan. Accept the auto-mode notice if it appears. Type `/exit`.
6. Long-lived token. `claude setup-token`, follow the same browser flow, and copy the token into a file: `echo 'export CLAUDE_CODE_OAUTH_TOKEN=<paste>' > ~/.secrets/claude.env && chmod 600 ~/.secrets/claude.env`. Add your API key to the same file as `export ANTHROPIC_API_KEY=<key>` only if you want rung 4 of the never-stall ladder (recommended; create the key at platform.claude.com and set a monthly spend limit there).
7. S3 access. `aws s3 ls s3://aljawab-asr-data/`. If you see the ten objects, done. If you get an access error: in the AWS console go to EC2 → Instances → select "Morphological Variability" → Actions → Security → Modify IAM role → choose `EC2-S3-Access` → Update, then rerun the command.
8. GitHub. On the server: `ssh-keygen -t ed25519 -f ~/.ssh/saudi-tts-deploy -N "" -C saudi-tts-server` and `cat ~/.ssh/saudi-tts-deploy.pub`. Copy that line into github.com/aljawab-ai/saudi-tts → Settings → Deploy keys → Add deploy key, tick "Allow write access". Then `printf 'Host github.com\n  IdentityFile ~/.ssh/saudi-tts-deploy\n  IdentitiesOnly yes\n' >> ~/.ssh/config` and `git clone git@github.com:aljawab-ai/saudi-tts.git ~/saudi-tts` (answer `yes` to the fingerprint prompt). Then `git -C ~/saudi-tts config user.name "Amin (agent)"` and `git -C ~/saudi-tts config user.email <your email>`.
9. Read-only path to the TTS-FT server. From the laptop (new PowerShell window): `scp C:\Users\Vector\aljawab\aws\keys.pem Amin-AWS:~/.ssh/majed.pem`. Back on the server: `chmod 600 ~/.ssh/majed.pem` and `printf 'Host Majed-AWS\n  HostName ec2-98-83-42-166.compute-1.amazonaws.com\n  User ubuntu\n  IdentityFile ~/.ssh/majed.pem\n' >> ~/.ssh/config`. Test with `ssh Majed-AWS 'ls /opt/dlami/nvme'`; you should see the folder list you pasted (fish-speech, OmniVoice, training, …). If the connection times out, the TTS-FT security group needs port 22 open from this server's IP (100.53.203.107/32): EC2 → Security Groups → the group on TTS-FT → Edit inbound rules → add SSH from that IP.
10. Plan file into the repo. Export this doc as Markdown and save it as `docs/plan-v2.md` in the repo: on the laptop, clone the repo, add the file, commit and push; or `scp` it to `Amin-AWS:~/saudi-tts/docs/plan-v2.md` and let the agent commit it. The agent reads it first.
11. Notion connector for the agent. On the server run `claude mcp add --transport http notion https://mcp.notion.com/mcp`, then start `claude`, type `/mcp`, pick notion and complete the browser sign-in the same way as step 5 (copy the code back). Confirm with `/mcp` showing notion as connected, then `/exit`. If this flow does not finish over SSH, skip it: the dashboard still shows everything, and the agent writes the tasks to `QUESTIONS.md` instead.
12. Dashboard DNS and ports. At Namecheap add an A record `training` → `100.53.203.107` (TTL 5 min). In AWS EC2 → Security Groups → Majed-dev → inbound rules → add HTTPS 443 and HTTP 80 from 0.0.0.0/0 (port 80 is only for the certificate challenge; Caddy redirects it). Pick the dashboard password now and keep it; the agent prints a QR code for the authenticator app on first login.
13. Settings and environment for the agent. On the server: `mkdir -p ~/.claude` and `nano ~/.claude/settings.json`, paste the block below, save (Ctrl-o, Enter, Ctrl-x). Then `echo 'source ~/.secrets/claude.env; export CLAUDE_CODE_RETRY_WATCHDOG=1 BASH_DEFAULT_TIMEOUT_MS=600000 BASH_MAX_TIMEOUT_MS=1800000 CLAUDE_CODE_TOOL_MEMORY_LIMIT=16G HF_HOME=/opt/dlami/nvme/hf-cache TMPDIR=/opt/dlami/nvme/tmp' >> ~/.bashrc && source ~/.bashrc && mkdir -p /opt/dlami/nvme/data /opt/dlami/nvme/hf-cache /opt/dlami/nvme/runs /opt/dlami/nvme/tmp`.
14. Launch the agent. `tmux new -s agent -c ~/saudi-tts`, then inside it `claude --permission-mode auto`. First message: "Read docs/plan-v2.md end to end. You are the autonomous researcher it describes. Start with the bootstrap stage: create PLAN.md from section 11's stage list, PROGRESS.md, DECISIONS.md, QUESTIONS.md, PERMISSION\_REQUESTS.md, FUNDING\_ASKS.md, PROMPT.md, CLAUDE.md with section 11's rules, ops/guard.py, init.sh, and the dashboard; inventory the S3 bucket and the TTS-FT server (read-only) before anything else." Then type `/goal PLAN.md shows every stage PASS with gate numbers in PROGRESS.md and the final checkpoint synced to S3; or stop after 400 turns`. Detach with Ctrl-b then d.
15. Launch the watchdog. `tmux new -s watchdog -c ~/saudi-tts`, then inside it: `while [ ! -f DONE ]; do pgrep -f 'claude --permission-mode auto' >/dev/null || claude -p "$(cat PROMPT.md)" --continue --permission-mode auto --permission-prompts none --max-budget-usd 15 --output-format json >> loop.log; sleep 300; done`. Detach the same way. The agent writes `PROMPT.md` in its first minutes; until it exists the loop just sleeps.
16. Phone. Re-attach any time with `ssh Amin-AWS -t tmux attach -t agent`; inside the agent session type `/remote-control` once to get the link for steering from the Claude app. The dashboard URL arrives in `PROGRESS.md` and as a Notion task when it is up.

Settings block for step 13 (`~/.claude/settings.json`):

```json
{
  "permissions": {
    "defaultMode": "auto",
    "deny": [
      "Bash(aws ec2 stop-instances *)", "Bash(aws ec2 terminate-instances *)",
      "Bash(sudo shutdown *)", "Bash(sudo reboot *)", "Bash(sudo poweroff *)",
      "Bash(rm -rf /opt/dlami/nvme*)", "Bash(rm -rf ~/saudi-tts*)", "Bash(rm -rf /home/ubuntu*)",
      "Bash(git push --force *)", "Bash(aws s3 rm *)", "Bash(aws s3 rb *)"
    ]
  },
  "hooks": {
    "SessionStart": [{"matcher": "compact", "hooks": [{"type": "command", "command": "tail -40 PROGRESS.md 2>/dev/null; git log --oneline -5 2>/dev/null"}]}],
    "PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "python3 ~/saudi-tts/ops/guard.py"}]}]
  }
}
```

The agent writes `ops/guard.py` in its bootstrap stage (section 11 lists what it blocks); until the file exists the hook exits cleanly, which is fine for the first minutes.

## Candidate models and selection logic

No single permitted model has Saudi quality, streaming, cloning and text voice design at once, so the plan runs two tracks and lets a day-1 bake-off pick the bases: a streaming track (default [Fish S2 Pro](https://huggingface.co/fishaudio/s2-pro), alternate [MOSS-TTS-Local-Transformer-v1.5](https://github.com/OpenMOSS/MOSS-TTS)) that becomes the served model, and a quality-and-design track (default [OmniVoice](https://github.com/k2-fsa/OmniVoice), alternate [VoxCPM2](https://github.com/OpenBMB/VoxCPM)) that sets the quality ceiling and designs the voice library. Qwen3-TTS is out as a base (no Arabic in its 10 pretraining languages), Voxtral 4B TTS is out (Mistral withheld the codec encoder, so it cannot tokenise new audio or clone locally), and Drama 3 is a reference ceiling, not a base (35B MoE, no fine-tuning path on one H100).

| Model | Type, size, rate | Arabic evidence | Streaming, first audio | Fine-tune on 1× H100 | Cloning | Voice design | Role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [Fish S2 Pro](https://arxiv.org/html/2603.08823) (Mar 2026) | Dual-AR codec LM, 4B+0.4B, 44.1 kHz | Tier-2 language; WER 3.5%, SIM 0.75 on MiniMax-24; Elo 1112 | Yes; \~100 ms vendor, 173 ms in the [SGLang cookbook](https://cookbook.sglang.io/omni/S2-Pro) (H200, batch 1); Fast-AR not batched, so concurrency is weak | Official VQ → protobuf → LoRA/full pipeline (`extract_vq.py`, `build_dataset.py`, `train.py`, `merge_lora.py`); base is GRPO-aligned, so SFT can drift | 10–30 s ref + transcript | Inline style tags only; no gender/age design | Default streaming base |
| [MOSS-TTS Local-v1.5](https://github.com/OpenMOSS/MOSS-TTS) (Jun 2026) | AR + depth transformer, 4B (Qwen3-4B), 12.5 Hz, 48 kHz | Arabic in 31-language list; no published Arabic metric; Apache-2.0 | Yes via SGLang-Omni (only backend); no TTFB published for the 4B | Official `prepare_data.py` + `sft.py`, full FT, lr 1e-5–2e-5; [Saudi precedent](https://huggingface.co/Rabe3/Saudi-tts-full-finetune-1): 246 h in 2 h 04 m on 2× H100 | Ref clip + transcript | Separate zh/en-only VoiceGenerator | Alternate streaming base |
| [MOSS-TTS Realtime](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Realtime) 1.7B | Multi-turn streaming AR, 24 kHz | Arabic in 20 languages; SIM 68.9 (weakest) | Yes; TTFB 180 ms after warm-up on an L20, vLLM-Omni online | Official multi-turn SFT | Ref clip | None | Bake-off only; agent-dialogue option later |
| [OmniVoice](https://arxiv.org/html/2604.00688) (Apr 2026) | Discrete diffusion-LM NAR, 0.6–0.8B, 24 kHz | Best open Arabic: WER 1.39%, SIM 0.776; saw Najdi 204 h, Gulf 99 h, Hijazi 22 h in training | No native streaming; RTF \~0.03 | Official JSONL full + LoRA scripts (lr 5e-5 / 1e-4); Saudi community FT exists (Lahgtna) | 3–10 s ref | Native text attributes (zh/en-trained; teachable) | Default quality + design base; pseudo-streaming experiment |
| [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) (2026) | Tokenizer-free diffusion-AR, 2B, 6.25 Hz, 48 kHz | Arabic in 30 languages; SIM 0.79 but WER 13% on MiniMax (authors: limited Arabic data) | Chunked streaming, RTF 0.12 on a 4090; no TTFB published; 160 ms latent patches | Best FT docs: LoRA \~20 GB, full \~40 GB, lr 1e-4 / 1e-5 | Ref clip, controllable | Native: parenthetical description + style tags | Alternate design base |
| [Higgs TTS 3](https://huggingface.co/bosonai/higgs-tts-3-4b) (Jun 2026) | AR codec LM, 4.65B, 25 Hz, 24 kHz | Arabic in the <5% WER tier of 102 languages; no SIM reported; Elo 1044 | SGLang streaming "sub-second", unquantified; 617 ms/request non-streaming on H100 | No official code; community LoRA path proven ([Algerian, \~303 h](https://huggingface.co/Natiq-Studio/Natiq-dz-ARG)) | Ref + transcript | 43 emotion/prosody tags; no gender/age | Bake-off tier B |
| [Fish Drama 3](https://huggingface.co/fishaudio/drama-3) (Oct 2026) | MoE 35B-A3B, same codec | Arabic CER 0.76% vs ElevenLabs v4 0.71%; #1 open model (Elo 1248) | Yes; 88.9 ms TTFA on H200 at concurrency 1 | None; 37 GB FP8 weights; infeasible here | Ref clip | Director-style tags | Reference ceiling; candidate teacher (licence check below) |
| [Audar-TTS-V1-Flash](https://huggingface.co/audarai/Audar-TTS-V1-Flash) (Jul 2026) | Qwen2.5-0.5B LM over 50 Hz NeuCodec, 0.55B | Arabic-first, Gulf strength; Saudi WER-O 14.9 vs ElevenLabs v3 15.4 on Habibi | None documented | Trivial with any Qwen trainer; no scripts; Arabic-tuned codec decoder not released | 5–15 s ref | 8 tags | Bake-off tier B; edge/CPU option |
| [Habibi-TTS](https://arxiv.org/html/2601.13802v2) SAU | F5-TTS flow-matching, 336M | Saudi WER-O 13.6, SIM 0.71, UTMOS 1.7; beats ElevenLabs v3 on SAU WER/SIM, loses on naturalness | No | Official recipe (issue #2); NAMAA fine-tuned it on one A100 | Ref + transcript | None | Baseline, data recipe, 7-dialect benchmark and ElevenLabs harness |
| F5-TTS v1 | Flow-matching, 336M | No Arabic (Emilia zh/en) | No | Mature `finetune_cli.py` | Ref | None | Substrate of Habibi only |
| Qwen3-TTS-12Hz-1.7B-Base + VoiceDesign | AR codec LM, 1.7B | Arabic absent from pretraining; community Arabic FTs needed embedding hacks and drifted | Claimed 97 ms, no conditions; no online serving | Official single-speaker SFT only | 3 s ref | VoiceDesign (no Arabic) | Excluded |
| [Voxtral 4B TTS](https://arxiv.org/html/2603.25551) | AR semantic + flow acoustic, 4B | Best paper numbers: WER 2.68%, SIM 0.746, 70 ms on H200 | Yes | Blocked: encoder weights withheld | Blocked locally | None | API-only reference |
| Chatterbox Multilingual V3 | 0.5B Llama T3 + S3Gen | Arabic CER 6–8% | No native streaming | Community toolkit, easy | ≥10 s ref | Scalar exaggeration | Excluded (low ceiling) |

Day-1 bake-off. Before any training, the agent runs every tier-A model zero-shot on the same 60-sentence Saudi probe set with the same five Saudi reference clips and scores WER-O (Omnilingual ASR), speaker similarity (WavLM-SV), UTMOS, dialect-ID agreement and measured first-audio latency on this H100; ElevenLabs V4 Turbo + Maryam E and Habibi SAU run through the same harness as ceiling and floor. Tier A (must run): Fish S2 Pro, MOSS Local-v1.5, OmniVoice, VoxCPM2. Tier B (time-boxed to 45 min of setup each, dropped if they do not install cleanly): Higgs TTS 3, Drama 3, Audar Flash, MOSS Realtime. Each model gets its own uv/conda environment; SGLang-Omni and vLLM-Omni versions conflict, so they never share one.

Decision rule. Streaming base = the tier-A model with native streaming (Fish S2 Pro or MOSS Local-v1.5) that scores higher on the combined Saudi score (0.4 WER-O, 0.3 SIM, 0.3 UTMOS, each normalised against ElevenLabs) with measured first audio ≤ 200 ms; ties go to Fish S2 Pro for its published Arabic numbers and official LoRA pipeline. Quality base = the best combined score overall, expected to be OmniVoice; VoxCPM2 replaces it only if its Saudi WER-O is within 2 points and its voice design works in Arabic out of the box. Both tracks train; only the streaming track ships for the voice-agent path, while the quality track ships for offline rendering and as the voice-design engine.

Voice design by proxy. Neither streaming base can design a voice from text, so the voice library is built on the quality base (OmniVoice `instruct` attributes, or VoxCPM2 parenthetical descriptions), rendered as a 20–30 s Saudi reference, then frozen as a cached clone prompt in the streaming model. If the day-1 probe shows the quality base cannot design Arabic-accented voices, the fallback is ElevenLabs Voice Design output used the same way.

Licences. Amin holds full written permission, all uses, from Fish Audio (S2 Pro and Drama 3), Boson AI (Higgs TTS 3), the OmniVoice and F5/Habibi authors, SDAIA and ElevenLabs, so Drama 3 may serve as a teacher and any winner may be served commercially; the Apache-2.0 bases (MOSS, VoxCPM2) need nothing. The agent still copies each licence's text into `reports/LICENCES.md` next to the permission reference so the model card is auditable, and it does not use Voxtral or Chatterbox weights for anything but zero-shot comparison.

## Data strategy

Cleanliness and speaker consistency, not hours, are the binding constraint: Habibi trained on \~249 h of SADA and still scored UTMOS 2.3 for Saudi, so the plan targets 100–150 h of filtered, speaker-clustered real Saudi speech plus 20–40 h of multi-voice ElevenLabs synthetic speech, rather than every hour that can be downloaded. Casablanca contains no Saudi dialect (its eight dialects are Algerian, Egyptian, Emirati, Jordanian, Mauritanian, Moroccan, Palestinian, Yemeni), so it moves from "core data" to auxiliary and to the second-dialect test.

| Source | What it is | How it is used | Expected yield |
| --- | --- | --- | --- |
| [SADA](https://ieee-dataport.org/node/14747) (SDAIA, unrestricted use) | \~438 h labelled Saudi TV audio (667 h claimed), 16 kHz, labels for dialect (Najdi 117 h, Khaliji 96 h, Hijazi 40 h), gender, age, environment; Clean 123 h, Music 167 h, Noisy 148 h; speaker IDs only within each 10-minute file; transcripts have diacritics stripped and utterances with digits or English dropped | Core real data. Step 1: Clean ∩ single-speaker ∩ {Najdi, Hijazi, Khaliji} ∩ 2–15 s, re-diarised within each file to build global speaker clusters. Step 2: recover the Music subset with UVR-MDX/Demucs vocal separation and keep what passes the filters. Keep original, not denoised, audio for training (Habibi: denoising raised UTMOS but hurt WER). Also the source of `saudi_eval_v1` (speaker-disjoint) and the MSA-labelled rows for curriculum stage A | 60–90 h after step 1, +40–80 h after step 2 |
| [Saudi ASWAT](https://aclanthology.org/2026.lrec-1.124.pdf) (KSGAAL, permitted) | 2,500 h spontaneous Saudi across five regions and 13 provinces, auto-QC SNR > 20 dB, CODA-adapted dialectal transcripts, ≤10 s aligned segments, age bands | The best Saudi source if the public portion downloads in time: it is already clean and single-speaker, so it skips separation and goes straight to the two-ASR check and filters; its regional labels give the Najdi/Hijazi/Eastern tags | Depends on what is public; even 50 h of it outranks the SADA Music subset, so the download starts on day 0 |
| [MASC](https://ieee-dataport.org/open-access/massive-arabic-speech-corpus-masc) (permitted) | \~1,000 h YouTube, 452 h channel-labelled Saudi, subtitle transcripts | Full pipeline incl. dialect-ID, because the Saudi label is channel-level and will contain MSA news and non-Saudi speakers | 20–30% of the Saudi hours survive, \~90–130 h |
| [Omnilingual ASR Corpus](https://huggingface.co/datasets/facebook/omnilingual-asr-corpus) (Meta, CC-BY-4.0, permitted) | \~2.8k prompted spontaneous clips each for Najdi (`ars`), Hijazi (`acw`), Gulf (`afb`), speaker-labelled | Speaker-consistent training examples and a second speaker-disjoint eval slice; disfluency tags stripped | \~25–30 h |
| SCC22 (SDAIA, permitted) | 4.5 h Saudi podcast with Arabic–English code-switching | Code-switch training rows and the code-switch items of the stress set | 4 h |
| Existing S3 holdings in `aljawab-asr-data` | `Casablanca/` (full), `Jordan/`, `jordanian/`, `Levant/`, `Egypt/`, `multidialectal/` (+ parquet), `tiktok/`, `finance/` | Inventoried on day 0 from `S3 directory sizes.txt`; Casablanca Emirati/Yemeni and MSA rows for stage A; Jordanian folders plus the TTS-FT server's `youtube-jordan-48khz` for the Jordanian dry run; `finance/` and `multidialectal/` screened for Saudi speakers by dialect-ID | \~10 h stage-A auxiliary; Jordanian dry-run set sized on day 0 |
| TTS-FT server (`Majed-AWS`, read-only) | Earlier Fish S2 Pro fine-tuning box: `fish-speech`, `OmniVoice`, `chatter`, `Splitter`, `training`, `full-audios-transcript`, `splitted-audio-transcript`, `speaker_embedings.ipynb`, `youtube-jordan-48khz`, `youtube-jordan-telephony`, `AGENTS.md`, `PROJECT_STATUS.md` | Day-0 inventory written to `reports/ttsft_inventory.md`: what the earlier pipeline did (splitting, transcription, speaker embeddings, S2 Pro fine-tune config and results), what to adopt, and what it reveals that this plan missed; useful code is pulled by `rsync` into `pipeline/legacy/` and credited | Reuse of tested S2 Pro fine-tune settings and a Jordanian corpus |
| ElevenLabs API (permitted, all uses) | List price $0.04 per 1,000 characters on `eleven_v4_turbo`, $0.08 on `eleven_v4` and `eleven_multilingual_v2`; at \~13 Arabic characters per second, 10 h ≈ $19–37, 50 h ≈ $94–187, 100 h ≈ $187–374; a Pro plan already includes \~53 h/month of v4 Turbo | Two corpora. (a) Target register: 5–10 h with Maryam E on `eleven_v4` using the customer-service text set, so the model learns the calm, warm register the product needs. (b) Breadth: 20–30 h spread over ≥ 50 voices — the 15 Saudi-tagged library voices plus voices made with the Voice Design API from Saudi descriptions — because synthetic corpora with few speakers collapse speaker similarity (WER plateaus at \~50 speakers, SIM keeps improving to \~500). Read style, `pcm_24000` or `pcm_44100`, no diacritics in the input (billed characters). Filter every clip: ASR round-trip WER ≤ 0.1, dialect-ID agreement, and discard MSA-ified readings | 25–40 h for $50–120 at list; generate early because a library voice can be withdrawn (`disable_at_unix`) |
| YouTube (permitted; re-align everything) | Saudi podcasts, interviews, talk shows | Top-up for voice diversity and spontaneous register, through the pipeline below with two ASRs in agreement as the transcript gate; only if the day-2 gap analysis shows a missing register or speaker class, now that ASWAT and MASC are in | 20–50 h usable per pipeline day |

&#91;embedded content: data pipeline · 4 source groups, 8 stages, gate on each\]

Every stage reads the shared manifest and writes it back with one more column, so a row can always be traced to the gate that kept or dropped it.

Processing pipeline, in order, all config-driven: yt-dlp or dataset loader → ffmpeg to 24 kHz mono (16 kHz copy for VoxCPM2, 48 kHz copy for MOSS) → UVR-MDX-Net-Inst\_HQ\_3 or htdemucs vocal separation (only for Music/Noisy rows) → pyannote `speaker-diarization-community-1` in exclusive mode → Silero VAD merged to 2–20 s single-speaker segments → transcription with Qwen3-ASR-1.7B and a second ASR (Cohere Transcribe Arabic or a SADA-tuned Nemotron), keeping segments where the two agree within CER 0.2 → forced alignment with `jonatasgrosman/wav2vec2-large-xlsr-53-arabic` for word boundaries and trim → filters: DNSMOS OVRL ≥ 2.7 calibrated on the SADA-clean distribution (a UTMOS 3.0 gate rejects nearly all of SADA), speech ratio ≥ 0.8, 4–25 characters per second, speaker-embedding cosine ≥ 0.7 to its cluster centroid, ADI-20 dialect check, then drop the bottom 15% by averaged rank → 24 kHz FLAC + one JSONL manifest (`id, audio, text, speaker, dialect, duration, source, ref_audio`) from which each model's own format is derived.

Text front-end (train undiacritised, one convention): NFKC, hamza and ta-marbuta normalisation, a CODA-style Saudi spelling map (الحين/هالحين/دحين, إيش/وش, أبغى kept as written in the chosen sub-dialect), an LLM-driven verbaliser for digits, currency, times, dates and percentages in Saudi spoken form (gender agreement is the known failure mode, so every verbalised line is checked by a second LLM pass), brand names in Arabic script, English code-switches wrapped in native-script language tags, diacritics added only on ambiguous homographs, and one sub-dialect per utterance. The same module produces both the training text and the normalised references for scoring.

Priority when the GPU and the pipeline are the bottleneck: ASWAT public portion and SADA Clean first (cleanest Saudi speech per hour of processing), then Omnilingual and SCC22 (small, clean, labelled), then MASC and SADA Music (both need separation and dialect-ID), then ElevenLabs breadth, then YouTube. Sources still off limits until Amin says otherwise, each logged in `PERMISSION_REQUESTS.md` if the agent wants them: the podcast-derived set `musabalosimi/saudi_dialect_asrv1.0` and the podcasts behind it, Mixat (Emirati), ADI-17 (labels only), QASR and MGB-2, Common Voice, Emilia.

## Training and fine-tuning plan

The recipe is adaptation of an Arabic-pretrained base, not teaching a language: small learning rates (about 5% of pretraining, 1e-5 to 2e-5 for full fine-tuning), frozen speaker encoder and codec, a short pan-Arabic stage before the Saudi stage, 10–20% replay of the base model's native languages every epoch, and checkpoint selection by the evaluation suite rather than by loss. Full fine-tuning of codec LLMs at pretraining-scale learning rates has been shown to raise WER from 4% to 12%, and every published Saudi fine-tune so far is single-speaker and unmeasured, which is exactly the gap this run closes.

Phases, in order:

1. Bake-off (day 1, zero-shot, no training): section 5's decision rule picks the streaming base and the quality base; the losing candidates are deleted from the NVMe.
2. Stage A, pan-Arabic warm-up (about 1 epoch): SADA MSA-labelled rows plus Casablanca MSA and Emirati rows plus 10–20% of the base model's native-language data (replay), stopped when predicted MOS on the dev set converges. Habibi's MSA-then-dialect curriculum beat direct fine-tuning by 1.4 WER; here the stage is kept short because the base already knows Arabic.
3. Stage B, Saudi adaptation (2–3 epochs over the 100–150 h real set plus the synthetic breadth corpus): multi-speaker, reference-audio conditioning kept on 30–50% of rows so zero-shot cloning survives, dialect tag `[sau]` (sub-dialect tags `[najdi]`, `[hijazi]`, `[gulf]` where SADA labels allow) prepended to the text, replay continued, unseen-speaker similarity scored every epoch.
4. Stage C, register and voices (half a day): continue from the best stage-B checkpoint on the Maryam-register corpus mixed 1:1 with stage-B data, so the calm customer-service style is learned without collapsing to one speaker; then build the voice library (section 5, voice design by proxy) and freeze each voice as a cached clone prompt.
5. Stage D, optional preference alignment (only if the stretch gates are close): GRPO or DPO with reward = mean(1 − CER, speaker similarity), LoRA r64, lr 1e-5, KL β 0.1, which lifted Fish S2 Pro Arabic CER 6.57 → 6.38 in the IWSLT 2026 recipe.

| Base | What is trained | Learning rate, batch, steps | Data format | Time on one H100 | Guard rails |
| --- | --- | --- | --- | --- | --- |
| Fish S2 Pro (streaming default) | LoRA r32 α16 on the Slow-AR attention + MLP; embeddings, output head and Fast-AR frozen (a Fast-AR LoRA homogenised pitch across speakers in the Hebrew fine-tune); full FT of the Slow-AR at 1e-5 only if LoRA plateaus | LoRA 5e-5 to 1e-4 constant, bf16, residual-codebook loss ×0.3, batch 8–12 utterances, 10–20k steps, checkpoint every 1k | `data/SPK/*.wav` + `.lab` → `fap loudness-norm` → `extract_vq.py --config-name modded_dac_vq` → `build_dataset.py` protos → `train.py --config-name text2semantic_finetune` → `merge_lora.py` | \~2–4 h per 150 h epoch for LoRA (estimate; no official figure) | Base is GRPO-aligned, so prefer the earliest acceptable checkpoint and verify free-running generation (teacher-forced loss can look fine while generation collapses); confirm `extract_vq` uses `s2-pro/codec.pth` on day 0 |
| MOSS-TTS Local-v1.5 4B (streaming alternate) | Full FT of the backbone + depth transformer, codec frozen | 1e-5 to 2e-5 cosine, 3% warm-up, AdamW wd 0.1, batch 1 × grad-accum 8–16, grad checkpointing, `--channelwise-loss-weight 1,32`, bf16 with fp32 master weights via ZeRO offload (the Saudi precedent lost updates in pure bf16) | JSONL `{audio, text, language: "Arabic", ref_audio}` → `prepare_data.py` (1.6B codec, GPU) → `sft.py` via accelerate | \~4 h per 250 h epoch (2 h 04 m on 2× H100 for 246 h) | Check Arabic zero-shot quality in the bake-off first; no published Arabic metric |
| OmniVoice 0.6–0.8B (quality default) | Full FT, or LoRA r16 α32 on q/k/v/o/gate/up/down with `audio_embeddings` and `audio_heads` trained | Full 5e-5, LoRA 1e-4, `batch_tokens` 8192, 20–50k steps for \~150 h | JSONL `{id, audio_path, text, language_id}` → `run_finetune.sh` / `run_finetune_lora.sh` → `omnivoice-merge-lora` | Hours, not days; can share the GPU with the streaming track | Teach voice design in Arabic by adding attribute strings (gender, age, "Saudi") to 20–30% of rows as Lahgtna did; verify `language_id` semantics (undocumented) |
| VoxCPM2 2B (design alternate) | LoRA r64 on LM + DiT (`enable_dit: true`, essential), then full FT at 1e-5 if needed | LoRA 1e-4, wd 0.01, batch 16, 1–3 epochs; \~20 GB VRAM | JSONL `{audio, text, ref_audio, duration, ref_duration, dataset_id}` with 30–50% `ref_audio` rows, 16 kHz input | Hours | Keep `training_cfg_rate` 0.1; authors recommend 10–20% zh/en replay |

Checkpoint selection and stop rules. Every checkpoint runs the automatic suite (section 8); the agent keeps the three best by the combined Saudi score and deletes the rest. A checkpoint is rejected if unseen-speaker similarity drops more than 0.03 below the base, if the dialect-ID score drops below the base, if hard-set WER rises above 2× normal, or if any failure-rate gate trips. Training stops when two consecutive checkpoints do not improve the combined score, or at the phase's time box, whichever comes first.

Synthetic data in training. Synthetic rows are tagged `source: elevenlabs` and are down-weighted to at most 30% of each stage-B batch and 0% of the last 500 steps, so real timbre diversity dominates the final checkpoint; ScienceSoft's 70 h synthetic-only Emirati run drifted to foreign-sounding Arabic on long passages, and the drift test in the stress set exists to catch exactly that.

GPU scheduling. The streaming track owns the GPU for its training windows; the quality track (OmniVoice full FT at under 20 GB, VoxCPM2 LoRA at \~20 GB) runs in the gaps or concurrently when the streaming job leaves 30 GB free, and evaluation jobs are queued, never run alongside a training step on the same GPU for latency measurements.

Reusability. Every stage reads one `dialect.yaml` (dialect code and tags, orthography map, verbaliser rules, dialect-ID model and class, evaluation ASRs, base model and adapter choice, replay sources) and the shared manifest; adding a dialect means a new YAML and manifests, nothing else. The Jordanian dry run on day 7 (Casablanca Jordanian rows, the S3 \`Jordan/\` and \`jordanian/\` folders, and the TTS-FT server's \`youtube-jordan-48khz\`) is the acceptance test for this claim; Amin judges the output himself. The pipeline is internal-only but is written to be reused for every future Aljawab TTS job.

## Evaluation protocol

Every number is relative: to real Saudi recordings scored by the same ASR, and to ElevenLabs rendered through the same harness. No MOS predictor, LLM judge or dialect classifier has been validated on Arabic, and the best open ASRs still score 17–37% WER on real Saudi speech, so absolute WER or UTMOS values prove nothing on their own. CER is the primary intelligibility number, ΔCER against ground truth is the gate, and the final word belongs to a small native panel.

| Metric | Tool | Gate |
| --- | --- | --- |
| ΔCER / ΔWER | [Cohere Transcribe Arabic](https://huggingface.co/blog/CohereLabs/cohere-transcribe-arabic-07-2026-release) and [oddadmix whisper-large-v3-arabic-dialectal-v2](https://huggingface.co/oddadmix/whisper-large-v3-arabic-dialectal-v2), jiwer, one shared normaliser (strip harakat and tatweel, NFKC, fold alef/hamza/ya/ta-marbuta, Eastern→Western digits, verbalise digits in references); never the ASR family used to pseudo-label training data | ΔCER ≤ +1 and ΔWER ≤ +2 points vs real clips; absolute CER ≤ 8% |
| Hard-set WER and failure rate | Same ASRs on the stress buckets; duration ratio outside \[0.5, 2.0\], trailing audio > 1.5 s, insertion rate > 30%, language-ID not Arabic, empty output all count as failures | Hard WER ≤ 2× normal; failures ≤ 1% normal, ≤ 5% hard |
| Speaker similarity | WavLM-large-SV cosine from [Seed-TTS-eval](https://github.com/BytedanceSpeech/seed-tts-eval) vs held-out clips of the same speaker; ECAPA secondary; real-vs-real ceiling computed on SADA first | ≥ 0.70 and ≥ ceiling − 0.05 (human reference 0.73–0.75; Habibi SAU 0.705) |
| Distributional quality | [TTSDS2](https://arxiv.org/pdf/2506.19441) against 500 real Saudi clips (covers Arabic) | Headline automatic score; no regression between checkpoints |
| Predicted MOS | UTMOS, UTMOSv2, Audiobox Aesthetics PQ/CE, always labelled "predicted" | Floor UTMOS ≥ 3.0; no drop > 0.1 vs previous checkpoint; never used for parity claims |
| Dialect fidelity | [Elyadata/ADI-whisper-ADI20](https://huggingface.co/Elyadata/ADI-whisper-ADI20) (KSA/Gulf labels) plus the 5-class MMS classifier, calibrated on real SADA clips before use; WER with the Saudi-tuned ASR must be below WER with an MSA ASR | Mean P(KSA ∪ Gulf) ≥ 0.8; argmax KSA on ≥ 85% of dialectal items |
| Pairwise LLM judge | Gemini 2.5 Pro pairwise vs cached ElevenLabs Maryam E renders of the same sentences, both orders, tie = ½ (EmergentTTS protocol, system-level ρ ≈ 0.9 with humans); validated once against the panel, kept only if ρ ≥ 0.8 | Win rate with CI, used for checkpoint ranking between panels |
| Latency | Harness below, against the served model and against ElevenLabs from the same box | Model-side p50 TTFA ≤ 100 ms (stretch ≤ 79 ms), p95 ≤ 200 ms at 8 streams; client-side ≤ ElevenLabs in the same harness |
| Human AB/CMOS, DMOS, SMOS | Static web page, 60 items, 8–10 native Saudi listeners, both orders, loudness-normalised, three attention pairs, binomial and Wilcoxon tests | Win-rate CI contains 50% and |

Test set `saudi_eval_v1` (\~1,200 items, frozen on day 0, never trained on): 500 SADA test utterances, 3–12 s, speaker-disjoint, spread over Najdi, Hijazi and Eastern/Gulf, for ΔCER, the SIM ceiling and TTSDS2; 300 written Saudi customer-service turns (banking, telecom, delivery, appointments) checked by the dialect-router text classifier and spot-read by Amin; 200 numeric and entity items (Eastern and Western digits, SAR amounts, 05x phone numbers, Hijri and Gregorian dates, alphanumeric IDs, brand names); 100 English code-switch items; 100 edge items (1–3 words, 400+ characters, repeats, punctuation-only). The dialect battery from the adaptation research is embedded in the 300 conversational items: ق→g and ك→ts words, dialect function words (دحين، وش، أبغى، مو، ليش، فين) against their MSA forms, and long passages that expose MSA leakage and drift.

Latency harness. One Python client, the same 60 sentences in short, medium and long classes, five warm-ups then at least 50 timed runs per class, timestamps at text send (with flush) and at first audio byte, handshake and median RTT measured and subtracted, run at 1, 4, 8 and 16 concurrent streams. ElevenLabs is measured over its streaming API with the Maryam E voice from the same box; its documented numbers for v4 Turbo are about 100 ms median inference and about 150 ms median time-to-first-speech over WebSocket with network removed ([ElevenLabs v4](https://elevenlabs.io/blog/eleven-v4)), and the 79 ms Amin saw is the figure the ElevenLabs agent builder shows on the v4 Turbo model card (an inference-only number, no network), so the plan reports both the model-side number against 79 ms and the client-side number against ElevenLabs measured live. Its documented stream-input WebSocket does not list v4 models, so the agent uses whichever streaming endpoint the ElevenLabs SDK selects for `eleven_v4_turbo` and records which one it was. WER and SIM are re-scored on audio captured through the serving stack, because the same weights have shown 3.8–8.9% WER across hosts.

Cadence. The automatic suite runs on every checkpoint (under 15 minutes on the H100 for 1,200 items) and writes one row to `PROGRESS.md` and to the dashboard; the LLM pairwise judge runs on candidates for "best so far"; the human panel runs twice, once on day 4 to validate the judge and catch a bad direction early, and once on the final candidates. The panel is Amin plus the Saudi listeners he brings in, rating blind pairs through the dashboard's Test page (it randomises order, hides which side is ElevenLabs, and stores every vote); Amin's own ratings are kept in a separate column so the report can show results with and without the person who ran the project.

## Deployment and serving

What ships is a container that serves the fine-tuned streaming base through SGLang-Omni (Fish S2 Pro or MOSS Local-v1.5 — SGLang-Omni is the only backend that runs MossTTSLocal) behind an OpenAI-compatible `POST /v1/audio/speech` and a WebSocket stream, plus a LiveKit `tts_node` plugin and an ElevenLabs-schema shim so the existing demo platform switches vendors by URL. The quality base is served separately through vLLM-Omni (`vllm serve <model> --omni`, which lists OmniVoice and VoxCPM2) for offline rendering and voice design.

Latency levers, applied in this order and measured after each: warm-up before `/ready`; cached clone prompt per library voice (zero prompt-encoding cost on the hot path); CUDA graphs on the LM and the codec decoder (never `--enforce-eager` in production); bf16 with FlashAttention-2 and an fp16 codec; the smallest first codec chunk the backend allows (vLLM-Omni `initial_codec_chunk_frames: 1`, SGLang's equivalent), then a ramp to larger chunks; leading-silence trim; raw PCM at 24 kHz over the WebSocket, no mp3 or opus on the hot path; `max_num_seqs` 8–10 so eight concurrent streams stay under 300 ms p95. Published reference points on the same class of hardware: Fish S2 Pro about 100 ms vendor / 173 ms SGLang cookbook on an H200 at batch 1, Drama 3 88.9 ms, Voxtral 70 ms, MOSS Realtime 180 ms on an L20 — so 100 ms on this H100 is plausible for the streaming base and 79 ms needs the whole lever list.

Concurrency reality. Fish S2 Pro's Fast-AR is not batched in the cookbook build (RTF rises above 1.0 at batch 8), so for a voice-agent product the plan measures throughput honestly and, if S2 Pro wins on quality but loses on concurrency, serves two or three replicas on the one card (about 24 GB each) or chooses MOSS Local-v1.5 for the agent path. Quality is re-scored on audio captured through the server, since the same weights have shown 3.8–8.9% WER across serving stacks.

Fallback for the quality track. If no streaming base passes the 200 ms gate at acceptable quality, the quality base is served with clause-level pseudo-streaming: the text is split at the first clause boundary, the first clause (1–2 s of audio) is rendered in full at RTF \~0.03 while the rest renders in parallel, which puts first audio in the 120–200 ms range on an H100 — below ElevenLabs on latency, but a usable interim product.

API shape. `POST /v1/audio/speech` with `{model, input, voice, response_format: pcm|wav, speed, stream}` and voice names from the library (`maryam-like-calm`, `najdi-male-formal`, …) or `voice: {reference_audio_b64, reference_text}` for ad-hoc cloning; `WS /v1/audio/speech/stream` accepting incremental text with a flush message; `GET /v1/voices`; optional `style` field mapped to the base model's tags. The ElevenLabs shim exposes `/v1/text-to-speech/{voice_id}/stream-input` returning base64 audio chunks with `isFinal`, so the Al Awaly and IKEA demo code needs only a base-URL change.

Deliverables on day 7: Docker image (SGLang-Omni, model, library prompts, shim), a `MODEL_CARD.md` with every gate's numbers and the ElevenLabs side-by-side, the LiveKit and Pipecat snippets (`OpenAITTSService(base_url=...)`, 24 kHz PCM), the latency report at 1/4/8/16 streams, and the final weights plus voice library synced to S3 and snapshotted to EBS before 31 Oct.

## Day-by-day schedule

The schedule runs backwards from the GPU stop on 31 Oct: the last day is a buffer, the two human panels sit on day 4 and day 7, and training must be in its Saudi stage by the morning of day 3. Everything before day 3 is data and the bake-off; everything after day 5 is voices, serving and proof.

&#91;embedded content: GPU window schedule · 23–31 Oct, 14 workstreams, 3 gates\]

What the agent does each day (dates are Europe/London; the agent works around the clock, Amin checks in on his own schedule):

1. Day 0, Thu 23 Oct: Amin runs the 16 setup steps; the agent then bootstraps the repo and state files, writes `ops/guard.py`, inventories `s3://aljawab-asr-data` and the TTS-FT server (read-only) into `reports/`, starts the SADA, ASWAT, MASC, Omnilingual and SCC22 downloads onto the NVMe, pulls reusable code from the TTS-FT box into `pipeline/legacy/`, builds and deploys the dashboard with login, downloads tier-A and tier-B weights into separate environments, freezes `saudi_eval_v1`, writes the nine evaluation scripts, and renders the ElevenLabs references for the eval sentences (about 1,200 short sentences, under $5).
2. Day 1, Fri 24 Oct: SADA step 1 (clean subset, re-diarised, filtered) and the ASWAT and Omnilingual sets through the two-ASR gate; run the bake-off and write the base-model decision to `DECISIONS.md`; start ElevenLabs corpus generation (target register first, then breadth over 50+ voices); delete losing candidates; the dashboard's Test page serves the bake-off winners so Amin can listen from his phone.
3. Day 2, Sat 25 Oct: MASC and SADA step 2 (music recovery) through the full pipeline, text front-end and verbaliser complete and unit-tested on the stress set, synthetic filtering, gap analysis (registers, gender, sub-dialects) that decides whether YouTube is needed, stage A warm-up on the streaming base, quality-base fine-tune starts in parallel.
4. Day 3, Sun 26 Oct: stage B Saudi training begins on the streaming base (through day 5), quality base in stage B, checkpoint evaluation every 1,000 steps, first LLM-judge rankings, Notion task asking Amin to schedule panel 1.
5. Day 4, Mon 27 Oct: panel 1 on the best checkpoint through the dashboard (validates the judge, catches drift or MSA leakage early), recipe or data fixes, YouTube top-up pipeline only if the gap analysis asked for it.
6. Day 5, Tue 28 Oct: stage C register and voice library on both tracks, serving container built, first measured TTFA on the H100 against ElevenLabs, served checkpoints wired into the Test page.
7. Day 6, Wed 29 Oct: latency engineering through the lever list, concurrency tests at 1/4/8/16 streams, LiveKit plugin and ElevenLabs shim, optional stage D alignment if the stretch gates are within reach.
8. Day 7, Thu 30 Oct: final automatic evaluation through the server, panel 2 on the final candidates, Jordanian dry run (one new `jor.yaml`, no code edits), model card and latency report, full S3 sync.
9. Day 8, Fri 31 Oct: buffer for whatever slipped, EBS snapshot, final sync, `DONE` file, hand-off note in `PROGRESS.md` and a closing Notion task listing what to do when the next GPU window opens.

## Agent operating spec

The agent is a TTS researcher on a deadline: it works stage by stage against `PLAN.md`, proves every PASS with a file another agent can check, writes everything it learns to disk, and never spends money or deletes data on its own. The spec below is what goes into `CLAUDE.md` (under 200 lines), `PROMPT.md` (the watchdog's standalone prompt) and `~/.claude/settings.json`.

```text
~/saudi-tts/
  CLAUDE.md            rules below; @PLAN.md and @PROGRESS.md imported
  PLAN.md              stages with gate, status (TODO/RUNNING/PASS/FAIL), evidence path
  PROGRESS.md          append-only log: timestamp, what ran, PID + log path, metric row
  DECISIONS.md         one entry per decision: options, evidence, choice, how to reverse
  QUESTIONS.md         questions for Amin with the default the agent is taking
  PERMISSION_REQUESTS.md / FUNDING_ASKS.md / elevenlabs_spend.json
  init.sh              recreates the environments from scratch (tested on day 0)
  configs/dialects/sau.yaml, jor.yaml     the only per-dialect inputs
  configs/models/{s2pro,moss_local,omnivoice,voxcpm2}.yaml
  data/                manifests only (audio lives on /opt/dlami/nvme/data)
  pipeline/            ingest, separate, diarise, segment, transcribe, align, filter, normalise, verbalise, export_<model>
  train/               one entry point per base, all reading the shared manifest
  eval/                the nine scripts from section 8 + saudi_eval_v1/
  serve/               container, API, shim, livekit plugin, bench_latency
  reports/             MODEL_CARD.md, latency report, panel results
```

Rules in `CLAUDE.md`:

1. Read `PLAN.md` and the last 40 lines of `PROGRESS.md` before doing anything; pick the first stage that is not PASS; one stage at a time.
2. A stage is PASS only when its gate in section 2 or 8 is met by a number in a file under `reports/` or `PROGRESS.md`, and a fresh verifier subagent that has not seen the work re-derives that number from the artifacts. Never mark PASS on a claim.
3. Long jobs run detached with a log file; record PID and log path in `PROGRESS.md` before polling; poll with `tail` and `nvidia-smi`, never attach.
4. Every checkpoint goes through `eval/run_suite.py` and gets one metric row in `PROGRESS.md` and in the dashboard's JSONL; keep the three best checkpoints per track, delete the rest, `aws s3 sync` after every keep.
5. Data sources: only rows marked allowed in the ledger (section 3). Anything else goes to `PERMISSION_REQUESTS.md` with owner, licence and expected value; continue without it.
6. Money: ElevenLabs calls stop when `elevenlabs_spend.json` reaches the cap; any other paid service or GPU change goes to `FUNDING_ASKS.md`; nothing is bought.
7. Questions: write to `QUESTIONS.md` with the default being taken, mirror it as a Notion task in the Life OS, and continue; stop only for irreversible actions (spend above cap, deletion of source data, anything touching either instance).
8. The TTS-FT server (`Majed-AWS`) is read-only: list, read and `rsync` from it; never write, install, run jobs, or delete there. Credit anything adopted from it in `DECISIONS.md`.
9. Time boxes are real: a model that does not install in 45 minutes is dropped from the bake-off; a training phase ends at its box even if improving, and the decision is logged.
10. Never stall: check `/usage` at the start of each stage, launch long GPU jobs before a window closes, leave a "next check" note in `PROGRESS.md`; the watchdog climbs the ladder in section 4 on its own and reports the rung on the dashboard.
11. Commit after every stage with a message naming the stage and the metric; `git log` is the recovery path after compaction.
12. Keep the pipeline dialect-agnostic: anything Saudi-specific lives in `configs/dialects/sau.yaml`, never in code; the day-7 Jordanian dry run must pass without code edits.
13. Keep the dashboard truthful: every page reads the same files and JSONL the rules above write; nothing is hand-edited for display.
14. Write for the next reader: `DECISIONS.md` explains why, `MODEL_CARD.md` reports every gate including the failed ones.

The loop. The interactive session runs `/goal "PLAN.md shows every stage PASS with gate numbers in PROGRESS.md and the final checkpoint synced to S3; or stop after 400 turns"`. The watchdog's `PROMPT.md` says the same in standalone form ("you are resuming an unattended run; read PLAN.md and PROGRESS.md; continue the first unfinished stage; do not redo finished work"). Stage order in `PLAN.md`: bootstrap → data (SADA step 1, eval set, synthetic, SADA step 2, text front-end) → bake-off → stage A → stage B → panel 1 → stage C → serving → latency → panel 2 → dry run → export. A verifier subagent runs at the end of each stage; a third agent's job on day 7 is to re-run `init.sh` on a clean directory and reproduce one number.

Guards in `~/.claude/settings.json` (deny rules hold even in bypass mode): deny `aws ec2 stop-instances *`, `aws ec2 terminate-instances *`, `sudo shutdown *`, `sudo reboot *`, `rm -rf /opt/dlami/nvme*`, `rm -rf ~/saudi-tts*`, `git push --force *`; a `PreToolUse` hook on Bash that blocks `kill` of any PID listed as a training job in `PROGRESS.md` unless the command line contains `--approved`, blocks `curl`/`wget`/`huggingface-cli download` of any dataset not in the ledger, and blocks ElevenLabs calls once the cap is reached; a `SessionStart` hook with `matcher: compact` that prints `tail -40 PROGRESS.md` and `git log --oneline -5`; `defaultMode: auto`.

What the agent must never do: stop, reboot or resize the instance; delete source data or `saudi_eval_v1`; train on any eval sentence; spend beyond the ElevenLabs cap or on any new service; use a data source not marked allowed; mark a stage PASS without a verifier run; run two GPU training jobs with combined memory above 70 GB; measure latency while a training step is running; report a predicted MOS as a human MOS.

## Risks and fallbacks

The three risks most likely to decide the week are latency (no open model has a published Arabic first-audio number on an H100), the quality of SADA after filtering, and the Max plan's weekly cap landing mid-run; each has a fallback the agent can take without asking.

| Risk | Early signal | Fallback |
| --- | --- | --- |
| Streaming base cannot reach 200 ms first audio at acceptable quality | Day-1 bake-off TTFA or day-5 served TTFA above gate | Switch to the alternate streaming base; serve two or three replicas on the card for concurrency; ship the quality base with clause-level pseudo-streaming (120–200 ms) as the interim product |
| Fine-tuning degrades pronunciation or stability (RL-aligned base drifts) | WER rises on the normal set, hard-set failures, pitch flattening across speakers | LoRA only, embeddings and Fast-AR frozen, lower learning rate, earlier checkpoint, more replay; MOSS as the alternate |
| SADA yields too little TTS-grade audio | Under 60 h after step 1 | Music-subset recovery, more ElevenLabs breadth within cap, YouTube top-up pipeline, request ASWAT |
| Synthetic data collapses speaker similarity or imports MSA-ified reading | Unseen-speaker SIM drops; dialect-ID score drops; long-passage drift | ≥ 50 voices, synthetic ≤ 30% of batches and 0% in the last 500 steps, dialect filter on every synthetic clip |
| Max plan 5-hour or weekly cap stalls the loop | `You've hit your weekly limit` in loop.log | Interactive auto-wait and `CLAUDE_CODE_RETRY_WATCHDOG`; usage credits with a cap; watchdog switches to `ANTHROPIC_API_KEY`; cloud sessions take the GPU-free work |
| Environment conflicts (vLLM-Omni, SGLang-Omni, CUDA 13.2 driver vs wheels) eat day 1 | A model fails to install in its 45-minute box | Separate uv environments per model; pinned versions from each recipe page; drop the model from the bake-off and log it |
| Instance stops early or NVMe is lost | AWS notice; `nvidia-smi` gone | `aws s3 sync` at every checkpoint; `init.sh` rebuilds the environment on any box; EBS snapshot on day 7 |
| Licence surprises block shipping (SADA derivatives are CC-BY-NC-SA publicly; S2 Pro is research-licensed; Drama 3 forbids outputs-for-training and hosted APIs; Higgs is non-commercial) | A letter does not name the clause | Confirm letters before day 5 (section 13); keep a MOSS Local-v1.5 (Apache-2.0) track alive as the clean-licence path |
| Maryam E withdrawn from the Voice Library mid-project | `disable_at_unix` set on `GET /v1/voices/{id}` | Generate the target-register corpus on day 1; keep the reference renders for evaluation |
| Agent stalls, loops or declares victory early | No new `PROGRESS.md` line for 2 hours; PASS without evidence path | Watchdog restart; verifier subagent on every PASS; `/goal` condition tied to files, not claims; Amin's phone check-ins via Remote Control |
| Automatic metrics mislead on Arabic (predictors and judges are unvalidated) | Judge ranking disagrees with panel 1 | Panel 1 on day 4 validates the judge; if ρ < 0.8 the judge is dropped and checkpoints are ranked by ΔCER, SIM and dialect-ID only, with a second small panel |
| MSA leakage in text or audio | Dialect-router flags generated text; ق read as \[q\]; أريد instead of أبغى | Dialect tags in training, LLM text verification before synthesis, the dialect battery in every eval run |

## Things Amin should go get now

Most of the asks from the first draft are done: all model and data permissions are in hand (all uses), the repo exists, the keys are on the server, the S3 bucket and IAM role exist, and the panel is Amin and his own listeners. What remains is the day-0 setup walk-through in section 4 and the items below.

Before kickoff (beyond the 16 steps):

- [ ] Enable usage credits on the Claude account with a monthly cap (rung 3), and create an API key with a spend limit for rung 4; put the key in `~/.secrets/claude.env` (step 6).
- [ ] Request the ASWAT download from falak.ksaa.gov.sa (the permission is in hand; the files still need fetching) and put the archive, or its link, where the agent can reach it (S3 or the NVMe).
- [ ] Confirm the two instances' security groups: port 22 from this server's IP into TTS-FT (step 9) and 443/80 into this server for the dashboard (step 12).
- [ ] Tell the listeners you will invite that panel 1 is around 27 Oct and panel 2 around 30 Oct, 30 minutes each, done on their phones through the dashboard link.
- [ ] Decide what to do with TTS-FT after the inventory: if nothing on it is needed after day 1, stopping it saves about $165/day; the agent will say in `reports/ttsft_inventory.md` whether anything still has to be pulled.

Funding worth asking for (the agent flags, never buys): extending this H100 by 3–4 days at about $6.88/h (≈ $165/day) is the single most valuable spend if day 5 looks good; a one-month ElevenLabs Scale plan ($299, ≈ 160 h of v4 Turbo) if synthetic breadth proves out; one extra GPU-day (\~$170) if the bake-off needs the full tier-B list.

## Open questions for Amin

- [x] 79 ms: it is the figure on the v4 Turbo model card in the ElevenLabs agent builder (inference-only). Kept as the stretch; ≤ 100 ms model-side is the must.
- [x] Flagship sub-dialect: Riyadh Najdi for the first voice (more formal, Amin's lean); the library gets a Hijazi voice too, and the Maryam-like voice is built from the ElevenLabs reference regardless of sub-dialect.
- [x] Permissions: all written permissions cover all uses for everything publicly accessible; ElevenLabs, SDAIA, Fish Audio and Boson AI double-checked; ASWAT, MASC, Omnilingual and SCC22 added.
- [x] Listening panel: Amin plus people he invites, through the dashboard.
- [x] Second dialect: Jordanian.
- [x] Pipeline: internal-only, reusable for future TTS work.
- [x] Qwen3-TTS: dropped as a base.
- [ ] GPU stop: is 31 Oct exact, and is a 3–4 day extension possible if day 5 is promising?
- [ ] ElevenLabs cap: $50 to start, $120 pre-approved ceiling for the breadth corpus?
- [ ] The dashboard password and the authenticator app you will use (any TOTP app works)?
- [ ] Should the served TTS API be reachable from outside (for the demo platform) during the week, or only after the final model is in?
- [ ] Does Notion access for the agent matter enough to do the browser sign-in over SSH (step 11), or is the dashboard's Questions page enough?
