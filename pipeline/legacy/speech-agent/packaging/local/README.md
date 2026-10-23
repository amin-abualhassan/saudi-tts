# Recreate Aljawab locally — read this first

This is a source/configuration handoff of the running Aljawab voice app, including its latest Noura voice, Saudi wording, spoken-number conversion, less-sensitive Silero VAD, and asynchronous Decisions acknowledgments. **No virtualenv, installed dependency tree, provider keys, LiveKit secrets, Docker image, or large model weights are included.**

The owner will install/manage **SGLang locally**. Do not automatically install GPU libraries, change CUDA/PyTorch, or run the historical AWS launchers. The optional Python gateway in this archive connects to an already running SGLang engine; it does not start or stop that engine.

## Instructions for the local AI agent

1. Read this README, `tts/README.md`, `.env.example`, `app/config.py`, and `02-VOICE-AGENT-PROMPT.md`.
2. Inspect the owner's OS, Python/Node versions, available ports, and existing SGLang endpoint. Linux or WSL2 with a supported NVIDIA setup is the closest match to the original deployment. The app environment was verified with Python 3.14; do not install its dependencies into the owner's GPU/model environment.
3. Preserve the requested model IDs: recognition **`gpt-live-transcribe`**, Responses and Decisions **`gpt-6-luna`**, local TTS **`attention_mlp_r16_early`**. Do not substitute another model. Ask only for missing credentials or the learned model files; do all other setup independently.
4. Use the supplied website snapshot as company knowledge. Keep its original numbers/facts; normalize numbers only in assistant output. Do not mix technical docs into company facts.
5. Configure self-hosted LiveKit and the local app. No LiveKit Cloud account is needed. Serve the app on port **8002**. Unlike the AWS deployment, the local default is IPv4 loopback, without a tunnel or AWS metadata calls.
6. Connect to the owner's SGLang endpoint through `tts/gateway.py` on port **3000**, or use an existing compatible port-3000 facade. Confirm actual audio streaming and the exact model/reference hashes. If SGLang is in Docker, bind-mount the reference and model paths so the engine sees the same absolute paths the gateway sends.
7. Run the tests and a real browser call. Clearly distinguish synthetic checks from microphone, voice-similarity, acoustic-noise, interruption, and second-network tests.

## What is included

| Path | Purpose |
|---|---|
| `app/` | Voice agent, Responses/Decisions integration, custom Realtime STT, streaming local TTS adapter, Saudi number normalization, gateway/backend. |
| `web/` | Bilingual TypeScript client, exact dependency lock, and prebuilt `dist/`. No `node_modules`. |
| `assets/voice/` | Eight real Noura clips, their manifest, prepared mono voice reference, and reference metadata. |
| `saudi_female_1minute-trimmed2.wav` | Owner-supplied original Noura recording; 9.46 seconds, stereo 44.1 kHz. |
| `knowledge/snapshot.json` | Verified rendered aljawab.ai content with URLs/timestamps/headings/hashes. |
| `tts/gateway.py` | Portable single-model implementation of the app's port-3000 HTTP contract. |
| `tts/model_config/` | Exact model config, tokenizer, shard index, and export/merge receipt. |
| `tts/model-artifacts.json` | Exact required model hashes/sizes and source-host paths. Large weights are excluded. |
| `tts/sglang.template.json` | Working SGLang-Omni pipeline settings, with the model path left for local configuration. |
| `tts/training_config.yaml` | Fine-tune/LoRA configuration for the selected early checkpoint. |
| `reference_aws/` | Original serving/export code, supporting Fish Speech code, and source snapshot from the actual SGLang-Omni package. Reference only; original host paths remain here. |
| `scripts/` | Local configuration/launcher, model verification/config rendering/transfer helpers, clip regeneration, website refresh. |
| `requirements.txt`, `requirements.lock` | Frozen app dependencies, excluding the GPU/model serving stack. |
| `MANIFEST.json` | Per-file hashes for archive contents. |

## Large learned model files are a separate transfer

To produce the same fine-tuned voice, source code and a generic S2 Pro base model are insufficient. You need the **exact merged `attention_mlp_r16_early` export**, including its codec. The three excluded weight files total approximately **10.3 GiB**; exact bytes/hashes are in `tts/model-artifacts.json`.

See `tts/README.md` for read-only copy/SCP commands and the checksum verification command. Do not replace the fine-tuned shards with generic base shards: they have different hashes. Noura is a per-request reference voice, not another fine-tuned checkpoint.

## Set up the local app

Use your own environment. No environment is in the ZIP:

```bash
cd aljawab-local
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional, only for refreshing the website snapshot:
python -m playwright install chromium

# Uses the included prebuilt web client, or rebuild from source:
npm --prefix web ci
npm --prefix web run build

# Creates a private .env and deployment/livekit.yaml with NEW local credentials.
python -m scripts.configure
```

Edit `.env` and set **your own `OPENAI_API_KEY`**. It must have access to all requested OpenAI models. Do not copy the production AWS key. Do not put any keys in the frontend. Generated LiveKit credentials are local and independent of the original deployment.

Install LiveKit Server **1.13.10** locally and put `livekit-server` on PATH, or set `LIVEKIT_SERVER_BIN` to its executable path. An optional Linux host-network Docker Compose definition is included. Server signaling is 127.0.0.1:7880; media uses UDP 7882 and TCP 7881. Keep these private for single-machine testing.

### If using the included TTS gateway

First configure/start your SGLang engine as described in `tts/README.md`. Then set:

```dotenv
SGLANG_URL=http://127.0.0.1:3002
SGLANG_MODEL_PATH=/absolute/local/path/to/attention_mlp_r16_early
TTS_URL=http://127.0.0.1:3000
TTS_STUDIO_DIRECTORY=./tts-data
TTS_VERIFY_MODEL_HASHES=true
```

The gateway registers the included Noura reference under the same identifier the cached clips use. The supplied `.env.example` already contains her exact reference ID/transcript/hash. **Do not regenerate or re-upload just to rename the voice.** Noura's actual voice identity is the WAV + transcript, not the label.

```bash
# SGLang must already be running. This starts only LiveKit, the gateway, and app.
python -m scripts.run_local --with-tts
```

Open **http://localhost:8002**, press Start, and grant the microphone. Localhost is suitable for browser microphone testing without a public HTTPS tunnel. Use `localhost` consistently instead of mixing origins. Press **Ctrl+C** to stop these local services; the owner-managed SGLang process remains running. Restart by rerunning the command.

### If you already have the original compatible TTS facade

Configure `TTS_URL` and `TTS_STUDIO_DIRECTORY` for your local facade. Register Noura using `python -m scripts.configure_noura` if it does not know her reference, then regenerate the clips. Launch `python -m scripts.run_local` without `--with-tts`. Do not expose the model server publicly.

### Separate-terminal alternative

```bash
# Terminal one: your existing SGLang process (managed by you).
# Terminal two:
livekit-server --config deployment/livekit.yaml
# Terminal three, if using the provided bridge:
python -m uvicorn tts.gateway:app --host 127.0.0.1 --port 3000 --no-access-log --log-level warning
# Terminal four:
python -m uvicorn app.server:app --host 127.0.0.1 --port 8002 --no-access-log --log-level warning
```

## Current behavior to preserve

- Voice **Noura / نورة**, conversational **Saudi Arabic**, feminine self-reference, English on request. The assistant is AI; do not claim it is the actual voice-reference speaker.
- Numbers are words before both transcript display and TTS: `160` → “one hundred and sixty” / “مية وستين”. Numeric fragments split across model deltas are held until the number is complete.
- Silero VAD: threshold **0.7**, minimum speech **220 ms**, end silence **550 ms**. Keep interruptions active; LiveKit AEC warmup blackout is disabled. Tune only after checking real room noise.
- Decisions: same `gpt-6-luna` key/model, **800 ms** deadline, earliest acknowledgment **350 ms**, cooldown **six seconds**, combined acknowledgment probability threshold **0.8**. Same-language variants share probability; ambiguity between two neutral clips should not suppress a clear acknowledgment. Never await Decisions on the answer generation path; answer-ready audio wins the output race.
- Arabic clips: “تمام.” and “فهمت عليك.” English: “Aha.” and “I see.” No additional “hmm/نعم/آها” clips were part of the delivered version.
- One active room, room-scoped five-minute tokens, in-memory chat state, no user recordings/transcripts logged. End clears state; abandoned admission expires after the lease/maintenance cycle.

## Maintenance and checks

```bash
python -m pytest -q
npm --prefix web run check
python -m scripts.verify_model /absolute/path/to/attention_mlp_r16_early

# Only with no active conversation:
python -m scripts.generate_clips
python -m scripts.refresh_knowledge
```

The initial app has **32 passing tests** on the original environment. Additional gateway contract tests in this bundle use a mock engine; they do not install a second GPU service. The ZIP is checked for integrity and production-secret leakage. The portable gateway is contract-tested, but its real GPU setup must be verified against your locally installed SGLang.

Cached files reject mismatched model/reference/checkpoint/settings hashes. If you register a new reference, regenerate all clips. The app's TTS adapter uses streaming `/api/stream` for live speech; `/api/generate` is for reusable clips.

The supplied compatibility gateway holds recent jobs in memory and writes temporary generated results under `tts-data/`; the app cleans its registered completed jobs. The historical AWS studio has additional metadata retention. Treat this as a testing app, not a strict end-to-end zero-retention system. Provider retention follows your OpenAI account controls.

For another computer/phone, localhost URLs no longer identify this machine: configure a reachable ICE address, HTTPS, secure cookies (`APP_SECURE_COOKIES=true`), and media rules. Do not reuse the original AWS public IP/tunnel. For local same-machine testing, keep the defaults. Do not silently disable cookie security on a publicly exposed deployment.

## Changes made for portability

Only the copied app is adapted: `TTS_URL` and TTS result directory are configurable; cookies default to local HTTP mode; local credential/config generation replaces IMDSv2; the launcher uses the current folder/current Python and never sudo/systemd. The selected model remains pinned. Original AWS sources live separately under `reference_aws/` and should never be executed blindly.
