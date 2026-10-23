# Aljawab AI voice agent

Running test application: **https://achievements-affiliate-pope-pepper.trycloudflare.com**. The numbered files below are the original requirements; the operator instructions below describe the deployed implementation.

Open the link, choose Arabic or English, press Start, and allow your microphone. The optional transcript stays in the page and is cleared at End. Multiple conversations can run at once, including separate tabs in the same browser. Each closes after 60 seconds without detected user speech, even if the browser keeps sending heartbeats.

## Files

- `01-BUILD-PROMPT.md`: main implementation prompt and execution order.
- `02-VOICE-AGENT-PROMPT.md`: customer-facing assistant instructions.
- `03-TECHNICAL-SPEC.md`: streaming, networking, knowledge, and interface requirements.
- `04-SPEECH-AND-DECISIONS.md`: punctuation, cached audio, and optional fast classification.
- `05-ACCEPTANCE-CHECKS.md`: functional checks and delivery requirements.
- `06-CONFIGURATION.example.yaml`: application configuration intent; implement and validate the real schema before using it.
- `07-SOURCES.md`: official documentation to verify during implementation.

## Agreed scope

Testing only. Multiple independent conversations (updated 2026-10-18 at the owner's request). Self-hosted LiveKit, with no LiveKit Cloud account required. Existing fine-tuned S2 Pro TTS served with SGLang on the same AWS instance in Northern Virginia. OpenAI `gpt-live-transcribe` for speech recognition and `gpt-6-luna` for spoken answers. Calm, confident Saudi Arabic and English. Answers about Aljawab must come exclusively from https://aljawab.ai. No customer-facing actions or external integrations.

The web application must use port 8002 on the instance. A working browser microphone requires a secure browser origin; supply an HTTPS tunnel for remote testing. LiveKit media uses separately configured reachable ports. The owner can open additional AWS ports.

## Defaults chosen for the first implementation

Arabic-first greeting, automatic language matching thereafter, a simple bilingual interface, optional in-session transcript, no saved user recordings or transcripts, and short neutral acknowledgments generated once using the existing TTS. These defaults remain configurable.

## What must be discovered on AWS

The existing TTS protocol, audio encoding, sample rate, model/voice reference, cancellation behavior, instance resources, installed software, public address, network rules, and available provider credentials. Do not invent any of these or include secrets in reports.

No company facts are included here because the website's rendered body was not available through the research reader. The implementing agent must fetch or render the website, build a verified local knowledge snapshot, and never substitute invented marketing copy.

Prepared on 2026-10-17. Documentation links are starting points, not a substitute for checking the installed SDK versions and live service contracts.

## Operate the deployment

From `/opt/dlami/nvme/speech-agent`:

```bash
bash /opt/dlami/nvme/speech-agent/start.sh
# Attach to the application console; detach with Ctrl-A, then D.
screen -r speech-agent

scripts/ctl.sh status
scripts/ctl.sh start
scripts/ctl.sh stop
scripts/ctl.sh restart
```

`start.sh` starts the dedicated LiveKit, application and HTTPS tunnel services, waits for application health, and prints the current public URL. The application runs inside screen session `speech-agent`, supervised by `aljawab-app.service`, and starts at boot. The command is safe to repeat. `scripts/ctl.sh restart` restarts the application and its screen session, preserving a running tunnel. Stopping and restarting the tunnel creates a **new URL**; `status` prints the current URL. The existing SGLang TTS on port 3000 must be running separately.

Every conversation has its own LiveKit room, agent, credentials, in-memory messages and input timer. Tabs use independent in-memory session tokens so shared cookies do not mix conversations. Detected user speech resets the input timer, including during a long utterance; silence, background microphone packets, assistant speech and browser heartbeats do not. `SESSION_IDLE_TIMEOUT_SECONDS` defaults to 60. A lost browser's heartbeat lease also expires after 60 seconds. `SESSION_START_LIMIT_PER_MINUTE` defaults to **60 starts per public IP per minute**, raised from eight to accommodate shared Wi-Fi and carrier NAT. There is no single-conversation admission limit. All conversations share the existing TTS queue, so simultaneous synthesis can increase latency; unlimited capacity has not been established.

Mobile audio is unlocked directly in the Start tap. Playback permission cannot
hold connection setup indefinitely: when a browser blocks playback, **Enable
audio** appears and resumes sound without ending the call. Admission throttling,
microphone access and audio-network failures have separate messages. Failed
allocated calls report only enumerated setup stage/reason and coarse platform,
once per stage; credentials, addresses and spoken text are excluded.

The application is on **http://[::1]:8002**. An unrelated application already owns `0.0.0.0:8002`; it was preserved. The public IPv4 address on port 8002 therefore serves that existing application. Cloudflare forwards Aljawab HTTPS to IPv6 loopback port 8002. LiveKit media travels directly to this instance, separately from the HTTPS/signaling tunnel.

## Networking

Instance `i-0bf8e6a113fff4d51`, public IP `98.83.42.166`, security-group name `Majed-dev`: allow inbound **UDP 7882** and **TCP 7881** from test devices' public IP addresses. Use `0.0.0.0/0` only when the audience cannot be scoped. No inbound application/signaling ports are needed for the tunnel. Preserve SSH; do not expose 7880, Redis, or additional model ports.

An authenticated **TURN/TLS fallback on TCP 443** is now advertised as
`turns:98.83.42.166:443?transport=tcp`. Direct UDP/TCP media remains available.
TLS terminates in the dedicated `aljawab-turn-proxy` HAProxy service and forwards
to embedded LiveKit TURN on 5349 using PROXY v2, trusted only from loopback.
Public access to 5349 is unnecessary. The existing Cloudflare URL remains the
website/signaling endpoint; a regular browser cannot use that HTTP tunnel as
the raw TURN relay. No DNS change or LiveKit Cloud account is required.

The relay uses a publicly trusted Let's Encrypt certificate for the IP itself,
currently expiring **2026-10-25**. Certificate state/keys are root-only in
`/etc/letsencrypt-aljawab/` and `/etc/aljawab-turn/`, outside the checkout.
`aljawab-turn-renew.timer` checks renewal twice daily. Keep TCP 80 available for
its standalone ACME challenge and TCP 443 reachable for relay clients. The
deploy hook reloads HAProxy gracefully, leaving active LiveKit calls running.
`start.sh` also starts the relay and renewal timer when installed; `ctl.sh status`
shows them. Root service definitions and the deploy hook are in `deployment/`
and `scripts/deploy_turn_certificate.sh`; the separate `.acme-venv` uses
`deployment/acme-requirements.lock`. A public-IP change requires a matching new
certificate and TURN address; `scripts/configure.py` preserves the relay on
ordinary reruns and refuses to silently retain an IP certificate for a new IP.

The IAM role cannot inspect/change EC2 security groups. Public certificate
validation and forced TURN/TLS browser calls succeeded, but physical phones on
the affected carrier/Wi-Fi still need a retry. See [verification](VERIFICATION.md)
for the distinction between browser emulation and real mobile-network testing.

## Required TTS model and exact contract

Every request explicitly selects **`attention_mlp_r16_early`**. Configuration rejects other models, with no fallback. Its checkpoint SHA-256 is `9dbf9c8b6ee71dc5c737e6a76f2166539e833a6aa8ab274fb192553714a294cf`.

The owner's `bash /opt/dlami/nvme/training/jordanian_s2pro/start_voice_server_sglang.sh` delegates to `sglang_launch.py`. The running process serving port 3000 is `sglang_server.py`, which routes this model to its existing engine on loopback 3002. The other model on 3001 is never selected by Aljawab.

Live conversation uses **POST `http://127.0.0.1:3000/api/stream`**. The request contains `model`, `reference_id`, `text`, `seed`, `max_new_tokens`, `temperature`, and `top_p`; voice reference is the uploaded **Noura (نورة)** recording, `saudi_female_1minute-trimmed2.wav`, with the exact owner-supplied Arabic reference transcript. The response is SSE with base64 little-endian signed 16-bit mono PCM, 44,100 Hz. Uploaded references also require `reference_text`. The adapter verifies each accepted job's selected model, reference audio hash, and checkpoint via `/api/status` before yielding speech. SSE stays local; the browser does not depend on SSE through Cloudflare.

This endpoint accepts complete phrases and streams audio output. It does not accept incremental text tokens. The phrase segmenter preserves punctuation, decimals, abbreviations, and URLs, and flushes long/unpunctuated phrases on a bounded time/length fallback. Audio is repacketized into 20 ms LiveKit frames independently of HTTP chunk boundaries.

Cached clips use **POST `/api/generate`**, poll **GET `/api/status`**, and download **GET `/api/audio/{job_id}.wav`**. All eight real assets, settings, voice, model, durations, and hashes are in [assets/voice/manifest.json](assets/voice/manifest.json). Missing optional acknowledgments are skipped; model/reference/hash mismatches reject stale assets.

The welcomes have long quiet gaps shortened offline: gaps of at least 300 ms
below −50 dBFS RMS retain 200 ms, with a 5 ms quiet crossfade. Speech outside
these gaps is unchanged. The regenerated, friendly-tone welcomes are 4.53 s
Arabic and 4.04 s English. Original
greetings and manifest are preserved in `assets/voice/archive/greeting-pauses-v1/`.
Future greeting generation applies the same preparation. No live buffering is
added, and conversational replies and acknowledgment clips are unaffected.

The existing service has no per-job computation cancellation endpoint. Interruption closes the stream, cancels the answer and acknowledgment, discards stale output by generation ID, and immediately interrupts LiveKit playout. Its worker may finish the current synthesis in the background.

Each completed reply sentence is submitted as a separate streaming TTS request.
After splitting, the application appends exactly one **`[short breath]`** to
each request, including intermediate sentences. Breath placement is absent
from the system prompt. Any legacy/model-authored breath cues are normalized
to one final cue, preventing duplicates or cue-only synthesis. This happens
only on the synthesis branch, so manually added cues never enter the LLM's
text or UI transcript. The LLM may still choose one `[friendly]`, `[calm]` or
`[empathetic]` cue at a sentence's start. Existing transcription filtering
also hides complete, split and interrupted model-authored cues. The sentence
splitter waits for complete sentences rather than using its earlier
150-character/650-ms fallback; normal punctuation precedes the trailing cue.

The exact Noura reference transcript, selected checkpoint and sampling settings
remain the same. Cached welcome generation prepends `[friendly]` and adds one
final `[short breath]` to the synthesis request, keeping the visible greeting
text plain. Cache validation includes this request, so older tone settings
regenerate automatically.
Cached acknowledgments retain their delivery. Breaths address the reported
clipped final sound experimentally; perceived completion still needs listening.

The runtime prompt asks for selective Arabic diacritics on ambiguous or
pronunciation-prone words, without fully vocalizing replies or adding formal
case endings. In the sense of written/transcribed text it explicitly requires
`النَّصّ` (fatha on ن), including related forms such as `نَصّ` and `للنَّصّ`.
The model authors these marks and the synthesis pipeline preserves them.

Streamed playback now passes through a **250 ms trailing RMS loudness window**,
then an in-process FFmpeg compressor and limiter using the installed PyAV
library. The window uses audio already received, so it adds no buffering or
lookahead. It replaces the earlier fixed quiet boost. Its −24 dBFS RMS target,
2 dB deadband and partial correction preserve small inflections while correcting
sustained fades. Gain is bounded to +9/−6 dB and changes smoothly, at up to
24 dB/second upward and 40 dB/second downward, with sample interpolation.
A 20 ms activity window stops settled material below −48 dBFS being boosted.
Existing gain holds through quiet gaps shorter than 80 ms, then releases;
this avoids losing gain on short unvoiced consonants between fading words.
It is level-based, not semantic speech/breath classification. Positive boost
is suppressed for the first 120 ms of active speech. Sentence markers clear
the level estimate and return its gain to neutral so a fading ending's boost
does not carry into a fresh attack. The native filters stay continuous across
phrases and acknowledgments; cached greetings use the same processing.
There is no pitch shifting, time stretching or hard gating. The compressor reduces loud/quiet swings
and the limiter caps pre-encoding sample peaks at **−6 dBFS**. It processes PCM
as it arrives, with a **1 ms lookahead**, no subprocess, no complete-clip wait
and no wait for future analysis samples. On normal completion the delayed tail is
drained without changing the sample count; interruption discards it immediately.

Sentence starts also receive a **40 ms soft attack after the limiter**. The
first audible sound begins at −12 dB relative gain and rises on a smooth cosine
curve to normal level. This is a playback gain ramp, not a 40 ms buffering
delay. Leading silence does not consume the ramp. Local sentence-boundary
markers trigger it even when a TTS/network pause contains no PCM silence, and
the marker is aligned past the limiter's delayed previous-sentence tail.
Automatic retriggering requires about 180 ms of near-silence, so brief word
gaps do not restart it. No samples are removed, and no fade-out is applied to
the preceding word.

`AUDIO_DYNAMICS_ENABLED=true`, `AUDIO_PEAK_DBFS=-6`,
`AUDIO_LEVELER_ENABLED=true`, `AUDIO_LEVELER_WINDOW_MS=250`,
`AUDIO_LEVELER_TARGET_DBFS=-24`, `AUDIO_LEVELER_MAX_BOOST_DB=9`,
plus `AUDIO_ONSET_MS=40` and
`AUDIO_ONSET_ATTENUATION_DB=-12`, are the defaults in
`app/config.py`; override them in `.env` and run `scripts/ctl.sh restart` to
change or bypass playback processing. These settings affect playback, including
greetings and acknowledgments, rather than the model's original generated WAVs
or transcripts. dBFS describes digital signal level, not the listener's speaker
volume. Live measurement and listening evidence is recorded in
[VERIFICATION.md](VERIFICATION.md).
Set `AUDIO_LEVELER_ENABLED=false` to restore the previous fixed quiet curve
(`AUDIO_QUIET_BOOST_DB=3`). With the leveler disabled, setting
`AUDIO_QUIET_BOOST_DB=0` bypasses that legacy curve too.
Set `AUDIO_ONSET_MS=0` to bypass only sentence-start smoothing.

## Configuration and maintenance

Server-only `.env` (mode 600) contains `OPENAI_API_KEY`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET`. The **same OpenAI key** serves recognition, Responses, and Decisions. Names/settings are in [.env.example](.env.example); the validated schema is [app/config.py](app/config.py). Generated `deployment/livekit.yaml` is mode 600; its secret-free template is [deployment/livekit.example.yaml](deployment/livekit.example.yaml).

Recognition uses `gpt-live-transcribe`, 24 kHz mono PCM, Arabic/English hints, and server turn detection disabled. Local Silero VAD requires 220 ms of speech at activation threshold 0.7 and commits after 550 ms of silence; those speech events are LiveKit Agents' sole turn owner. Answers use streaming Responses with `gpt-6-luna`, reasoning `none`, `store=false`, and no external action tools.

Decisions uses **`gpt-6-luna`**, asynchronously beside the answer pipeline, with no retries and an 800 ms deadline. Failure, refusal, low confidence, stale results, or substantive audio being ready suppresses the acknowledgment. One clip may play after 350 ms, with a 6-second cooldown. `DECISIONS_ENABLED=false` or `ACKNOWLEDGMENT_ENABLED=false` disables it. It cannot hold up answer generation or synthesis; an already playing short clip may finish naturally before the answer.

```bash
# Refresh rendered approved website content; failure preserves the prior snapshot.
.venv/bin/python -m scripts.refresh_knowledge
scripts/ctl.sh restart

# Regenerate reusable clips while calls are stopped.
sudo systemctl stop aljawab-app
.venv/bin/python -m scripts.generate_clips
sudo systemctl start aljawab-app

# Automated checks; fixtures are synthetic, not human microphone checks.
.venv/bin/python -m pytest tests -q
# Real concurrent calls, generated speech, and idle expiry (about 80 seconds).
.venv/bin/python -m scripts.check_connections
# Two browser tabs sharing cookies; silent microphone and idle UI cleanup.
.venv/bin/python -m scripts.check_browser_connections
```

Knowledge is [knowledge/snapshot.json](knowledge/snapshot.json), retrieved from the homepage and technologies route on 2026-10-17. Each page includes URL, headings, retrieval time, language metadata, content hash, and rendered text. Technical documentation is separate. Website metrics/compliance statements are treated as published claims, not independent guarantees.

Noura is registered with `.venv/bin/python -m scripts.configure_noura`, which sets `TTS_REFERENCE`, `TTS_REFERENCE_TEXT`, `TTS_REFERENCE_SHA256`, and `TTS_VOICE_NAME` in `.env`. The reference is bound to the source recording hash in `assets/voice/reference.json`; a missing reference fails rather than falling back to a different voice. Stop calls before registering/regenerating clips, then restart. The previous public clips are retained under `assets/voice/archive/single_speaker/`. `scripts/configure.py` generates LiveKit secrets/config and discovers the public IP using IMDSv2; rerun after an IP change and restart only the dedicated LiveKit service.

`scripts/install.sh` reproduces setup using `requirements.lock`, `web/package-lock.json`, LiveKit Server 1.13.10, Node 22.21.1, and Cloudflared 2026.10.1. Binary hashes are in `deployment/binaries.sha256`. The app uses Python 3.14, LiveKit Agents 1.8.8, and OpenAI Python 3.33.0. No CUDA, PyTorch, or existing model environment was changed.

## Privacy and evidence

The application does not record microphone audio, persist user transcripts/browser history, enable room recording, or log transcript contents. Context is in memory and released at End, after 60 seconds without detected user speech, or after a lost-session heartbeat lease of 60 seconds. Provider retention depends on account data controls, even though Responses storage is disabled.

**Inherited model-service retention:** the existing studio saves synthesized assistant audio and job metadata. Aljawab registers its own job IDs and deletes only those result files after completion, including interrupted jobs, on a five-second cycle. The studio still keeps recent generated-text metadata in its own memory and `voice/status.json`. This behavior was preserved to avoid modifying either existing service. Stopping Aljawab before a canceled model job finishes can leave that generated audio until the next app start/cleanup. This is not a strict end-to-end zero-retention deployment; use non-sensitive test speech.

Operational logs are in `logs/`; provider/LiveKit content logs and URL access logging are suppressed. Browser evidence in `tests/results/` contains only generated test speech and answers, not visitor sessions. See [VERIFICATION.md](VERIFICATION.md) for tests/timings and remaining human checks, and [ENVIRONMENT.md](ENVIRONMENT.md) for discoveries.

## Noura voice and speaking update

The assistant is Noura (نورة), uses conversational Saudi Arabic with feminine self-reference, and spells out numbers in Arabic/English before both transcript display and synthesis. Numeric chunks are held across LLM deltas so a split `256` becomes one whole number rather than three separate digits. Percentages, decimals, Arabic-Indic digits, grouped thousands, and negative values are supported. The website snapshot retains its original facts/notation. Saudi wording and voice similarity still require listening in the target browser.

The runtime system prompt preserves embedded words and names in their original
language/script: English terms stay in Latin letters in Arabic answers, and
Arabic terms stay in Arabic script in English answers. It verbalizes addresses
and technical separators for speech, for example `aljawab.ai` becomes
“Aljawab dot AI” and `contact@aljawab.ai` becomes “contact at aljawab dot ai”.
The latter is a formatting example only; company contact details still require
verified website evidence. Source citations and raw clickable links are omitted
from spoken replies.

Silero VAD remains local; sensitivity is slightly lower: activation threshold `0.7` (was `0.6`), minimum speech `220 ms` (was `120 ms`), and end silence `550 ms` (was `450 ms`). These are configurable via `VAD_ACTIVATION_THRESHOLD`, `VAD_MIN_SPEECH_DURATION_MS`, and `END_OF_TURN_SILENCE_MS`. Interruptions remain enabled and Decisions stays asynchronous.

Acknowledgment tuning: the selector now has an 800 ms budget, a threshold of 0.8 for the combined probability of an appropriate acknowledgment, and a six-second cooldown. Playback still begins no earlier than 350 ms and is skipped as soon as substantive audio is ready. The scheduler now also accepts a selection that arrives after the eligibility timer, provided answer audio is still pending.

The acceptance threshold uses the combined probability of the language-matched acknowledgment variants. A clear acknowledgment decision is no longer suppressed merely because “تمام” and “فهمت عليك” share probability. Selection of silence and complaint/correction safeguards remain in force.
