# Build Aljawab AI's voice-to-voice test agent

You are the coding agent running directly on the existing AWS instance in Northern Virginia. Build and run the system described in this package. Inspect the instance and existing TTS implementation yourself. You have sudo access; use it only as needed. Complete the implementation and available checks, then provide the working browser URL and concise operating instructions. Do not stop at a plan or a mock interface.

Read `02-VOICE-AGENT-PROMPT.md` through `07-SOURCES.md` before implementation. These are product requirements and design intent; verify actual API contracts and library signatures against official documentation and installed versions.

## Required outcome

A user opens a friendly Aljawab-branded test page, presses Start, grants microphone access, hears an Arabic greeting, and converses naturally in Saudi Arabic or English. The assistant explains Aljawab using only information published on aljawab.ai. The user can interrupt speech, change language, mute the microphone, and end the session. One active conversation is sufficient. No bookings, lead collection, tickets, account lookups, or other customer actions.

## Stack and placement

- Existing locally hosted fine-tuned S2 Pro TTS with SGLang streaming. Reuse its exact model, voice, and serving setup.
- Self-hosted open-source LiveKit Server and LiveKit Agents on this instance. Do not introduce a LiveKit Cloud dependency or require a LiveKit account.
- OpenAI `gpt-live-transcribe` for live recognition and `gpt-6-luna` for streamed answer generation.
- An optional bounded Decisions API selector using `gpt-6-luna`, with cached neutral acknowledgment audio. It must never hold up the answer pipeline.
- Prefer a Python agent/backend and a lightweight TypeScript web client using the LiveKit client SDK. Keep implementation small. Reuse a suitable existing project if present.
- Serve the web application/backend on port 8002. Provide a secure public test URL using an HTTPS tunnel; prefer Cloudflare Quick Tunnel when suitable, with ngrok as an alternative if already configured.
- LiveKit media needs its own reachable transport. Implement public-IP advertisement and document the necessary AWS security-group rules. Do not attempt to send ordinary WebRTC UDP media through an HTTP tunnel.

## Work sequence

1. Inspect the workspace instructions, OS, services, listening ports, CPU/RAM/GPU headroom, package runtimes, and TTS code/configuration. Identify the TTS request/stream protocol, model/voice parameters, encoding, sample rate, and cancellation behavior. Keep unrelated services and the existing TTS healthy. Use a dedicated environment for new dependencies.
2. Write a short discovered-environment note with no secrets. Locate existing provider credential configuration without printing secret values. If a required key or AWS permission is absent, continue independent work and request only the missing item.
3. Verify current OpenAI and LiveKit contracts. Use the requested model IDs. Confirm account access with minimal live requests if credentials are available. Do not silently replace models. If the installed LiveKit plugin lacks `gpt-live-transcribe` support, build a small adapter around the documented OpenAI streaming interface and report the exact contract used.
4. Fetch and render https://aljawab.ai and its relevant public same-site pages. Create a local source snapshot containing readable text, headings, URLs, and retrieval timestamps. Check that actual business content was extracted. Use that material as the only factual company knowledge. Do not use these specification files as company facts. Never ship placeholder company claims as real knowledge.
5. Prove the audio pipeline in small steps: known TTS request, incremental decode/playback, live microphone transcription, streamed LLM text, and complete voice conversation. Implement phrase-boundary punctuation and cancellation correctly before adding filler behavior.
6. Implement website grounding, bilingual behavior, the friendly interface, and session isolation. Use `02-VOICE-AGENT-PROMPT.md` as the runtime instruction template.
7. Generate several short audio files through the actual TTS: greeting, Arabic/English acknowledgments, and neutral retry phrases. Cache by voice/model/audio settings. Integrate a single cancellable playout path. Do not include fake or silent audio placeholders as working assets.
8. Add the optional asynchronous Decisions selector and compare its latency against the baseline. Keep the feature switchable and fail open to silence. Do not replace generative answers with Decisions.
9. Start the local services and HTTPS tunnel. Verify signaling and actual two-way media from a remote browser if available. If AWS network changes require the owner, provide the exact minimal rules and continue everything else that can be validated.
10. Run the acceptance checks, fix failures, and document measured results and checks that still need a human microphone/browser. Deliver the running URL, service commands, configuration locations, stop/restart instructions, and limitations. Do not claim end-to-end validation unless a real browser audio exchange was observed.

## Implementation boundaries

Avoid modifying or restarting the model server unless the discovered integration actually requires it. Do not reinstall the system CUDA/PyTorch stack. Keep model files and unrelated services intact. Do not expose the TTS endpoint publicly just to integrate the agent; call it locally.

Use server-side secrets, generated LiveKit credentials, short-lived participant tokens, a room-scoped token endpoint, a one-session capacity limit, and basic session/token request limits. No provider keys or server secrets in client bundles. Use least privilege for the running app; sudo access during setup is not a reason to run every component as root. Keep SSH reachable when modifying firewall rules.

Store no user audio or transcripts by default. Runtime timing/error logs should omit transcript contents and credentials. Keep conversation state in memory and clear it when a session ends. A source snapshot and reusable non-user audio assets may persist.

Do not add developer metrics, model settings, raw API errors, networking controls, or infrastructure terms to the public interface. Put operational details in logs and the runbook.

## Expected implementation deliverables

- Working agent, backend, frontend, adapters, and local knowledge snapshot.
- Multiple real cached audio assets generated with the existing TTS, plus a regeneration command and asset manifest.
- Reproducible dependency definitions, configuration validation, and `.env.example` with names only.
- LiveKit and gateway configuration appropriate to the discovered instance.
- Start/stop/restart scripts or service definitions; do not kill unrelated processes.
- A knowledge refresh command, a concise operator README, and a verification report.

Make routine implementation choices yourself. Treat proposed timing numbers as initial tuning values, not guaranteed performance. If a blocker requires the owner, state the concrete blocker and the smallest next action while finishing unaffected work.
