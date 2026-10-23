# Documentation and source boundaries

Checked during preparation on 2026-10-17. Recheck current documentation and installed versions during implementation. These links are technical references; only aljawab.ai supplies customer-facing company facts.

## Company knowledge

- https://aljawab.ai — sole approved company source, including relevant public same-site pages. The research reader returned no extractable body content; implementation must render/fetch and verify the actual website. This package intentionally contains no fabricated company profile.

## OpenAI

- https://developers.openai.com/api/docs/models/gpt-live-transcribe — requested streaming speech recognition model.
- https://developers.openai.com/api/docs/guides/realtime-transcription — follow the current transcription-session contract; the documented GPT-Live-Transcribe example disables server turn detection and uses 24 kHz PCM. Do not transfer VAD settings from other models.
- https://developers.openai.com/api/docs/models — verify `gpt-6-luna`, supported APIs, and current account availability.
- https://developers.openai.com/api/docs/guides/latest-model — verify model parameters and supported reasoning settings.
- https://developers.openai.com/api/docs/guides/decisions — bounded choice/predicate/score requests; use for optional acknowledgment selection, not arbitrary answer generation.
- https://developers.openai.com/api/docs/guides/decisions-voice — native GPT-Live delegation is a distinct integration. Do not assume it automatically applies to a separate LiveKit + transcription + LLM + custom TTS pipeline.

## LiveKit and browser connectivity

- https://docs.livekit.io/transport/self-hosting/ — self-hosted server and agents without requiring LiveKit Cloud.
- https://docs.livekit.io/transport/self-hosting/ports-firewall/ — media ports and transport configuration; HTTPS exposure alone is insufficient for media.
- https://docs.livekit.io/agents/ — Agents framework and supported integration concepts.
- https://docs.livekit.io/agents/models/tts/ — custom speech integration and provider adapters; verify compatibility with the existing local endpoint before reusing a provider plugin.
- https://docs.livekit.io/agents/logic/turns/ — turn handling and interruption behavior.
- https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia — microphone permission and secure-context requirements.
- https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/ — temporary HTTPS exposure for testing, without requiring a domain/account. Verify limitations against the actual web and signaling protocols.

## Local evidence required

The existing TTS source, configuration, working request, and actual streamed response are authoritative for model integration. Do not assume a Fish Audio hosted-service API matches this fine-tuned SGLang endpoint. Record discoveries without secrets.
