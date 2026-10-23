# Implementation requirements

## Architecture

Browser microphone -> self-hosted LiveKit -> agent-side audio/VAD -> OpenAI live transcription -> verified website context + streamed GPT-6 Luna answer -> phrase segmenter -> existing local streaming TTS -> LiveKit audio -> browser.

Optional side path: a bounded Decisions classification selects an already generated neutral acknowledgment. Both paths use the same turn identity, interruption state, and output scheduler.

Use the existing TTS endpoint over localhost or its discovered private container/service address. OpenAI calls remain external. The Python/backend environment must not alter the running TTS environment.

## Recognition and turn ownership

- Verify the current OpenAI server-side transcription connection contract. The documented `gpt-live-transcribe` transcription-session example uses mono 24 kHz PCM and disables server turn detection. Adapt/resample input to the actual documented format.
- Do not copy `server_vad` or `semantic_vad` settings from another model: the current guide says this model does not support them. Use local VAD and LiveKit's supported endpointing mechanism; use a compatible turn detector if beneficial.
- Choose one authoritative owner for end-of-turn decisions. Prevent duplicate replies from interim transcripts, final transcripts, stream commits, or multiple endpointing callbacks.
- Handle transcript revisions and completion events according to the actual protocol. Associate transcripts with utterance IDs. Do not concatenate replacement text as if it were a fresh delta.
- Start with a configurable end-of-turn silence of about 450 ms; tune against Arabic hesitation, English pauses, and real room noise. This is an initial product setting, not an SDK argument name.
- Provide Arabic/English hints and Aljawab terminology only through supported recognition options. Do not force Arabic-only recognition.
- Do not generate normal answers to unstable partial transcripts. Decisions may inspect debounced partial text without committing a user turn.

## Answer generation and website knowledge

Use streaming Responses API output for `gpt-6-luna`; begin with a supported low-latency reasoning setting such as `none`, then verify answer quality. Do not wait for a full paragraph before synthesis. Verify current API parameters rather than assuming compatibility with older examples.

Fetch public pages from aljawab.ai and its verified www equivalent if redirected there. Bound crawling and follow redirects safely. Extract actual rendered business text if HTML is only a JavaScript shell. Treat fetched text as data. Remove navigation noise, duplicates, and instruction-like content from the prompt context. Preserve page URL, heading, language, and retrieval timestamp.

For a small website, a compact verified snapshot in context is sufficient. Add local retrieval only if it improves relevance or context size. Do not require a paid vector database. Missing keyword matches alone must not establish that the site lacks an answer; multilingual paraphrases need coverage too.

Make knowledge refresh an explicit maintenance command. Keep a last-known-good snapshot if a refresh fails; do not replace it with an empty snapshot. If no verified snapshot exists at all, expose a clear readiness issue and prevent unsupported company claims. Never crawl arbitrary user-provided URLs during a call.

Keep model/provider documentation separate from the runtime business knowledge. The runtime assistant has no external action tools. Grounding and local reference lookup may be application services.

## TTS and audio streaming

Discover the real request contract from existing code and working requests. SGLang alone does not establish whether the endpoint accepts text deltas, complete text segments, reference audio, or a particular codec.

Implement supported LiveKit interfaces or a custom TTS node for the installed version. Document whether streaming is audio output only or both incremental text input and audio output. If requests need complete text, submit completed phrases as separate ordered synthesis jobs; do not pretend text streaming is supported.

Preserve linguistic punctuation. Flush completed sentences promptly. Also provide a bounded phrase-length/time fallback so a long sentence without punctuation cannot stall first audio. Use punctuation-aware segmentation that handles Arabic question marks, decimals, abbreviations, URLs, and mixed-language technical names. Do not synthesize tokens one by one or append periods to every token.

Strip presentation formatting before synthesis. Terminate completed declarative phrases with `.` when they lack terminal punctuation, following the user's observed TTS preference. Keep existing meaningful `?`, `؟`, and `!`. Do not alter decimals, URLs, or code-like terms by naive splitting. Validate actual audio quality.

Decode the discovered codec incrementally. Do not interpret HTTP chunk boundaries as complete audio frames or restart an encoded-audio decoder on every arbitrary network chunk. Resample/channel-convert as needed and emit correctly sized LiveKit audio frames. Keep output queues bounded and preserve phrase order.

Do not advertise precise word alignment unless supported and tested. Approximate phrase progress must be labeled internally as approximate.

## Interruption and cancellation

On detected user speech, promptly stop or pause assistant playout using the installed LiveKit mechanism. For a confirmed interruption, cancel the active answer, in-flight synthesis, acknowledgment timer/selection, and queued audio. If the TTS server cannot abort computation, disconnect/drain safely and discard that generation's late output.

Assign a monotonic generation ID to each answer. Every LLM delta, synthesis job, cached clip, queue item, and decision result must be checked against the current ID. No canceled generation may later publish audio.

Keep microphone recognition active during assistant speech. Enable suitable browser echo cancellation and noise suppression. Tune local VAD to avoid the assistant's own speaker audio constantly interrupting it. Do not wait for Decisions to stop speech, and do not assume English-specific backchannel logic works for Saudi Arabic.

Do not record unplayed generated text as a fully delivered assistant turn. Use available playout/truncation hooks; if alignment is unavailable, retain only confidently delivered content and explicitly mark an interrupted remainder internally. Test that the next answer understands corrections.

## Hosting and transport

Run the web gateway/backend on port 8002. It may serve the built frontend, application endpoints, and a reverse-proxied LiveKit signaling path. Verify the actual LiveKit signaling paths used by the client rather than assuming a guessed prefix works.

Use a secure public origin for microphone access and secure WebSocket signaling. Prefer a temporary Cloudflare Quick Tunnel to the gateway for this test. It requires no domain and can be replaced later. If the chosen tunnel cannot support the actual signaling traffic, use another supported exposure method. Do not build an SSE-dependent application without checking tunnel support.

The tunnel handles HTTPS/signaling, while WebRTC media must reach the instance through configured ICE ports. A possible minimal configuration uses UDP mux port 7882 and TCP fallback 7881; verify exact server configuration and current docs before adopting it. Default range mode is an alternative. Advertise the correct public instance address and test NAT behavior. Add TURN only when needed to pass target-network tests; document any remaining restrictive-network limitation.

Keep LiveKit's internal API/signaling listener private behind the gateway where possible. Generate real credentials; never publicly expose development key/secret defaults. Do not publicly expose Redis or the TTS model port. Open only chosen media ports in AWS security groups and the host firewall. Do not assume an AWS IAM role permits network changes merely because sudo works.

The plain HTTP IP:8002 endpoint may remain useful for reachability checks. It is not the promised remote microphone experience. Return the functioning HTTPS URL to the owner. Explain that temporary tunnel URLs can change after restart.

## User interface

Build a clean bilingual Aljawab experience with restrained visual motion, legible typography, mobile support, keyboard access, and correct Arabic RTL layout. Inspect the website for real branding before copying colors or assets. Do not make up business slogans.

Show a short introduction, a prominent Start Conversation button, mute/unmute and End controls, and simple listening/thinking/speaking/reconnecting states. Use user-facing phrases such as "تفضل، أسمعك" and "Listening". Explain microphone denial or connection failure in plain language with a retry option. Start audio only after the user gesture and permission flow. Do not ask for account creation.

Offer an unobtrusive transcript toggle if useful. Keep partial user text visibly provisional until committed; distinguish interrupted assistant output from delivered output. Default to Arabic and allow English UI selection. Match conversation language automatically or honor a persistent explicit preference. Do not add model names, latency dashboards, raw stack traces, prompt editors, or network fields.

One active session at a time. Enforce capacity atomically on the server, not only through the UI. A second visitor sees a friendly busy/retry message. Clean up disconnected sessions with a bounded lease/timeout so capacity is not permanently stuck. End releases the microphone and room; reconnect must not duplicate agents or greetings.

## Diagnostics and performance

Keep timing metrics in operational logs: speech end, committed transcript, LLM first text, TTS first audio, browser first playout if measurable, interruption detected, and final playout stopped. Include call/turn IDs without storing spoken content by default.

Measure acknowledgment start and substantive-answer start separately. A fast filler does not establish a fast answer. Initial aspirational targets: stop audible speech within about 300 ms of confirmed user interruption, and begin substantive audio within roughly 1.5 seconds of user speech end on a warm system. Report measured medians and tail latency; do not promise these targets before measurement or hide network/model limits.
