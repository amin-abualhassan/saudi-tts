# Smooth speech, cached audio, and fast decisions

## Purpose

The user has observed that punctuation, especially a final period, improves this TTS. Preserve that preference and verify it with the actual fine-tuned voice. Generate reusable short audio clips through the running TTS before conversations, then play them without a synthesis request when appropriate.

Decisions provides bounded classifications/selections. It does not generate the substantive spoken answer. The proposed integration below is an application-side design for this STT/LLM/TTS pipeline, not an assumption that native GPT-Live delegation is available in `gpt-live-transcribe` or LiveKit.

## Initial asset manifest

Generate separate files, using a compatible decoded or lossless playback format determined by the adapter. Keep exact text, voice/model revision, settings, format, sample rate, channels, duration, and content hash in a manifest.

| ID | Exact synthesis text | Intended use |
| --- | --- | --- |
| greeting_ar | حيّاك الله، معك المساعد الصوتي للجواب. كيف أقدر أساعدك؟ | Once after Start, Arabic default |
| greeting_en | Hello, I'm Aljawab's voice assistant. How can I help you? | Once when English selected |
| ack_ar_tamam | تمام. | Neutral acknowledgment, Arabic |
| ack_ar_understood | فهمت عليك. | Alternative neutral acknowledgment, Arabic |
| ack_en_aha | Aha. | Neutral acknowledgment, English |
| ack_en_understood | I see. | Alternative neutral acknowledgment, English |
| retry_ar | ما سمعتك بوضوح. ممكن تعيد سؤالك؟ | Confirmed unintelligible speech |
| retry_en | I didn't catch that clearly. Could you repeat your question? | Confirmed unintelligible speech |

Do not promise that Arabic text pronounced as "aha" or a specific English spelling sounds natural without listening. Add variants only after validating voice continuity and tone. Cache generation happens during setup or a maintenance command, not concurrently with a live test on a capacity-constrained model. Do not normalize loudness so aggressively that clips become distorted.

These files contain no user recordings and can persist. Invalidate/regenerate them when the model, reference voice, pronunciation settings, or audio format changes. If an asset is missing, skip that optional clip; do not block conversation startup for an acknowledgment.

## Acknowledgment policy

Default to no acknowledgment when the substantive answer is already about to play. After a completed, intelligible user turn, a single neutral cached clip may cover a short wait. Start with an eligibility delay around 350 ms after the committed turn and a cooldown around 10 seconds. These are configurable initial tuning values, not guaranteed best values.

For the first build, do not play acknowledgment audio over an actively speaking user. Partial-transcript classification may prepare a choice, but playback requires a valid completed turn. Never play a clip merely because audio/VAD fired: silence, noise, incomplete speech, requests to stop, and emotionally sensitive complaints should default to silence.

Do not imply approval or agreement. "تمام." means understanding here, not confirmation that an unsupported price, promise, or allegation is true. Avoid a cheerful "Aha." after a complaint. Do not fill every pause, loop fillers, or synthesize arbitrary classifier text.

Both cached clips and real answers go through the same output scheduler. Do not overlap them. If substantive audio becomes ready before a clip starts, discard the clip. If a clip has started, allow its short natural end unless user interruption occurs, then immediately continue the answer. Measure whether it increases substantive latency; reduce frequency or disable it if so.

## Optional Decisions side path

1. At a stable partial transcript or committed turn, start at most one debounced classification per active turn. Keep the relevant transcript/context small. Do not send images; the user's image example illustrates the API shape, not a requirement for this voice app.
2. In parallel, continue ordinary end-of-turn processing and answer generation without awaiting Decisions.
3. Ask a bounded choice question, for example `acknowledgment`, with fixed allowed values `none`, `ack_ar_tamam`, `ack_ar_understood`, `ack_en_aha`, and `ack_en_understood`. Instruct the classifier to choose `none` for uncertainty, ongoing/incomplete speech, stop/correction requests, complaints/distress, and cases where an acknowledgment could imply agreement.
4. Parse the documented typed answer, not free-form JSON fabricated by a completion prompt. Handle refusal, unknown answer type, invalid choices, network errors, and unavailable API access by selecting `none`.
5. Match result to session ID, generation ID, transcript revision, and language. Discard stale results and results that arrive after substantive audio is ready. Recheck local playback eligibility immediately before enqueueing.
6. Begin with a short application deadline around 250 ms and no retries on this path. A late classification is skipped rather than delaying anything. Tune or disable using actual measurements. An optional calibrated confidence threshold may further suppress choices; confidence is not a universal correctness guarantee.

The model ID remains `gpt-6-luna`. Verify the installed SDK supports `client.decisions.create` or the asynchronous equivalent and current question schemas. Use an async client/nonblocking integration in the agent. Keep the feature configurable and optional so a missing Decisions entitlement cannot disable the required voice pipeline.

Do not use Decisions as the sole speech activity detector, transcript finalizer, factual knowledge checker, or interruption controller. Immediate cancellation must be local.

## Fair comparison

Evaluate the same representative Arabic and English conversations with cached acknowledgments disabled, enabled without the classifier where locally safe, and enabled with Decisions. Measure audible acknowledgment time, substantive answer time, interruption response, unnecessary/incorrect filler rate, and extra API calls. Keep the best measured configuration; do not label Decisions faster solely because it returns a small answer.
