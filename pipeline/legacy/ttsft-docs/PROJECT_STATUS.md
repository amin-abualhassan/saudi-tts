# Project status

Last updated: 2026-10-18 (UTC).

## Friendly welcome delivery (2026-10-18)

The owner found the welcome robotic and requested a tone tag. Added a private
`[friendly]` prefix to Arabic/English cached welcome synthesis and regenerated
both, retaining the final breath, short-pause preparation, voice/checkpoint,
and Arabic ending "كيف أقدر أساعدك؟". Display text remains untagged; other six
cached assets are unchanged. Durations are 4.532/4.040 seconds. Source final
100 ms and full playback sample counts are preserved.

All 102 tests pass, including invalidation of old breath-only caches without
changing display text or other clips. Both welcomes pass live browser playback
with hidden tags, no page errors and the −6 dBFS cap. App is active in screen
with the same URL. See [generation report](speech-agent/tests/results/welcome-friendly-20261018.json),
[playback report](speech-agent/tests/results/welcome-friendly-playback-20261018.json),
and [verification](speech-agent/VERIFICATION.md). Perceived naturalness needs
the owner's listening check.

## Arabic welcome wording updated (2026-10-18)

At the owner's request the Arabic welcome now ends with "كيف أقدر أساعدك؟".
Updated the cached text and runtime greeting example, and regenerated Arabic
with its hidden final breath and short-pause preparation. New duration is
4.636 s. Other seven cached assets are unchanged; asset hash/full playback
sample count validate. App is active in screen with the same URL. See
[generation check](speech-agent/tests/results/welcome-kaif-20261018.json).

## Welcome audio regenerated with a final breath (2026-10-18)

The owner reported the welcome ending was clipped. Regenerated both Arabic
and English cached greetings with one final `[short breath]`, retaining the
same voice/model/checkpoint/settings and offline short-pause preparation.
Generator cache validation now includes the tagged synthesis text, while the
visible greeting remains plain. Only the two welcomes regenerated; six other
cached assets are unchanged. Prior welcomes/manifest are archived.

New durations are 4.781 s Arabic and 3.437 s English. Final 100 ms are preserved
exactly from the generated source. All 101 tests pass, including cache
invalidation/idempotence and unchanged unrelated clips. Both actual greetings
pass browser playback, return to listening without exposed tags/errors, and
retain complete sample counts through the −6 dBFS playback chain. App is healthy
in screen `speech-agent` (PID 1893034), with the same public URL. See
[generation and browser evidence](speech-agent/tests/results/welcome-breaths-20261018.json)
and [verification](speech-agent/VERIFICATION.md). Preferred perceived word
completion still needs the owner's listening check.

## Multiple users and mobile audio relay (2026-10-18)

The owner reported mobile users loading the page but failing/stalling on Start.
Confirmed separate concurrent sessions and cache-bypassed public responses.
Raised the shared-public-IP admission allowance from eight to a configurable
60 starts/minute, began playback unlock within the Start gesture, added an
Enable audio recovery button and distinct retry/network/microphone messages.
Authenticated failure reports retain only enumerated stage/reason and coarse
platform. The original affected-device error was not captured; these findings
do not establish one exclusive root cause.

Added authenticated TURN/TLS on **98.83.42.166:443** with a publicly trusted
Let's Encrypt IP certificate, so DNS changes and LiveKit Cloud are unnecessary.
HAProxy forwards TLS-decoded traffic with trusted-loopback PROXY v2 to embedded
LiveKit TURN. Certificate state/keys are root-only outside the checkout.
Persistent renewal checks run twice daily, with graceful proxy reload instead
of LiveKit restart. One-command start/status handles the additional services;
config regeneration preserves TURN and refuses unhandled IP/certificate drift.
Direct media, TTS services and the public Cloudflare website URL are preserved.

All **100 backend tests**, TypeScript checking and production build pass.
Three isolated Chromium clients connect concurrently, both normally and with
direct candidates disabled; every forced connection selects **relay/TLS**.
Audio and active calls survive proxy reload. Simulated autoplay denial recovers
without closing the call, and rate/media failures show distinct messages.
A complete Arabic voice exchange also passes over forced relay, with hidden
cues and two separately suffixed TTS requests. These are emulated mobile
profiles on the server's network, not physical Safari/carrier tests: affected
phones must retry. See [relay evidence](speech-agent/tests/results/turn-relay-connections-20261018.json),
[full speech check](speech-agent/tests/results/turn-relay-speech-ui-20261018.json),
[network audit](speech-agent/tests/results/mobile-network-audit-20261018.json),
and [verification](speech-agent/VERIFICATION.md).

## Selective Arabic diacritics for pronunciation (2026-10-18)

The owner reported النص pronounced with the wrong vowel. The runtime prompt
now explicitly requires `النَّصّ` (fatha on ن) when referring to written or
transcribed text, including related forms, and asks the LLM to add selective
diacritics to other words prone to mispronunciation. It retains conversational
Saudi speech without fully vocalizing replies or adding formal case endings.
No manual word replacement or TTS model change was introduced.

A live `gpt-6-luna` check of the owner's unvocalized sentence produced the
requested `النَّصّ`; it survives phrase splitting, private breath insertion and
TTS payload preparation. App restarted in screen with the same URL. See
[model/payload check](speech-agent/tests/results/selective-diacritics-20261018.json)
and [verification](speech-agent/VERIFICATION.md). Actual pronunciation still
requires listening; this check establishes the text sent to synthesis.

## Manual trailing breath for each streamed TTS sentence (2026-10-18)

The owner clarified that `[short breath]` should be added in code after each
TTS split instead of being requested from the LLM. Previously ordinary replies
requested only one final breath, leaving intermediate requests untagged. The
application now appends exactly one canonical breath immediately before every
split reply sentence's `Speech.stream` call. Existing model-authored breaths
are normalized to avoid duplicates. Emotion cues remain optional LLM choices;
all breath instructions/examples were removed from the runtime system prompt.
The suffix exists only on the synthesis branch and never enters the UI or LLM
history. Cached welcomes and acknowledgments are unaffected.

All 95 tests pass, including three untagged mixed-language sentences producing
three independently suffixed requests, cue idempotence and transcript hiding.
Live Chromium produced two TTS jobs with exactly one final breath each, clean
transcripts and advancing playback. App is healthy in screen `speech-agent`
(PID 1862044), with zero automatic restarts and the same public URL. See
[browser evidence](speech-agent/tests/results/manual-breaths-ui-20261018.json)
and [verification](speech-agent/VERIFICATION.md).

## Shorter welcome pauses (2026-10-18)

The owner reported excessive pauses in the welcoming phrase. Cached Arabic
audio had three near-silent gaps of 700, 510 and 630 ms; English had one of
400 ms. Shortened each to 200 ms offline, joining only inside the quiet region
with a 5 ms crossfade. Arabic duration falls from 5.851 to 4.611 seconds;
English from 3.901 to 3.701 seconds. Every sample outside the identified quiet
gaps is unchanged. Original greetings and manifest are archived, source/output
hashes and cuts are recorded, and future greeting generation applies the same
preparation. No TTS model, speech speed or live buffering change.

All 88 tests pass. Both greetings pass live Chromium playback, finish in
4.630/3.722 seconds, and return to listening without page errors. Asset hashes
and full-chain sample counts validate. App is running in screen with the same
public URL. See [pause cuts](speech-agent/tests/results/greeting-pauses-20261018.json),
[browser and sample evidence](speech-agent/tests/results/greeting-pauses-browser-20261018.json),
and [verification](speech-agent/VERIFICATION.md).

## Inference-file fade investigation and recovery adjustment (2026-10-18)

The owner still heard a gradual fade in the latest reply. Original latest-call
WAVs had already been removed by normal owned-job cleanup (metadata remained
in TTS status). Inspected the newest surviving original inference WAV and
regenerated the latest two sentences using the configured model/voice/settings,
capturing diagnostic streamed PCM. These are reproductions, not verified
bit-identical copies of the deleted last call. No call-retention policy changed.

Measured source fading plus insufficient recovery: immediate gain release
during brief quiet phonemes discarded boost before the following soft sound.
The rolling leveler now holds existing gain for 80 ms of inactivity before
release, raises its upward slew limit from 12 to 24 dB/second, and defaults to
9 dB maximum boost instead of 6. The 250 ms past-only window, deadband/partial
correction, silence reset, new-sentence neutral gain, onset smoothing and
−6 dBFS peak cap remain. This introduces no buffering or new lookahead.

All 84 tests pass. In fixed early/late windows of the reproduced final English
sentence, RMS drop is 10.119 dB raw, 4.053 dB with prior processing, and 2.332 dB
revised. Three diagnostic inputs preserve all samples and respect the peak
ceiling; median full processing is 0.240–0.255 ms per 20 ms frame. See
[analysis/provenance and paired recordings](speech-agent/tests/results/latest-inference-20261018/analysis.json)
and [verification](speech-agent/VERIFICATION.md). Deployed in screen
`speech-agent` (PID 1855503), healthy with zero automatic restarts. The public
URL is unchanged. [Live browser playback and hidden-cue checks](speech-agent/tests/results/inference-fades-ui-20261018.json)
pass, with no page errors. Naturalness and any extra breath prominence need listening.

## Rolling loudness window for gradual fades (2026-10-18)

The owner reported remaining gradual volume drops. Voice-agent playback now
replaces the fixed quiet curve with a 250 ms trailing RMS leveler, using only
past samples. Gain correction is partial, bounded (+6/−6 dB), smoothly slewed
and disabled for settled noise. Positive gain is suppressed at new sentence
attacks; sentence markers clear the level estimate and carried gain. The
native compressor/limiter and onset smoothing remain. This adds no buffering
or lookahead; the limiter retains its existing 1 ms. Defaults and fallback
are documented in [speech-agent/README.md](speech-agent/README.md).

All 82 tests pass. A controlled gradual-fade fixture improved from 8.129 dB
early-to-late drop to 3.043 dB. Generated Arabic/English clips with normal and
controlled-fade envelopes retain all samples and respect the −6 dBFS cap;
median full processing is 0.243–0.252 ms per 20 ms input frame. The first frame
immediately produces 839 samples. Browser speech playback and hidden-cue
checks pass. App is healthy in screen `speech-agent` (PID 1850390), with zero
automatic restarts and the same public URL. See [measurements and paired audio](speech-agent/tests/results/rolling-leveler-20261018.json),
[browser report](speech-agent/tests/results/rolling-leveler-ui-20261018.json),
and [verification](speech-agent/VERIFICATION.md). Naturalness still requires
listening; the bounded correction intentionally retains some dynamic range.

## Soft sentence starts after pauses (2026-10-18)

The owner narrowed the remaining problem to sudden peaks when the next
sentence starts. Existing gain state already persisted across sentences; the
new change adds a post-limiter 40 ms cosine soft attack, beginning at −12 dB
relative gain on the first audible sample. Sentence/acknowledgment markers
trigger it across network pauses with no PCM silence, while sample-index
alignment preserves the previous sentence's delayed tail. Long near-silent
gaps also rearm it; brief word gaps do not. Leading silence does not consume
the ramp. It adds no buffering or samples. The peak cap, quiet boost, native
filter state, hidden cues and immediate interruption remain intact.
`AUDIO_ONSET_MS=0` bypasses only this stage; its default is 40, with
`AUDIO_ONSET_ATTENUATION_DB=-12`.

All 77 tests pass, covering pause restarts, short word gaps, leading silence,
network/phrase markers, delayed previous-word tails and chunk independence.
Four actual independently synthesized Arabic/English sentence starts had
8.95–11.21 dB lower first-10-ms peaks, without changing sample counts. Full
processing measured median 0.115–0.122 ms per 20 ms input frame. Browser speech
and hidden-transcript checks pass; the app is healthy in screen `speech-agent`
and the public URL is unchanged. See [measurements and paired audio](speech-agent/tests/results/sentence-onsets-20261018.json),
[browser evidence](speech-agent/tests/results/sentence-onsets-ui-20261018.json),
and [verification](speech-agent/VERIFICATION.md). These measurements verify the
targeted onset reduction; preferred perceived smoothness remains a listening
judgment.

## Gentle lift for quiet streamed speech (2026-10-18)

The owner requested a small, natural boost to quieter audio. Added a bounded
FFmpeg compand stage before the existing compressor/limiter: up to 3 dB extra
gain, soft knees, 20/180 ms envelope smoothing, zero additional lookahead.
The curve returns to zero extra gain at the settled noise floor and normal/loud
levels. It is level-based, not a speech/breath classifier, and does not shift
pitch or alter sample timing. Per-reply state, tail draining, immediate
interruption, hidden LLM cues and the −6 dBFS peak ceiling remain intact.
`AUDIO_QUIET_BOOST_DB=0` bypasses just this new stage; default is 3.

All 72 tests pass, including quiet gain bounds, unchanged settled noise/silence,
smooth gain transitions, pitch sign crossings, loud-level recovery, peak safety
and existing sample/timing checks. Paired generated Arabic/English clips show
approximately 2–3 dB median lift in quieter audible windows, capped at 3 dB.
Total median CPU remained 0.059–0.065 ms per 20 ms frame; no new lookahead or
sample-count change. Real Chromium speech playback, hidden cues and error-free
transcripts pass. The app is healthy in screen `speech-agent`; the public URL
is unchanged. See [measurements and paired files](speech-agent/tests/results/quiet-leveling-20261018.json),
[browser evidence](speech-agent/tests/results/quiet-leveling-ui-20261018.json),
and [verification](speech-agent/VERIFICATION.md). Naturalness is a listening
judgment; the gentle cap preserves expressive volume variation rather than
forcing a constant level.

## Low-latency streaming volume control and fewer breaths (2026-10-18)

At the owner's request, the prompt no longer requests a breath after every
sentence: ordinary one-to-three-sentence replies have only a final breath, with
occasional internal breaths reserved for longer answers. Cues remain authored
by the LLM and hidden from the voice-agent UI.

Voice-agent playback now applies a per-reply native FFmpeg compressor and
limiter through the existing PyAV dependency. The pre-encoding sample peak
ceiling defaults to −6 dBFS, with 1 ms lookahead and no subprocess or full-clip
buffering. State persists across reply phrases and cached acknowledgments;
greetings use the same processing. Normal completion drains the delayed tail;
interruption discards it. Original model WAVs and model services are unchanged.
`.env` overrides: `AUDIO_DYNAMICS_ENABLED`, `AUDIO_PEAK_DBFS`.

All 68 backend tests pass. Synthetic spikes respect the ceiling; tests verify
reduced level swings, first-frame output, chunk-independent state, end samples,
bypass and interruption. Measured median/p95 CPU processing is 0.0597/0.0816 ms
per 20 ms input frame. Real Arabic/English LLM replies each used one final
breath. Both actual streamed replies and both cached greetings retained their
sample counts; stream first-frame processing measured 1.49/1.60 ms, separately
from the 1 ms waveform lookahead. Chromium playback and hidden-tag checks pass.
See [measurements and paired audio](speech-agent/tests/results/stream-dynamics-20261018.json),
[browser checks](speech-agent/tests/results/stream-dynamics-ui-20261018.json),
and [verification](speech-agent/VERIFICATION.md). The app is healthy in screen
session `speech-agent`; the public URL is unchanged. Digital peak control does
not set speaker volume or guarantee constant perceived loudness.

## LLM-authored sentence breaths and hidden cues (2026-10-18)

Historical breath placement, revised above to avoid a breath after each sentence.

The owner reported a clipped final letter and requested `[short breath]` after
sentences and at the end of replies, authored by the LLM. The runtime prompt
now requests these exact cues and lets the LLM choose friendly/calm/empathetic
delivery. The previous automatic `[friendly]` insertion is removed. The
sentence splitter waits for complete sentences and attaches trailing breaths
across token boundaries, without inserting cues or creating breath-only jobs.
Streaming transcription, assistant history and outgoing transcript events strip
complete or interrupted delivery cues before they reach the voice-agent UI.

All 56 backend tests pass. Actual Arabic and English LLM replies each contained
two authored breath cues, retained in two TTS phrases. Four real tagged/plain
TTS comparisons preserved every streamed PCM sample relative to completed WAVs;
the adapter's final partial-frame handling is also covered by regression tests.
Chromium received speech and showed a tag-free transcript with no page errors.
See [LLM/audio evidence](speech-agent/tests/results/llm-breaths-20261018.json),
[browser evidence](speech-agent/tests/results/llm-breaths-ui-20261018.json),
[screenshot](speech-agent/tests/results/llm-breaths-transcript-20261018.png), and
[verification report](speech-agent/VERIFICATION.md). The app is healthy in
screen session `speech-agent`; the public URL and model services are preserved.
This verifies cue delivery and sample completeness, not subjective completion
of the reported final letter. Human listening remains necessary. Cached greeting
and acknowledgment clips are unchanged.

## Voice agent mixed-language and address formatting (2026-10-18)

The runtime system prompt now preserves English terms in Latin letters inside
Arabic answers and Arabic terms in Arabic script inside English answers. URLs,
emails and technical separators are rendered as spoken words, e.g.
“Aljawab dot AI” and “contact at aljawab dot ai”, with address identity preserved.
The email example is explicitly not evidence of a company contact. The existing
verified-knowledge boundary and private TTS cue remain intact.

All 48 backend tests pass. Three real LLM checks using the assembled agent
instructions passed for both mixed-language directions and spoken domain/email
formatting. See [evidence](speech-agent/tests/results/spoken-format-20261018.json)
and [verification report](speech-agent/VERIFICATION.md). The updated prompt is
deployed through the existing voice-agent screen service.

## Voice agent private TTS delivery cue (2026-10-18)

Historical implementation, superseded by LLM-authored cues above.

At the owner's request, live voice-agent synthesis now prefixes each TTS phrase
with a single `[friendly]` cue only in the outbound local TTS request. Plain LLM
text, conversation context, transcript events and the visible voice-agent UI do
not receive that cue. The Noura reference transcript, exact model, cached clips,
sampling and phrase scheduling are preserved. This is a delivery experiment;
subjective naturalness still needs the owner's listening comparison.

All 48 backend tests pass. A real Chromium Arabic exchange observed two tagged
TTS requests and successful streaming playback, while both WebSocket transcript
events and the rendered transcript contained no delivery tags. No page errors
or error banner appeared. The app was restarted successfully in its existing
`speech-agent` screen service; the public URL is unchanged. See the
[validation](speech-agent/tests/results/delivery-tags-20261018.json),
[transcript screenshot](speech-agent/tests/results/delivery-tags-transcript-20261018.png),
and [verification report](speech-agent/VERIFICATION.md).

## Voice agent concurrent connections (2026-10-18)

The voice-to-voice application in `speech-agent/` now supports independent
simultaneous conversations, including multiple tabs sharing browser cookies.
The global single-call admission rule was removed. Each conversation expires
after 60 seconds without detected user speech; browser heartbeats do not extend
that input timer. Continuous speech updates it, and cleanup releases only the
expired call's room, agent and in-memory conversation. The existing per-IP
eight-starts-per-minute rate limit remains. New callers can join while the
existing exact-model TTS is generating; all calls share its bounded queue.

One-command startup is `bash /opt/dlami/nvme/speech-agent/start.sh`.
The application now runs inside detached screen session `speech-agent`,
supervised by the existing enabled `aljawab-app.service`. The running LiveKit,
HTTPS tunnel and SGLang TTS were preserved. The application health and readiness
checks pass, and its public URL remains
`https://achievements-affiliate-pope-pepper.trycloudflare.com`.

All 44 backend regression tests, TypeScript checking, frontend production build,
shell syntax and systemd unit validation pass. Three simultaneous real LiveKit
sessions were admitted; two completed generated-speech question/answer turns.
A silent session closed after about 61 seconds despite heartbeats, while later
speech extended only the second call. Two Chromium tabs sharing cookies also
connected successfully, then both released their microphone/audio and displayed
the idle message after 61.37 seconds, with no JavaScript errors. See the
[verification report](speech-agent/VERIFICATION.md), [launch guide](speech-agent/README.md),
[live connection evidence](speech-agent/tests/results/connections-20261018.json),
and [browser connection evidence](speech-agent/tests/results/browser-connections-20261018.json).

## Isolated laptop NF4 derivative (2026-10-17)

User authorized direct 4-bit conversion of the exact merged
`attention_mlp_r16_early/392cfb1cf6824476e616df3d0459986efdd268a13991737fdbb8260d200a5cd4`
export, with original services and environments preserved. Both parent shards,
unchanged codec and exact Aljawab Noura WAV match the supplied SHA-256 values.
Work is isolated in [attention_mlp_r16_nf4_20261017](training/attention_mlp_r16_nf4_20261017/).
Candidate Fish NF4 source pinned at `fc4e1e24ff3b8d7d28fdd66e6789f23acb63c5bb`.
Installed SGLang Fish layers omit quant_config forwarding; proceeding with the
explicitly authorized Fish Speech/bitsandbytes fallback and strict tensor coverage.
Completed direct packed NF4 conversion and strict coverage of all 358 learned
tensors, including all 200 merged fine-tuned attention/MLP weights. There are 201
NF4 linear layers. The packed file is 3,117,086,440 bytes; no parent shards or
duplicate codec are delivered. Fresh packed reload with requantization forbidden,
actual quantized GPU kernels, four matched BF16/NF4 Noura audio pairs, real
incremental PCM and WAV HTTP, and explicit gateway/app/cache identity checks pass.
The final ZIP was independently restored through the checksummed parts and
cold-loaded in a container with no BF16 shards mounted; synthesis passed again.

The [delivery ZIP](training/attention_mlp_r16_nf4_20261017/download/attention_mlp_r16_early-nf4.zip)
is 3,140,570,283 bytes, SHA-256
`f391c3e224b3b522e1ec35e80d58d34d6a659b975885438dc83147d4954a13a7`.
See the [launch/restore/integration guide](training/attention_mlp_r16_nf4_20261017/bundle/README.md),
[validation report](training/attention_mlp_r16_nf4_20261017/bundle/VALIDATION.md),
[paired listening artifact](training/attention_mlp_r16_nf4_20261017/bundle/listen.html),
and [final verification](training/attention_mlp_r16_nf4_20261017/download/FINAL_VERIFICATION.json).
Windows standard-library restore tooling and three optional 1 GiB-or-smaller
checksummed ZIP parts are in the download directory. Reuse and verify the
existing exact codec; retain the original BF16 transfer and manifest unchanged.

This is explicitly the Fish Speech/bitsandbytes fallback, **not SGLang**.
H100 four-case full-pipeline peak increment: 5.684 GiB whole-device, 3.665 GiB
host RSS. Warm RTF approximately 2.69–2.74 and first PCM approximately 3 seconds;
this memory-oriented eager fallback is slower than real time on H100. The final
restored-container cold/HTTP test peaked at 5.123 GiB incremental GPU and returned
device memory to baseline on exit. The laptop's reported 2 GB available host RAM
is below measured needs. SM89 laptop fit/latency and human listening quality are
UNVERIFIED. No equivalent laptop GPU was available. Production services, original
models, CUDA/PyTorch environments and credentials are untouched; no public upload.

This is the central record of completed work, current work, results, and next steps
across this speech workspace. Working conventions live in [AGENTS.md](AGENTS.md).
Detailed reports and live job status remain with their respective projects;
link them here rather than copying logs. The entries below cover the work verified
in the current session; other repositories have not received a fresh status audit.

## Current focus

The requested SGLang voice studio is **running and ready on port 3000** in
screen session `jordanian-sglang`. Both exact listening-selected checkpoints,
`fast_r32_final` and `attention_mlp_r16_early`, are resident on the H100. It
supports Talha, Wesam and single-speaker references, uploaded voice cloning,
WAV playback/download, original comparison at `/compare`, and paired
original/SGLang listening at `/sglang-compare`.

Actual host access and sudo are now verified. Docker and containerd storage
migration to NVMe completed with original stores retained; container installation,
codec execution, CUDA-13/BF16 preflight and both GPU engines passed. Live startup
exposed renamed CUDA-graph options, duplicated builder context arguments and
OpenAPI's omitted raw request body; the launcher/worker now handle the installed
release. All **54 regression tests**, **30 paired clips**, both uploaded-voice
checks and both studio HTTP generation requests passed. Full-model exports
still preserve every original tensor and slow/fast logits exactly on CPU.

Both engine logs show steady decoding around **98–100 tokens/second** with FA3
and CUDA graphs. Median paired-case RTF is 0.253 / 0.256; warm short requests
produce 2.6–3.1 seconds of audio in approximately 0.7–0.8 seconds. First requests
for a new reference take about 14–19 seconds for CPU reference encoding.
Historical original decoder throughput was about 12 tokens/second / RTF 1.84;
these are separate runs, not a concurrent controlled benchmark. Whole-device
usage is approximately 53.3 GiB including a separate resident Talha testing app.

See the [live validation report](training/jordanian_s2pro/reports/sglang_live_validation_20261011.json),
[HTTP checks](training/jordanian_s2pro/reports/sglang_http_validation_20261011.json),
[benchmark snapshot](training/jordanian_s2pro/reports/sglang_benchmark_snapshot_20261011.json),
[launch guide](training/jordanian_s2pro/SGLANG.md), and
[exact export audit](training/jordanian_s2pro/reports/sglang_export_validation.json).
The user reported quieter endings in recent uploaded-voice generations from
both models. Added reversible ending-volume recovery in the SGLang studio:
**Keep ending volume steady** defaults on for playback/download, with untouched
**Original WAV** access. Gain is limited to 9 dB over at most the final three
seconds, with smooth ramps, silence gating and peak protection. Earlier audio
and duration are preserved. Five affected clips measured roughly 2–5 dB
last-second RMS improvement with about 20–25 ms processing. All 59 tests and
five processed/original HTTP downloads pass; rendered JavaScript syntax also
passed in isolated QuickJS. This is an output correction, not a confirmed fix
to the model's underlying generation. The studio restart completed; both models are ready again, and fresh uploaded-voice
synthesis plus corrected WAV downloads passed for each model.
See the [ending-volume validation](training/jordanian_s2pro/reports/ending_volume_validation_20261011.json).

The user requested a separate streaming endpoint while preserving the normal
studio. Added `/stream` and `POST /api/stream`, with incremental PCM playback,
actual first-audio-token TTFT, queue wait and first-audio arrival. Both paths
share the existing bounded queue and model copies. All 70 tests and JavaScript
syntax pass; H100 configuration/hook preflight passed. Both models streamed
with warm TTFT about 110 ms and local first-audio arrival about 0.9 seconds,
well before full generation finished. Uploaded-voice streaming, normal generation
on both models and disconnect/save recovery passed. The final deployment is
ready with PCM payloads omitted from progress logs. Headless Chromium 153
verified playback before generation completed, playback completion, stopping,
and normal UI generation with ending-volume correction enabled. That browser
run displayed 92 ms TTFT and 945 ms first-audio arrival; new uploaded references
still incur approximately 13 seconds of first-use encoding overhead. See the [streaming validation](training/jordanian_s2pro/reports/sglang_stream_validation_20261011.json).

Listening review remains necessary for voice identity, accent and pronunciation;
real-browser microphone recording has not been exercised. The original
`start_voice_server.sh` remains available for returning to port 7860.

## Latest results

| Area | Latest verified result | Evidence |
| --- | --- | --- |
| Dataset exploration | 23 Wesam recordings; original subtitle clips can contain multiple speakers | [Exploration](training/wesam_exploration/README.md) |
| Voice identification | Enrolled Wesam reference; WeSpeaker embeddings with user-selected 0.5 cosine threshold and consistency checks | [Workflow](training/wesam_review/README.md) |
| Diarization | Community-1 completed for all 23 recordings on H100; saved timelines retained for reuse | [GPU status](training/wesam_dataset_gpu/status.json) |
| H100 benchmark | Same 300-second audio: CPU 252.97 s, GPU 4.15 s; 60.89× practical speedup; CPU batch 32, GPU batch 64 | [Benchmark](training/wesam_review/reports/gpu_benchmark.json) |
| Original SRT export | 4,978 clips; 12.38 hours; 280 Wesam clips | [Summary](training/wesam_dataset_gpu/summary.json) |
| VTT word-based export | 7,102 clips; 20.005 hours; 1,061 Wesam clips totaling 1 h 58 m 53 s; 25 episode-local guest IDs | [Dataset](training/wesam_dataset_words/README.md) |
| Validation | All 7,102 file/label/boundary checks passed; 40 source audio crops matched within PCM-16 tolerance; 43 regression tests passed | [Audit](training/wesam_dataset_words/audit.json) |
| Listening review | Filter by speaker, episode, transcript, and review decision; play audio and download decisions | [Review page](training/wesam_dataset_words/review.html) |
| Jordanian S2 Pro suite | Four sequential LoRA trials; bounded 10.246 h plan / 8.172 h train; recording splits, balanced sampling, GPU pilots/resume and matched base/early/best/final comparison implemented | [Run instructions and experiment plan](training/jordanian_s2pro/README.md) |
| Suite verification and fix | 14 regression tests passed; real zero-worker loaders, pilot packing and 44.1 kHz codec round-trip verified; corrected GPU run progressed to full training | [Data-loader fix and evidence](training/jordanian_s2pro/reports/dataloader_fix.json) |
| Attention gradient fix | 17 CPU tests passed; corrected H100 attention and attention+MLP pilots passed resume and updated all 160 / 400 adapter tensors respectively | [Live GPU evidence](training/jordanian_s2pro/reports/live_progress_20261010.json), [Checkpointing fix](training/jordanian_s2pro/reports/checkpointing_fix.json) |
| Full experiment results | All four trials completed 1,500 steps; all 13 comparison variants and 195 matched clips complete; user selected fast_r32_final and attention_mlp_r16_early | [Campaign status](training/jordanian_s2pro/suites/36aa33e6739594a4858d620c/status.json), [Listening selections](training/jordanian_s2pro/reports/voice_server_validation.json) |
| Selected-model voice server | Two exact adapters, three original speaker references, uploaded/recorded voice cloning, serial queue, WAV downloads, original comparison at /compare; 32 regression tests passed; host server ready with both models and successful live synthesis | [Launch guide](training/jordanian_s2pro/README.md), [Validation](training/jordanian_s2pro/reports/voice_server_validation.json) |
| Voice acceleration rollback | Original decoder restored after user reported fast server not working; both launcher filenames use the original version; 33 regression checks passed; persisted host status ready with successful eager synthesis | [Original launcher](training/jordanian_s2pro/start_voice_server.sh), [Rollback evidence](training/jordanian_s2pro/reports/voice_server_rollback_20261010.json) |
| SGLang review | Dedicated S2-Pro support verified from primary sources; H100 FA3 path and native checkpoint layout inspected; existing export format and sampling differences identified; no installation/runtime changes or GPU benchmark | [Assessment and experiment recommendation](training/jordanian_s2pro/reports/sglang_assessment_20261011.md) |
| SGLang live serving | Ready on port 3000 in screen; both exact models resident; 54 tests, 30 paired clips, both uploaded references and studio HTTP requests pass; FA3/CUDA graphs ~98–100 tokens/s, median RTF ~0.25; Docker/containerd on NVMe | [Guide](training/jordanian_s2pro/SGLANG.md), [Live validation](training/jordanian_s2pro/reports/sglang_live_validation_20261011.json) |
| Ending-volume recovery | Reversible studio playback/download correction for both models; five recent uploaded-voice clips show ~2–5 dB last-second lift; 59 tests and all HTTP downloads pass; original audio retained | [Validation](training/jordanian_s2pro/reports/ending_volume_validation_20261011.json), [Implementation](training/jordanian_s2pro/audio_finish.py) |
| Separate streaming studio | `/stream` + `POST /api/stream`; actual first-token telemetry and incremental PCM; 70 tests, H100 streams on both models, upload/normal/disconnect checks and Chromium playback controls pass; normal volume-corrected studio retained | [Validation](training/jordanian_s2pro/reports/sglang_stream_validation_20261011.json), [Browser screenshot](training/jordanian_s2pro/reports/sglang_stream_browser.png) |

## Current artifacts and tracking

- Refined dataset: `training/wesam_dataset_words/`; mono 48 kHz PCM-16 WAVs,
  matching `.lab` transcripts, model-neutral JSONL manifests, and relative word times.
- Recording-disjoint splits: training 5,790 clips / 16.574 h; validation 718 /
  1.684 h; test 594 / 1.747 h.
- Latest refinement status: [status.json](training/wesam_dataset_words/status.json)
  shows `complete`. GPU preparation also completed; there is no remaining
  preparation work in these two jobs.
- Jordanian suite: [configuration](training/jordanian_s2pro/experiments.json),
  [live status](training/jordanian_s2pro/monitor/status.json), and
  [launcher](training/jordanian_s2pro/run.py). Default command after activating
  Fish: `python training/jordanian_s2pro/run.py`; dashboard uses port 7860.
  Active campaign: `suites/36aa33e6739594a4858d620c`. All four GPU pilots passed.
  Full bounded preparation, all four trials and comparison synthesis completed.
  User listening selected two variants; merged-model export remains optional.
- Active SGLang studio: [launch script](training/jordanian_s2pro/start_voice_server_sglang.sh),
  [guide](training/jordanian_s2pro/SGLANG.md), detached screen `jordanian-sglang`,
  public port 3000, private engine APIs 3001/3002. Logs and WAVs remain under
  `training/jordanian_s2pro/voice/`; paired listening is at `/sglang-compare`.
  Separate live streaming and TTFT are at `/stream`, with SSE API `POST /api/stream`.
- Original voice server: [launch script](training/jordanian_s2pro/start_voice_server.sh),
  [backend](training/jordanian_s2pro/voice_server.py), and
  [interface](training/jordanian_s2pro/voice_ui.html). Start in screen with
  `bash /opt/dlami/nvme/training/jordanian_s2pro/start_voice_server.sh`. It replaces
  this suite's completed dashboard on port 7860; live state/logs and outputs live
  under `training/jordanian_s2pro/voice/`. Original comparison is retained at `/compare`.
- Reproducible commands and pipeline details:
  [DATASET.md](training/wesam_review/DATASET.md).
- Review UI: after restarting the review server, open
  [localhost:3000/dataset](http://localhost:3000/dataset).
  Start command: `bash /opt/dlami/nvme/training/wesam_review/start.sh`.

## Open issues and decisions

- VTT timestamps provide original caption word starts. Word ends are inferred
  from the next distinct start or caption end, so boundary and transcript
  accuracy still need listening review or forced alignment.
- Nine ambiguous or insufficient identity clusters were excluded. Episodes
  `0j515zfr5l` and `c5bcjps40r` have no accepted Wesam identity. Evidence is in
  [excluded_speakers.json](training/wesam_dataset_words/excluded_speakers.json).
- Guest IDs are local to each episode. Treating the same guest across episodes
  as one identity requires verification. Some trainers require independent
  recordings in each split for every speaker.
- Review decisions can be downloaded and applied to the Jordanian suite using
  `--review`; `--accepted-only` additionally requires explicit podcast accepts.
  The default suite uses automatic candidates and retains transcript/identity uncertainty.
- S2 Pro is selected for controlled adaptation experiments. Four LoRA strategies
  are implemented and completed. The user selected fast_r32_final and
  attention_mlp_r16_early for serving; additional custom-voice testing is pending.
- Host sudo, Docker, CUDA and network access are now available. Compiled SGLang
  performance and live uploaded-reference generation have passed; listening
  quality and real-browser recording remain unverified. S2 Pro is RL aligned;
  generated-audio quality must be assessed alongside loss. New references incur
  CPU encoding overhead before warm inference can reuse them.
- The pinned training embedding path omits a normalization used by generation
  when `scale_codebook_embeddings` is enabled. This is separate from the gradient
  failure and unchanged by this fix; audit numerical alignment before production selection.

## Next steps

1. Listen to Wesam and guest clips, checking voice identity, transcript accuracy,
   cross-talk, and clip beginnings/endings. Review the excluded identity clusters.
2. Optionally apply podcast review decisions through the new launcher's `--review`.
3. Continue serving on port 3000 with `start_voice_server_sglang.sh`; test
   uploaded-reference listening quality and browser microphone recording.
4. Optionally save further listening ratings at `/compare` and export a selected
   adapter using the documented `merge.py` command.
5. Review the running `/sglang-compare` clips and measured warm timings.
   Consider optimizing first-time CPU reference encoding if cold-reference
   latency matters. The original launcher remains available on 7860 with
   `--replace-voice-server` for switching back once jobs finish.

## Update history

- 2026-10-10: Central status record created. Documented completed H100
  diarization, benchmark, SRT export, VTT refinement, validation, and review UI.
- 2026-10-10: Implemented the bounded Jordanian S2 Pro experiment suite and
  comparison dashboard. CPU tests and real preview validation passed; GPU work
  remains pending user launch. See the suite's linked instructions and validation report.
- 2026-10-10: User launch exposed upstream's unconditional persistent-worker
  setting with zero workers. Added a suite-local data module supporting both
  train and validation loaders and preserving full-context pilot padding.
  Fourteen tests and real pilot loader checks passed; cached pilot tokens and
  packing remain reusable. GPU train/resume must be retried.
- 2026-10-10: Both fast-only H100 pilots passed save/resume and updated all 44
  adapter tensors. Reproduced attention-only backward failure with frozen
  embeddings; added worker-local non-reentrant checkpointing and missing-gradient
  guards. Seventeen CPU tests passed; corrected attention GPU pilots need retry.
- 2026-10-10 13:54 UTC: Verified all four corrected H100 pilots passed;
  fast_r16 and fast_r32 completed 1,500 steps each. Baseline and rank-16
  comparison samples are ready; rank-32 synthesis is active, followed by full
  attention trials. Recorded the evidence and a provisional 2.5–3.5 hour
  remaining-runtime estimate in the linked progress report.

- 2026-10-10: All four H100 trials and 195 matched comparison clips completed.
  User listening selected fast_r32_final and attention_mlp_r16_early. Implemented
  a dedicated two-model voice server with original references, custom uploads,
  optional recording and WAV downloads; 32 tests passed. Live launch requires
  the host because sandbox socket creation and CUDA are unavailable.

- 2026-10-10: Verified host voice server ready with both selected models and
  successful live synthesis. Recorded about 12 audio tokens/sec and RTF about
  1.84. Read-only audit found compilation disabled and token attention forced
  to math; documented compile/fused-attention benchmark priorities.

- 2026-10-10: Added a fast voice launcher using compiled single-token decoding
  with stable cache shapes, per-model warm benchmarks and automatic fallback.
  It preserves the selected checkpoints, references and existing voice data;
  All 42 regression tests passed, including CPU graph/cache/replay/restart checks. H100 speed measurement awaits
  the user's launch; see the linked fast-server report.

- 2026-10-10: User requested rollback after the fast server was reported not
  working. Restored original direct token generation and archived the acceleration
  implementation. Both launcher filenames now use the original version; 33
  regression checks passed. Persisted host evidence shows both selected models
  ready and a successful eager request. See the linked rollback report.

- 2026-10-11: Reviewed current SGLang-Omni S2-Pro support, H100 attention backend,
  dependency requirements, checkpoint layout and sampling differences. Saved an
  isolated experiment recommendation. Existing serving files/processes were not
  changed; accelerated CUDA results remain unmeasured.

- 2026-10-11: Implemented the authorized SGLang launcher on user-selected port
  3000. Exported both exact adapters in native HF layout and verified every
  state tensor plus slow/fast logits against the full Fish models on CPU.
  All 43 tests passed. Requested startup was attempted and stopped at sandbox
  Docker permission denial before changing the existing server. Host GPU
  loading, installation, timing and listening validation remain pending;
  see the linked SGLang implementation report and guide.

- 2026-10-11: Fixed the host-reported externally managed Python Docker build
  failure by installing and running SGLang in a container virtual environment.
  All 43 tests pass; local externally managed Python environment selection also
  passed. Container rebuild/GPU validation require the host. See the build fix report.

- 2026-10-11: Fixed the host-reported audiotools/protobuf dependency conflict
  using the upstream SGLang protobuf override explicitly. Added codec build
  checks and image-cache invalidation for override changes. All 45 tests pass,
  including a real offline resolver reproduction. Full image/GPU validation
  remains pending the host retry; see the protobuf fix report.

- 2026-10-11: Host dependency/version/codec build checks passed, but image
  extraction failed on the system disk. Added the one-command `--nvme-docker`
  migration with cache preservation, idle-workload guards and rollback. All 53
  tests pass. Actual migration is pending host execution because sandbox sudo
  is blocked; see the storage fix report.

- 2026-10-11: Host access and sudo enabled; verified completed Docker/containerd
  NVMe migration and cached image. Fixed installed-release graph arguments,
  builder context duplication and raw-request OpenAPI detection. Started the
  port-3000 SGLang studio in detached screen with both models resident. All 54
  tests, 30 matched clips, two uploaded-voice checks and two studio HTTP requests
  passed. Recorded ~98–100 decode tokens/s and ~0.25 median RTF; first-reference
  CPU encoding remains ~14–19s. See the linked live validation report.

- 2026-10-11: User reported fades in both models using their uploaded voice.
  Verified amplitude decline in the latest WAVs, and added bounded ending-volume
  recovery with an on/off checkbox and original downloads. Five affected clips
  measured ~2–5 dB last-second lift, with raw files unchanged. All 59 tests,
  processed/raw HTTP downloads and rendered JavaScript syntax passed. Live
  studio updated; both models reloaded successfully and fresh uploaded-voice
  synthesis plus corrected downloads passed on both.

- 2026-10-11: Added the requested separate streaming page/API while retaining
  the normal studio and ending-volume downloads. Measured actual first-token
  timestamps in the pinned S2-Pro runner; PCM plays before generation finishes.
  Both H100 models, uploaded cloning, legacy generation and disconnect/save
  recovery passed. All 70 tests pass. Fixed private-port reuse/fallback during
  restart, and filtered PCM payloads from progress logs. Chromium 153 verified
  playback/stop/completion and the original UI, showing 92 ms TTFT / 945 ms first
  audio on a warm request. Final server is ready in `jordanian-sglang` on 3000.
  See the linked streaming validation and browser screenshot.

## How to maintain this file

After each meaningful milestone, failure, or change of direction, update the date,
current focus, latest results, open issues, and next steps. Add a brief dated
history entry. Distinguish completed, running, pending, and blocked work, and
link to the evidence. Keep credentials and lengthy logs out of this file.
