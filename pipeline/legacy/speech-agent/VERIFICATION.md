# Verification — updated 2026-10-18

## Friendly tone for the welcomes (2026-10-18)

At the owner's request, added `[friendly]` at the beginning of both cached
welcome synthesis requests, retaining the final `[short breath]`. The display
text remains plain, and Arabic still ends with "كيف أقدر أساعدك؟". Regenerated
both with the existing Noura voice, checkpoint and sampling settings. The
short-pause preparation remains; the other six cached assets are unchanged.

- Arabic duration **4.532 s**, English **4.040 s**. Both preserve the source's
  final 100 ms exactly and every playback sample through the dynamics chain.
  [Generation evidence](tests/results/welcome-friendly-20261018.json).
- **102 tests pass**, including migration from both untagged and breath-only
  valid caches to the friendly prefix, plain display text, repeat-run cache
  reuse, and unchanged unrelated clips. Python compilation passes.
- Original cached welcomes and manifest are preserved in
  `assets/voice/archive/welcome-before-friendly-20261018/`. App is healthy in
  screen `speech-agent` (PID 1896503), with zero automatic restarts.
- Both actual welcomes pass Chromium playback, finish and return to listening,
  hide both tags and show no page errors. Peak cap and complete sample counts
  validate. [Playback report](tests/results/welcome-friendly-playback-20261018.json).
  Recheck using `CHECK_ARTIFACT_PREFIX=welcome-friendly .venv/bin/python -m scripts.check_cached_welcomes`.

This verifies the requested synthesis instructions and audio integrity.
Whether the delivery sounds less robotic remains a listening judgment.

## Arabic welcome ending changed to كيف أقدر أساعدك (2026-10-18)

Updated the cached greeting text and runtime prompt greeting example to
"حيّاك الله، معك نورة، المساعدة الصوتية للجواب. كيف أقدر أساعدك؟".
Regenerated only Arabic, with one hidden final `[short breath]`. Duration is
4.636 s, with the existing short-pause preparation applied. The other seven
cached entries/assets remain unchanged. Hash and full playback sample count
validate; app is active in screen with zero automatic restarts. Previous Arabic
clip/manifest are archived in `assets/voice/archive/welcome-before-kaif-20261018/`.
[Exact synthesis request and validation](tests/results/welcome-kaif-20261018.json).

## Regenerated welcomes with a final breath (2026-10-18)

The owner reported a clipped welcome ending. Cached welcomes had been exempt
from the private breath suffix used on streamed replies. The cache generator
now appends exactly one `[short breath]` to greeting synthesis only, compares
cache settings against that tagged payload, and keeps `CLIPS`/manifest display
text untagged. This invalidates the two old greeting assets while leaving the
six acknowledgment/retry assets unchanged. Regenerated Arabic and English
with the same Noura reference, `attention_mlp_r16_early`, exact checkpoint and
sampling settings. No LLM prompt change.

- Arabic final asset: **4.781 s**. English: **3.437 s**. The previous long-pause
  preparation still applies: three Arabic quiet gaps retain 200 ms, removing
  420 ms total; the new English clip has no qualifying long gaps. The final
  **100 ms match the generated source exactly** in both files. Source WAVs,
  job IDs, exact synthesis text and measurements are in
  [the report](tests/results/welcome-breaths-20261018.json).
- **101 tests pass**, including a regression verifying that a valid untagged
  cache is replaced once, the suffix survives into synthesis settings, display
  text remains plain, and another clip's bytes/metadata remain unchanged.
- Previous cached clips and manifest are archived in
  `assets/voice/archive/welcome-before-final-breath-20261018/`. Python compilation
  passes; the app is healthy in screen `speech-agent` (PID 1893034), with zero
  automatic restarts and the same public URL.
- Both actual greetings pass Chromium playback, return to listening, and
  expose no breath tag or error banner. No page errors; readyState 4 and audio
  playback advance. Full playback processing preserves the updated sample
  counts and respects the −6 dBFS cap. Browser evidence is in the same report.

## Multiple users and mobile connection recovery (2026-10-18)

The owner reported that some mobile devices could load the page but Start
failed or remained connecting, on an unknown mix of Wi-Fi/cellular networks.
Public HTML/readiness returned 200 with `Cache-Control: no-store` and Cloudflare
`DYNAMIC`; the advertised media IP matched the instance. No global one-user
limit was present. Found two configuration gaps (eight admissions/minute per
shared public IP and no TURN relay) plus playback permission awaited after
network setup. These are plausible contributors; the affected phones' original
error was not captured, so no single root cause is claimed.

- Raised admission allowance to a configurable 60 starts/IP/minute. Existing
  room/session/token isolation and 60-second speech-idle behavior remain.
- Audio unlock begins in the Start gesture and cannot block call setup. A
  visible Enable audio button recovers denied autoplay without closing the
  call. Messages distinguish rate limiting, microphone trouble and media-network
  failure. Bounded, authenticated diagnostics contain only enumerated setup
  stage/reason and coarse platform. Fixed the observed disconnect-close traceback.
- Installed authenticated TURN/TLS at `turns:98.83.42.166:443?transport=tcp`.
  Production IP certificate is publicly trusted and expires 2026-10-25. HAProxy
  terminates TLS and sends PROXY v2 to native LiveKit TURN, trusted only from
  loopback. This preserves the client's mapped address. Certificates/accounts
  are root-only outside the repo; no TURN passwords/session credentials are
  recorded in reports. Direct media and the Cloudflare website URL remain.
- Renewal is checked twice daily by a persistent systemd timer, using a
  separate Certbot 5.8.0 environment. TCP 80 serves only the temporary standalone
  validation challenge. Certificate deployment gracefully reloads HAProxy
  instead of restarting LiveKit. One-command start/status includes these units.
  The config generator preserves TURN and rejects unhandled IP/certificate drift.
- **100 backend tests pass**, including nine same-IP concurrent admissions,
  retained per-IP limits, bounded/authenticated diagnostic logging, preserved
  relay configuration and certificate/IP drift rejection. TypeScript checking,
  Vite production build, Python compilation, shell syntax and systemd validation
  pass. Pre-existing systemd unit deprecation warnings were unrelated.
- Three isolated Chromium contexts (desktop, Android profile, iPhone profile)
  connect concurrently with received audio, both normally and with
  `iceTransportPolicy: relay`. **All three forced calls select relay/TLS**, not
  a direct candidate. A graceful proxy reload during those active calls leaves
  them connected. Simulated autoplay denial recovers with Enable audio;
  synthetic admission and media failures show correct messages and diagnostics.
  [Direct/default test](tests/results/mobile-connections-20261018.json),
  [forced relay test](tests/results/turn-relay-connections-20261018.json),
  [mobile recovery screenshot](tests/results/mobile-enable-audio-20261018.png).
- A full synthetic Arabic microphone → recognition → LLM → streaming speech
  exchange also passes with relay-only candidates: complete user/assistant
  events, advancing playback, two TTS requests with one final breath each,
  hidden cues and no page errors. [Voice exchange evidence](tests/results/turn-relay-speech-ui-20261018.json).

Production certificate issuance, simulated renewal with the deploy hook, and
post-renewal trusted TLS validation pass. The renewal timer is enabled and its
service exits successfully. App, LiveKit and proxy are active with zero
automatic restarts. [Network/certificate audit](tests/results/mobile-network-audit-20261018.json).

The iPhone/Android profiles above use **Chromium emulation from the server's
network**, not physical Safari, phones, or the affected carriers. Owner/device
retry remains necessary to confirm that the reported failures are resolved.
There is no claim of unlimited synthesis capacity: concurrent calls share TTS.

Implementation references: [LiveKit connectivity fallback](https://docs.livekit.io/intro/basics/connect/),
[mobile autoplay recovery](https://docs.livekit.io/reference/client-sdk-js/index.html),
[Cloudflare Tunnel protocols](https://developers.cloudflare.com/tunnel/concepts/routing/),
and [Let's Encrypt IP certificates](https://letsencrypt.org/2026/03/11/shorter-certs-certbot).

## Selective Arabic pronunciation diacritics (2026-10-18)

Added a runtime prompt rule to mark pronunciation-prone or ambiguous Arabic
words selectively, preserving conversational Saudi speech and avoiding full
vocalization or formal case endings. The word for written/transcribed text is
explicitly `النَّصّ` with fatha on ن, rather than `النُّصّ`; related forms are
included. Updated the prompt's earlier style example to use the same spelling.

A real `gpt-6-luna` check with the owner's unvocalized sample produced
"ويدعمون بعد التفريغ المباشر، مع تمييز المتحدثين وتحديث النَّصّ أول بأول."
The marks survive sentence splitting, manual final-breath insertion and TTS
payload preparation. No hardcoded word replacement was added. This verifies
prompt behavior and text delivery; the specific TTS pronunciation still needs
listening. [Recorded check](tests/results/selective-diacritics-20261018.json).
The app was restarted in screen with the same public URL.

## Application-added breath at every TTS split (2026-10-18)

Previously the prompt asked for a breath only at the end of an ordinary reply,
so intermediate sentence requests could be untagged. At the owner's request,
the application now adds `[short breath]` after each complete phrase is split,
immediately before calling `Speech.stream`. Model-authored/duplicate breaths
are stripped first and one canonical final cue is appended. Optional emotion
cues remain intact. This is on the audio branch only: the LLM output, history
and UI transcript do not receive the manually inserted cue. Removed all breath
placement instructions and examples from the runtime prompt; optional emotion
cue instructions remain. Cached greetings/acknowledgments are unaffected.

- **95 backend tests pass**. New cases cover idempotence, legacy/duplicate
  breaths, interrupted cues and cue-only text. An actual `tts_node` regression
  with three untagged English/Arabic sentences verifies three distinct stream
  requests, each ending with exactly one breath, and retained PCM frames.
- Python compilation passes. No additional buffering is introduced; only the
  complete sentence text sent for synthesis receives the suffix.
- Deployed in screen `speech-agent` (PID 1862044), healthy with zero automatic
  restarts. A real Chromium exchange produced **two separate TTS jobs**, both
  ending with exactly one `[short breath]`. Playback advances, complete
  user/assistant events arrive, cues are absent from visible and WebSocket
  transcripts, and there are no page errors or error banners.
  [Browser evidence](tests/results/manual-breaths-ui-20261018.json).

## Shorter pauses in the cached welcomes (2026-10-18)

The Arabic welcome contained near-silent internal gaps of 700, 510 and 630 ms;
the English welcome contained one of 400 ms. Shortened each to 200 ms, using
10 ms RMS analysis at −50 dBFS and a 300 ms minimum qualifying gap. Only the
middle of each long quiet region is removed, leaving quiet margins beside
speech and a 5 ms crossfade inside the quiet join. This is offline preparation,
with no live buffer, model change or speech speed adjustment. Existing playback
dynamics still apply normally.

- Arabic duration: **5.851 → 4.611 s**, removing **1.240 s** of quiet gaps.
  English: **3.901 → 3.701 s**, removing **0.200 s**.
  [Exact cut locations](tests/results/greeting-pauses-20261018.json).
- **88 backend tests pass**, including long-gap shortening, unchanged short
  gaps/quiet speech above the threshold, idempotence, retained final samples
  and format validation. Exact comparison confirms that every sample outside
  the identified quiet gaps in both actual clips is unchanged. Manifest hashes
  validate; playback dynamics retain every sample of each updated clip.
- Live Chromium checks pass in Arabic and English: speaking state lasts
  **4.630 / 3.722 s**, returns to listening, audio advances with readyState 4,
  and no page errors or error banner appear.
  [Browser and exact-sample evidence](tests/results/greeting-pauses-browser-20261018.json).
- Original WAVs and manifest are preserved in
  `assets/voice/archive/greeting-pauses-v1/`. Updated manifest includes source
  hashes and cut metadata. Future greeting generation applies this preparation.
  Migration command `.venv/bin/python -m scripts.tighten_greetings` is safe to
  repeat. The app was restarted in screen with the same public URL.

## Inference-file investigation and fade recovery (2026-10-18)

The latest call's original WAVs (jobs `c0f505f8b81f4c6687936b7f48f9ed34`
and `5c117231da2c41d099724d20494f5b9f`) returned 404: the app's existing
privacy cleanup had removed completed owned jobs, while TTS kept their metadata
in memory. The newest surviving original WAV is job
`08480260a5e04045ae3842635f3c7e7e`, from 17:46 UTC. Investigated that recording
and regenerated the last two sentences through the same configured voice,
model and sampling settings, capturing the streamed PCM before cleanup. These
are reproductions, not the deleted original call's exact waveforms; SGLang
seed support does not guarantee identical synthesis. Diagnostic copies and
their provenance are in [the analysis report](tests/results/latest-inference-20261018/analysis.json).

The source reproductions exhibit fading. The playback leveler's 6 dB gain
bound and immediate gain release during short quiet phoneme gaps left measurable
level loss. It now holds existing gain through the first 80 ms of inactivity,
then releases toward neutral; sustained silence still clears the analysis
state. Gain can rise at 24 dB/second instead of 12, with a default 9 dB cap.
The 250 ms past-only window, partial correction/deadband, new-sentence gain
reset, initial boost suppression, onset smoothing and −6 dBFS limiter remain.
This can also lift breaths at speech-like levels; no semantic breath/noise
classification is claimed.

- **84 backend tests pass**, including a new short-gap gain hold/long-silence
  release regression and bounded, smooth recovery with the default headroom.
- In the reproduced final sentence, comparing fixed 0.25–0.75 s and 2.5–3.0 s
  windows gives an early-to-late RMS difference of **10.119 dB raw**, **4.053 dB
  before**, and **2.332 dB revised**. These are recording-specific windows,
  not forced word alignments or a guarantee for other replies.
- All three captured inputs retain every sample. Revised peaks are
  −12.866 / −11.275 / −7.875 dBFS, below the −6 dBFS cap. First 20 ms input
  still produces 839 samples immediately. Median full processing is
  0.240–0.255 ms/frame; p95 0.292–0.400 ms. No additional lookahead.
- Replay the actual reproduction through [before processing](tests/results/latest-inference-20261018/regenerated-5c117231da2c41d099724d20494f5b9f-before.wav)
  and [revised processing](tests/results/latest-inference-20261018/regenerated-5c117231da2c41d099724d20494f5b9f-revised.wav).
  The [raw reproduction](tests/results/latest-inference-20261018/regenerated-5c117231da2c41d099724d20494f5b9f-raw.wav)
  is also retained for this requested investigation. Normal call cleanup is
  unchanged. Recheck with `.venv/bin/python -m scripts.check_inference_fades`.
- Deployed app is healthy in screen `speech-agent` (PID 1855503), with zero
  automatic restarts. Python compilation and the real Chromium speech check
  pass: advancing playback, complete user/assistant events, hidden cues, no
  page errors. [Browser evidence](tests/results/inference-fades-ui-20261018.json).
  Public URL and model services are unchanged.

## Rolling loudness control for gradual fades (2026-10-18)

Replaced the fixed quiet gain curve with a past-only 250 ms RMS leveler ahead
of the existing compressor/limiter. Default target is −24 dBFS RMS, with a
2 dB deadband, 80% correction outside it, and +6/−6 dB gain bounds. Gain is
interpolated per sample, with +12/−40 dB/second slew limits. A separate 20 ms
activity estimate prevents boosting settled material below −48 dBFS. Sentence
markers clear the estimate and gain; the first 120 ms of active speech cannot
receive positive gain. This avoids carrying an ending's boost into a new
sentence. No future samples are requested and the native filter graph remains
continuous. Floating-point boosted audio reaches the limiter without an
intermediate int16 conversion. The existing onset soft attack remains.

- Backend tests: **82 passed**, including sustained-fade correction, bounded
  smooth gain, no initial boost, silence/noise recovery, floating-point hot
  transients, chunk independence, sample tails, peak safety and cancellation.
- A controlled four-second, 12 dB input fade had an early-to-late output drop
  of **8.129 dB before / 3.043 dB after**. This is a synthetic envelope test,
  not a measurement that every real reply fades by that amount.
- Actual previously generated Arabic/English clips, both normal and with a
  controlled fade, preserve every sample and respect the −6 dBFS cap.
  First 20 ms input produces **839 samples immediately**. New total processing
  median is **0.243–0.252 ms**, p95 **0.288–0.307 ms** per 20 ms frame;
  the leveler adds no lookahead beyond the existing limiter's 1 ms.
  [Full report](tests/results/rolling-leveler-20261018.json).
- Listening pairs with a controlled fade:
  [Arabic before](tests/results/rolling-leveler-ar-gradual-fade-before-20261018.wav),
  [Arabic after](tests/results/rolling-leveler-ar-gradual-fade-after-20261018.wav),
  [English before](tests/results/rolling-leveler-en-gradual-fade-before-20261018.wav),
  [English after](tests/results/rolling-leveler-en-gradual-fade-after-20261018.wav).
  Normal unmodified-input comparisons are also linked in the report. Preferred
  naturalness and breath prominence require the owner's listening judgment.

`AUDIO_LEVELER_ENABLED=false` restores the earlier fixed quiet curve. Window,
target and maximum boost are configurable in `.env.example`. No model or cue
changes were made for this update.

Deployed in screen `speech-agent` (PID 1850390), healthy with zero automatic
restarts. The real Chromium speech exchange passes with advancing playback,
readyState 4, complete user/assistant events, no exposed tags and no page errors.
[Browser report](tests/results/rolling-leveler-ui-20261018.json).
Python compilation passes; the public URL is unchanged.

## Sentence restart soft attack (2026-10-18)

The owner reported that a sentence could start abruptly at high volume after a
pause. The existing filter graph already persisted across sentences. Added a
separate post-limiter onset smoother to address this specific transition:
−12 dB initial relative gain, smoothly reaching unity over 40 ms with a cosine
curve. The first sample remains audible rather than being faded from zero.
Leading quiet samples do not consume the ramp. A 180 ms near-silence interval
can rearm it; ordinary short word gaps do not.

`_tts_frames` emits private local segment markers before each sentence and
cached acknowledgment. The processor schedules each marker at the input sample
position, then applies it at the corresponding output position after limiter
delay compensation. This prevents the last 43 delayed samples of the preceding
word from being attenuated. Markers never enter the UI/audio transport. Native
compressor/limiter/quiet-lift state is not reset at sentence boundaries.

- `.venv/bin/python -m pytest tests -q`: **77 passed**. New cases verify
  restart smoothing after a 250 ms pause, no rearm after an 80 ms word gap,
  preservation of the previous delayed tail at a phrase marker, leading
  silence, sample/frame boundary independence, exact bypass and no loss of
  sample count. Existing peak, quiet-lift, cancellation and hidden-cue tests
  continue to pass.
- `.venv/bin/python -m scripts.check_sentence_onsets`: **passed**. Two Arabic
  and two English sentences were actually synthesized independently and
  processed with explicit boundary markers. First-10-ms peak changes were
  **−8.954 / −9.647 dB** in Arabic and **−9.219 / −11.213 dB** in English.
  Sample counts were retained and the −6 dBFS ceiling remained intact. Full
  processing CPU was median **0.1148–0.1224 ms**, p95 **0.1573–0.1606 ms** per
  20 ms input frame. No added buffering or algorithmic lookahead is introduced.
  [Measurements](tests/results/sentence-onsets-20261018.json).
- Paired listening examples: [Arabic before](tests/results/sentence-onsets-ar-before-20261018.wav),
  [Arabic after](tests/results/sentence-onsets-ar-after-20261018.wav),
  [English before](tests/results/sentence-onsets-en-before-20261018.wav),
  [English after](tests/results/sentence-onsets-en-after-20261018.wav).
  The files include a 250 ms listening-only gap between independent sentences;
  the DSP itself processes the actual arriving frames and local markers.
- Real Chromium exchange passes: complete user/assistant speech, advancing
  playback with readyState 4, no exposed tags, no page errors or error banner.
  [Browser evidence](tests/results/sentence-onsets-ui-20261018.json).
- The app is healthy in screen `speech-agent` (PID 1845461 at verification),
  with zero automatic restarts. Python compilation passes. Model services,
  cached source files and the public URL are preserved. `AUDIO_ONSET_MS=0`
  bypasses the new stage; `AUDIO_ONSET_ATTENUATION_DB` adjusts its starting gain.

This verifies onset attenuation and correct tail/sample handling. Perceived
smoothness and first-phoneme clarity still need the owner's listening check.

## Gentle quiet-level lift (2026-10-18)

Added a bounded FFmpeg `compand` stage before the existing compressor and peak
limiter. Its transfer points, at the default 3 dB cap, are
`-90/-90|-50/-50|-45/-45|-36/-33|-24/-21|-18/-18|0/0`, with a 3 dB soft knee,
20 ms attack envelope / 180 ms decay envelope, and `delay=0`. This gently lifts
quiet audible material while returning to unity gain at the settled noise
floor and normal/loud levels. It is not a semantic voice/noise detector: a
breath at a similar level can also receive some lift. The limited gain and
smooth envelope avoid aggressive normalization; pitch and sample timing are
not intentionally modified. `AUDIO_QUIET_BOOST_DB=0` retains the previous chain.

- `.venv/bin/python -m pytest tests -q`: **72 passed**. New cases verify an
  approximately 3 dB settled lift at quiet voiced levels, unchanged silence and
  settled −60 dBFS tone, preserved zero crossings, recovery to the old chain at
  normal/loud levels and smooth gain transitions. The measured largest
  per-sample gain step in the settled-level transition fixture was 0.0133 dB.
  All prior peak ceiling, sample-tail, chunk independence, latency, bypass and
  cancellation tests continue to pass.
- `.venv/bin/python -m scripts.check_quiet_leveling`: **passed**, using the
  earlier generated Arabic/English speech and a deliberately quieter version
  of each. Quiet audible windows gained a median **2.05–2.89 dB**, with maximum
  **3.00 dB**; sample counts were preserved and output peaks remained below
  −6 dBFS. Full chain median CPU was **0.0591–0.0652 ms** per 20 ms frame,
  p95 **0.0828–0.0941 ms**. This stage adds no lookahead; the limiter retains
  its existing 1 ms lookahead. [Measurements](tests/results/quiet-leveling-20261018.json).
- Paired listening examples at deliberately quiet input levels:
  [Arabic before](tests/results/quiet-leveling-ar-quiet-before-20261018.wav),
  [Arabic after](tests/results/quiet-leveling-ar-quiet-after-20261018.wav),
  [English before](tests/results/quiet-leveling-en-quiet-before-20261018.wav),
  [English after](tests/results/quiet-leveling-en-quiet-after-20261018.wav).
- `CHECK_ARTIFACT_PREFIX=quiet-leveling .venv/bin/python -m scripts.check_delivery_tags`:
  **passed** on the deployed app. Complete user/assistant speech, advancing
  playback, readyState 4, hidden delivery tags, no JavaScript errors or error
  banner. [Browser report](tests/results/quiet-leveling-ui-20261018.json),
  [screenshot](tests/results/quiet-leveling-ui-20261018.png).
- The app is running inside screen `speech-agent` (PID 1840454 at verification),
  with no automatic restarts. Python compilation passes. The existing model,
  LiveKit, tunnel and public URL are preserved.

These measurements show controlled level changes and low processing cost.
Perceived naturalness and the preferred boost amount remain listening judgments.

## Streaming compressor/limiter and occasional breaths (2026-10-18)

The owner requested fewer breaths and immediate volume-spike control on streamed
audio. The runtime prompt now requires just one final breath for ordinary short
answers; longer replies may include occasional natural internal breaths.

Added `app/audio.py`: native FFmpeg `acompressor` (−18 dBFS threshold, 3:1 ratio,
1 ms attack, 100 ms release, 2 dB makeup) followed by `alimiter` (−6 dBFS sample
ceiling, 1 ms lookahead, 60 ms release, automatic normalization disabled, delay
compensation enabled). PyAV's existing native filter graph avoids a subprocess
and processes each arriving frame. A per-reply graph persists through phrases
and acknowledgments; cached greetings use the same chain. Normal EOS drains the
remaining samples; interruption closes the source without playing stale tails.
The model's original saved WAVs remain unmodified. These are pre-encoding sample
levels, not a physical speaker loudness limit or a post-codec true-peak guarantee.

- `.venv/bin/python -m pytest tests -q`: **68 passed**. Covers ±full-scale
  spikes, compression of a quiet-to-loud transition, exact sample count/final
  sound down to one-sample inputs, identical output across different chunking,
  first-frame output, exact bypass, cancellation and untagged intermediate
  sentences. Existing queue/session/cue tests continue to pass.
- `.venv/bin/python -m scripts.check_audio_dynamics`: **passed**. Over 1,000
  20 ms frames: median **0.0597 ms**, p95 **0.0816 ms** CPU time; first-frame
  CPU **0.2835 ms** in the warmed harness. The fixed waveform lookahead is about
  **1 ms**. Actual stream first-frame processing, including graph setup, measured
  **1.4926 / 1.5961 ms** for Arabic/English. No additional complete frame was
  needed before first output. Both full streams and both greetings retained
  every input sample position through EOS. Arabic/English generated replies
  each contained only one final breath. Actual output peaks were below the
  configured ceiling; synthetic full-scale spike tests exercise the cap itself.
  See [measurements](tests/results/stream-dynamics-20261018.json).
- Paired generated audio: [Arabic original](tests/results/stream-dynamics-ar-raw-20261018.wav),
  [Arabic processed](tests/results/stream-dynamics-ar-limited-20261018.wav),
  [English original](tests/results/stream-dynamics-en-raw-20261018.wav),
  [English processed](tests/results/stream-dynamics-en-limited-20261018.wav).
- Real Chromium exchange: audio plays with advancing time/readyState 4, a
  complete user/assistant turn, no exposed cue tags, no page errors or error
  banner. [Browser report](tests/results/stream-dynamics-ui-20261018.json),
  [screenshot](tests/results/stream-dynamics-ui-20261018.png).
- The app is running in screen session `speech-agent` (PID 1836232 at validation)
  with zero automatic restarts. Original LiveKit/tunnel services and the public
  URL are preserved. Python compilation passes. `AUDIO_PEAK_DBFS` and
  `AUDIO_DYNAMICS_ENABLED` permit server-only adjustment/bypass after app restart.

Measured processing cost is small; this does not remove TTS/network latency or
guarantee constant perceptual loudness. Listening remains necessary to choose
the preferred ceiling and compression strength.

## LLM-authored short breaths and hidden delivery cues (2026-10-18)

Historical every-sentence breath placement, superseded by the prompt above.

The owner reported a cut-off final letter and requested `[short breath]` after
sentences and at reply endings, chosen/generated by the LLM. The runtime prompt
now explicitly requests those cues, plus optional context-appropriate
friendly/calm/empathetic delivery. Removed automatic cue insertion from the TTS
payload. Cached greeting/acknowledgment clips remain unchanged.

Added a streaming transcription filter that handles cues split across tokens,
and sanitization for assistant history and final/outgoing transcript events.
The TTS phrase splitter preserves a trailing breath with its preceding sentence,
waits for complete sentences, skips cue-only requests, and puts punctuation
before a trailing cue. It does not invent missing cues.

- `.venv/bin/python -m pytest tests -q`: **56 passed**. New cases cover every
  two-chunk tag split, partial tags at interruption, hidden cues, delayed breath
  tokens, complete long sentences, cue-only input, cue placement and exact
  retention of the final partial PCM frame.
- `.venv/bin/python -m scripts.check_llm_breaths`: **passed**. Actual Arabic
  and English LLM responses each generated two `[short breath]` cues. Chunked
  phrase processing preserved both cues in two sentences, with no manual
  insertion. Real tagged and plain TTS requests used those generated answers;
  all four SSE streams were byte-for-byte equal to the PCM in their completed
  WAVs, including the ending. See [report](tests/results/llm-breaths-20261018.json).
  Listening files: [Arabic with cues](tests/results/llm-breaths-ar-tagged-20261018.wav),
  [Arabic plain](tests/results/llm-breaths-ar-plain-20261018.wav),
  [English with cues](tests/results/llm-breaths-en-tagged-20261018.wav),
  [English plain](tests/results/llm-breaths-en-plain-20261018.wav).
- `.venv/bin/python -m scripts.check_delivery_tags`: **passed** against the
  deployed app. Browser playback advanced with readyState 4. All observed
  tagged TTS requests ended with `[short breath]`, and both transcript events
  and the rendered transcript were tag-free. No JavaScript errors or error
  banner appeared. See [browser report](tests/results/llm-breaths-ui-20261018.json)
  and [screenshot](tests/results/llm-breaths-transcript-20261018.png).
- The app is healthy and ready in screen session `speech-agent` (screen PID
  1831800 at validation), with zero automatic restarts. Model services and the
  public URL are unchanged. Python compilation passes.

Sample equality rules out PCM loss in the tested SSE-to-WAV path. It does not
prove that the model's final phoneme sounds complete; that remains a listening
judgment for the owner, using the live app or paired generated samples.

## Mixed-language text and spoken addresses (2026-10-18)

Updated the runtime prompt in `02-VOICE-AGENT-PROMPT.md` to preserve embedded
English/Arabic words in their original scripts, and output spoken forms for
URLs, email, paths, filenames and technical separators. Reconciled the prior
punctuation/source-link instructions with spoken-only output. The example
`contact@aljawab.ai` does not establish a company contact; the existing verified
knowledge boundary remains explicit.

All **48 tests pass**. Three actual `gpt-6-luna` Responses calls using the
runtime agent's assembled instructions passed: English names inside Arabic,
Arabic names inside English, and verbalized domain/email with no raw address.
Observed address output: “Aljawab dot AI. Contact at Aljawab dot AI.” See
[generated prompt checks](tests/results/spoken-format-20261018.json).
This verifies generated text formatting; it is not a new pronunciation or
acoustic listening assessment.

## Private live-TTS delivery cue (2026-10-18)

Historical implementation, replaced by LLM-authored cues above.

Live `Speech.stream()` adds `[friendly]` only when constructing the local TTS
payload. Conversation and transcript text stay plain. Reference transcription
and cached clips are unchanged. The payload helper permits controlled trials
with `friendly`, `calm` and `empathetic`; the deployed live default uses only
`friendly`, without stacking cues or adding pauses.

- `.venv/bin/python -m pytest tests -q`: **48 passed**, including new Arabic
  and English number-normalization/payload checks, exact reference preservation,
  invalid-cue rejection and streaming-request versus transcript separation.
- `.venv/bin/python -m scripts.check_delivery_tags`: **passed** against the
  running app and actual SGLang engine. A generated Arabic browser microphone
  question received a complete answer; two local TTS jobs carried `[friendly]`.
  Both transcript events and visible transcript were free of delivery tags.
  Audio was playing with readyState 4 and advancing playback time. No JavaScript
  errors or error banner appeared. See [result JSON](tests/results/delivery-tags-20261018.json)
  and [browser screenshot](tests/results/delivery-tags-transcript-20261018.png).
- The app is running under the same screen/systemd arrangement (screen PID
  1809216 at validation), with zero automatic restarts. TTS reports ready and
  retains both selected models. The HTTPS URL is unchanged.

These checks verify tag delivery and UI separation. Perceived expressiveness,
voice identity and naturalness remain listening judgments; this change does
not alter the phrase splitter or synthesis scheduling.

## Concurrent connections, idle expiry and screen deployment (2026-10-18)

The owner's revised requirements replace the original single-conversation rule.
The application is running in detached screen session `speech-agent`, supervised
by the enabled `aljawab-app.service` (screen PID 1803286 at final verification).
`bash /opt/dlami/nvme/speech-agent/start.sh` starts the dedicated services, waits
for health and prints the current HTTPS URL. It is safe to repeat. Attach with
`screen -r speech-agent`; detach with Ctrl-A then D. The original service unit
is retained in `deployment/aljawab-app.before-screen-20261018.service`.

- `.venv/bin/python -m pytest tests -q`: **44 passed**. Covers simultaneous
  admission, per-tab authentication, isolated language/events/end behavior,
  heartbeat versus input expiry, per-call speech activity, failed-start
  isolation, idempotent cleanup, WebSocket origin/token checks and admitting a
  caller while the exact required TTS model is generating.
- `npm --prefix web run check` and `npm --prefix web run build`: pass using the
  bundled Node 22. Shell syntax, Python compilation and systemd unit validation
  pass. Vite retains its existing large-bundle advisory.
- [Real LiveKit checks](tests/results/connections-20261018.json): **three**
  simultaneous sessions admitted. Two received generated microphone speech,
  final user/assistant turns and audio. The first expired after its initial
  speech; a second utterance extended only the second session, which remained
  usable. A third silent session expired after **60.56 seconds despite
  heartbeats**. No pipeline error events were observed. The two spoken calls
  completed one and two assistant turns respectively.
- [Chromium checks](tests/results/browser-connections-20261018.json): two tabs
  in one cookie-sharing browser context connected independently; the admission
  check was also exercised while TTS was generating. Both silent tabs closed
  after **61.37 seconds**, released audio/microphone resources, showed the idle
  message and returned to Start with no page errors or error banner. Screenshots:
  [tab 1](tests/results/connections-idle-tab-1.png),
  [tab 2](tests/results/connections-idle-tab-2.png).
- Final `/health` and `/api/readiness` return HTTP 200 / true. The application
  service remained active with **zero automatic restarts** during checks. The
  existing LiveKit and HTTPS tunnel processes remained running, and TTS still
  reports both selected models loaded. The public URL is unchanged.

These checks use generated speech and silent browser microphone fixtures.
They verify concurrent operation and cleanup, rather than maximum-load capacity
or a remote human microphone. All calls still share the existing serial TTS
queue. The existing per-IP eight-starts-per-minute rate limit remains; the
one-active-conversation limit is removed. Older evidence below records the
previous deployment and original acceptance criteria.

## Noura / Saudi / spoken-numbers update

The current voice is **Noura (نورة)**, registered from the owner's `saudi_female_1minute-trimmed2.wav` with the exact supplied transcript. The file is 9.46 seconds, stereo, 44.1 kHz; the existing upload API prepares its mono reference. `assets/voice/reference.json` records source/prepared hashes and reference ID; `.env` binds synthesis to that reference, transcript, and prepared-audio hash. All eight clips were regenerated with the same required `attention_mlp_r16_early` model. The old clips remain archived; the model services were not restarted.

The runtime prompt now specifies conversational Saudi Arabic, Noura's name, feminine self-reference, and numbers written in words. A streaming number normalizer also handles Western/Arabic-Indic digits, decimals, percentages, grouped thousands, negatives, and digits split across LLM deltas before display and synthesis. Thirty automated tests pass, including the new number cases. Current update evidence is in `tests/results/noura-update.json`, `noura-vad.json`, and `logs/noura-browser.log`.

Silero was already installed/in use. It now uses activation threshold 0.7, minimum speech 220 ms, and end silence 550 ms, replacing 0.6 / 120 ms / 450 ms. Synthetic silence/noise and the supplied speech reference are tested separately; room noise, perceived voice similarity/accent, and the preferred real-microphone interruption sensitivity still require listening in the target browser. Historical measurements below describe the earlier voice/settings, not a new acoustic benchmark for Noura.

Test URL: **https://achievements-affiliate-pope-pepper.trycloudflare.com**. Local application: **http://[::1]:8002**. These results distinguish synthetic media and server checks from human microphone/remote-network checks.

## Completed checks

| Check | Result and evidence |
|---|---|
| Dedicated deployment | App, self-hosted LiveKit Server, and Cloudflare tunnel are active under their three `aljawab-*` systemd units. HTTPS returns 200, the page renders in Chromium, and the application responds on IPv6 loopback port 8002. |
| Existing services preserved | Original model/application PIDs and start dates are unchanged: IPv4-8002 app 65220/65268, port-3000 facade 463421, worker 463543, engines 463949/465108. `/api/status` still reports both original models ready. No CUDA/model environment, model configuration, or original service restart was performed. |
| Required model routing | Non-streaming job metadata and every accepted live job identify `attention_mlp_r16_early` and checkpoint `9dbf9c8b6ee71dc5c737e6a76f2166539e833a6aa8ab274fb192553714a294cf`. Code routing leads to the existing engine on 3002, whose `/v1/models` lists the corresponding export. Configuration rejects the other model. |
| Actual cached audio | Eight clips generated through port 3000 `/api/generate`. All WAV headers, durations, SHA-256 hashes, and nonzero RMS checked. Durations: Arabic greeting 5.526 s, English greeting 3.901 s; acknowledgments 0.650–1.161 s; retry clips 3.576/3.715 s. Manifest records exact text/settings/reference/checkpoint. Human listening is pending. |
| Incremental live TTS | Actual `/api/stream` request returned first audio at 0.855 s and completed at 1.044 s for a 1.689 s Arabic test clip, yielding 169 PCM frames. First audio was received before the response completed. Live responses use this streaming endpoint, never cached-clip generation. |
| Provider account access | Actual `gpt-6-luna` Responses request with reasoning `none` succeeded; actual `gpt-6-luna` Decisions request succeeded; `gpt-live-transcribe` accepted the documented 24 kHz, bilingual, no-server-VAD configuration. The same owner-provided key serves all three. |
| Rendered website grounding | Two traceable page snapshots, approximately 10,330 characters of business/technical/FAQ content, with headings, timestamps, hashes, languages, and published contact links. Generated Arabic/English answers support products and voiceprints; unsupported price/guarantee/integration claims are declined. Booking/ticket requests do not result in claims of action. User instruction-injection test was declined. |
| Knowledge refresh failure | Intentionally blocked retrieval raised an extraction failure and preserved the SHA-256 of the last good snapshot. No placeholder knowledge was written. |
| Browser two-way media | Chromium on this instance opened the public HTTPS URL, sent a generated English question through its microphone input, received a final transcription, and played the streamed response. Inbound audio had approximately 260 kB received and total audio energy 0.827; playback was unpaused with readyState 4 and an advancing currentTime. This is a synthetic browser exchange, not a remote human call. |
| Arabic browser exchange | A generated Arabic correction was transcribed and answered in Arabic. Received media had nonzero energy; bilingual RTL/LTR screens were inspected. |
| Interruptions over actual media | A native LiveKit test participant injected generated Arabic speech during the cached greeting, then a correction during the first answer. Both canceled. The first assistant answer is marked interrupted; the next answer addresses voiceprints and does not resume the canceled answer. Receiver PCM-energy traces and events are saved. |
| Language switch over actual media | After Arabic turns, generated “Can you explain that in English?” was transcribed; the next complete spoken answer was English and grounded in the same voiceprint source content. |
| Busy and End | Concurrent admission returns 409 and the UI has a friendly busy message. End stops local capture, disconnects the room, clears the transcript, releases capacity, and exposes Start again. The mute control changes to the correct unmute label. Physical microphone muting still needs a human check. |
| Abandoned admission | A real admission without heartbeats stayed busy through the 60-second polling sample; a new admission succeeded at the 70-second sample. This brackets cleanup between 60 and 70 seconds, consistent with the 60-second lease plus maintenance/cleanup. The replacement room was ended. Evidence: `tests/results/lease.json`. |
| Errors and concurrency | 17 automated tests pass: punctuation/decimals/URLs/abbreviations, bounded phrase flush, model pinning, source/assets, missing credentials, recoverable TTS failure, slow Decisions fail-open, answer generation independent of a stalled selector, suppression when substantive audio is ready, and rejection of late TTS frames after interruption. Frontend TypeScript check/build and Python compilation pass. |
| Secrets and persistence | Provider/server secrets are server-side in mode-600 files; checked client assets contain none. Participant tokens are scoped to a random room with five-minute expiry. User microphone audio and user transcript logging/recording/browser persistence are disabled. Inherited studio retention is a limitation below. |

Evidence is in `tests/results/english-call.json`, `arabic-call.json`, `interruptions-passed.json`, `media-interruptions.json`, `grounding.json`, `grounding-language-fixed.json`, `decisions.json`, and screenshots. All saved conversation evidence uses generated fixtures, not visitor audio. Native Python test-client teardown emitted non-fatal FFI deallocator warnings after saving completed media evidence; the running application/model services remained healthy.

## Measurements

These are small engineering samples, not a latency guarantee or a statistically robust benchmark. Speech-end timestamps are estimated from local VAD end minus the configured 450 ms silence. Audio-ready timestamps are measured when the server yields its first substantive frame, not when a remote speaker hears it.

- Six completed instrumented starts: five Arabic turns and one English switch, with Decisions enabled and no acknowledgment selected. Speech end to substantive server audio: **2.791, 3.223, 2.983, 3.087, 2.974, 9.423 seconds**. Median **3.04 s**; nearest-rank p95 **9.42 s** for this tiny sample. The 1.5-second aspirational target is **not met**. Other early English runs also showed variable response-generation latency.
- Controlled interruption examples: server playout cancellation completed in **2.3 and 3.6 ms after local speech detection**. For the answer interruption, the local receiver became sustainably quiet approximately **243 ms after injected speech began**, approximately **40 ms after detection**. The greeting test overlapped a natural pause, so its silence timing cannot isolate cancellation latency. These are local receiver results, not browser/acoustic/remote-network timings.
- Four standalone Decisions samples: **717.3 ms cold**, then **96.8, 364.8, 81.7 ms**. All selected `none`. Results exceeding the application's 250 ms deadline are discarded. The concurrency test stalls classification for ten seconds and still yields the first substantive answer in under 50 ms with the deterministic fake answer source. No retries or waits are on the main answer path.
- A controlled full-call comparison with forced acknowledgment playback and listening-based filler-quality scoring has **not** been completed. No claim is made that Decisions improves answer latency. Local acknowledgment-without-classification playback is intentionally silent; classifier output must pass confidence, language, generation, cooldown, and audio-ready checks.

## Remaining human/remote checks and limitations

- **Human microphone/browser:** allow the microphone on another device, listen for voice quality, pronunciation, pauses, volume continuity, and decoder artifacts; confirm actual mute and End behavior. Test quiet hesitation, mixed Arabic-English terms, fan noise, speaker echo, and interruptions during each cached clip. Browser echo cancellation/noise suppression are enabled; the SDK's initial three-second AEC interruption blackout was explicitly disabled after it failed the greeting test.
- **Remote media networking:** browser tests above run on the EC2 instance and can use advertised internal media candidates. They do not prove AWS inbound media reachability. Open **UDP 7882** and **TCP 7881** in `Majed-dev` for the test clients if needed. The instance IAM role cannot inspect/change those rules. No additional inbound web port is needed for HTTPS.
- **TCP/second network:** TCP 7881 is configured/listening, but attempts to force the automated browser onto TCP still negotiated UDP. TCP media fallback and second-network behavior therefore remain **unverified**. No TURN/TLS relay is configured for restrictive networks.
- **Reconnect/tab close:** browser-independent abandoned-admission cleanup passed, as recorded in `tests/results/lease.json`. Deliberate tab close, transient remote reconnect, and repeated recovery on another device still need a browser check. Repeated `/api/ready` is guarded against duplicate greetings.
- **Inherited studio retention:** Aljawab deletes only its registered generated-result files after completed/failed jobs. The preserved TTS studio retains recent synthesized-text job metadata in memory and `voice/status.json`; interrupted synthesis can finish after app stop and leave its audio until next cleanup. This is not strict end-to-end zero retention. Provider retention also depends on account data controls. Use non-sensitive test speech.
- **TTS cancellation:** the model service cannot abort a single synthesis. Playback/generation cancellation and stale-output rejection are verified; GPU work may continue briefly for a canceled job.
- **Temporary address:** Cloudflare Quick Tunnel URLs change when the tunnel restarts. `scripts/ctl.sh status` prints the current address. IPv4 port 8002 continues to serve the unrelated existing app; Aljawab's local origin is `[::1]:8002`.

## Acknowledgment frequency update

The live selector budget is now 800 ms, cooldown six seconds, earliest playback 350 ms. Eligibility uses combined acknowledgment probability of at least 0.8 rather than requiring high confidence in one of two synonymous clips. A selection arriving after initial eligibility remains eligible while substantive audio is pending. Tests verify immediate substantive audio still wins, and a delayed classifier does not hold generation/synthesis. Thirty-two automated tests pass. A live `gpt-6-luna` selection of `ack_ar_understood` produced the actual cached Noura clip through the shared TTS frame path at about 353 ms in the server-side test; this does not claim a new remote-browser acoustic measurement. Evidence: `tests/results/acknowledgment-tuning.json`.
