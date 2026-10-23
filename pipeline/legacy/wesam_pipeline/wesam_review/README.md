# Wesam clip listening and speaker review

Local review of all 6,365 aligned clips across 23 interviews. Source audio,
transcripts, splits and the exploration inventory stay untouched.

From `/opt/dlami/nvme`, launch playback on port **3000**:

```bash
bash training/wesam_review/start.sh
```

Open `http://<server-ip>:3000`. If another app already uses this port, stop that
app in its terminal before launching this one. This launcher does not stop other apps.

To enable speaker predictions, install into a separate environment, then launch:

```bash
bash training/wesam_review/setup.sh
bash training/wesam_review/start.sh
```

Setup needs package access and installs pyannote.audio 4.x and its dependencies.
The existing Hugging Face login is reused without printing the token. The
Community-1 model and its dependent repositories must be accessible to that login.
Models and preview files are cached under this directory, not in training datasets.
CUDA is used when available; CPU analysis also works but is slower.

Select an episode, play clips, and click **Analyze speaker turns**. **Analyze 3
samples** queues the first, middle and last clip. The single analysis worker uses
30 seconds of original recording context before and after each selected clip.
Predictions are intersected with the existing clip boundaries, showing speaker
changes and overlap. Click a timeline segment or interval button to listen to it.
Analysis results persist under `cache/analysis/`; full-episode/corpus diarization
is not performed automatically. The app serves lazily generated 48 kHz WAV
previews to support browser playback and seeking.

Speakers are anonymous and local to each analysis window. They are not persistent
identities across windows or episodes. A predicted single speaker does not prove
clean speech or transcript accuracy. Short labels below 0.3 seconds remain visible
but do not count toward the displayed speaker count. Multiple-speaker predictions
flag clips for listening; music can cause false detections. Original overlap is
shown using regular diarization rather than exclusive diarization. No separation,
new transcript alignment or speaker-labelled training data is produced here.

[Community-1 documentation](https://huggingface.co/pyannote/speaker-diarization-community-1).
SepFormer is deferred until overlap examples are heard; separation alone does not
establish speaker identity or repair transcripts.

Analyze without a server after setup:

```bash
training/wesam_review/.venv/bin/python training/wesam_review/review.py \
  --analyze 2366wqj7sw-1 2366wqj7sw-3
```

Verification in this sandbox: all clip audio paths exist; six regression tests
pass for context intersection, speaker-count thresholds, overlapping duration,
HTTP episode discovery/path restrictions, and range playback of a real clip.
Model inference is unverified: pyannote 4 is not installed and package/model
network access is unavailable. Launching a server fails because socket creation
is denied. Run the launch/setup commands in the normal server shell.

## Manual Wesam enrollment and speech-window identification

The page now includes an enrollment panel. After rerunning `setup.sh` and
restarting `start.sh`, open port 3000 and:

1. Listen to a clip; use **Mark start now** and **Mark end now**, or type the
   offsets in seconds within the clip. **Play selection** previews the exact interval.
2. Save clean 2–10 second excerpts as **Wesam enrollment** until the counter
   reaches roughly 30 seconds. Prefer several 3–8 second excerpts across episodes.
   Use only uninterrupted Wesam speech, without guest speech, music or overlap.
3. Optionally save at least three additional **Wesam validation** excerpts and
   three **Other-speaker validation** excerpts. Include several guests and
   recording conditions. These must not overlap the enrollment audio. Examples
   from separate recordings give a more useful evaluation than adjacent excerpts.
4. Choose an embedding backend and click **Build reference embedding**. At least
   25 seconds of enrollment is required; more than 30 seconds is allowed.
5. Choose **Silero VAD + short windows** or **Pyannote turns + overlap flags**.
   Run the first five minutes before scaling to an entire episode. Click the
   prediction buttons beside each original clip to listen to the scored intervals.

`identity.py` implements the same workflow from the command line:

```bash
# Example selection only: listen and replace these times with verified Wesam speech.
training/wesam_review/.venv/bin/python training/wesam_review/identity.py \
  add 2366wqj7sw-1 2 7 --role enrollment
# Repeat add with different verified excerpts until the counter reaches ~30 s.
training/wesam_review/.venv/bin/python training/wesam_review/identity.py status
training/wesam_review/.venv/bin/python training/wesam_review/identity.py \
  --backend wespeaker build
# A bounded trial for one episode:
training/wesam_review/.venv/bin/python training/wesam_review/identity.py \
  --backend wespeaker identify --source 2366wqj7sw --vad silero --limit-seconds 300
# All 23 full original recordings:
training/wesam_review/.venv/bin/python training/wesam_review/identity.py \
  --backend wespeaker identify --vad silero
```

Use `--vad pyannote` to score diarized turns and leave detected overlap uncertain.
Backend options are `wespeaker`, `ecapa`, and `wavlm`; build a separate profile
for each backend before identification. The default is the pyannote wrapper
around WeSpeaker ResNet34, not the segmentation model. ECAPA-TDNN and WavLM
Base Plus SV are alternatives for comparison. No superiority on Jordanian
Arabic interviews is claimed; compare them on the same manually verified
validation excerpts and then assess separate held-out recordings.

Implementation details:

- Audio is loaded as mono 16 kHz with SoundFile/scipy, avoiding torchaudio file
  decoding. Model inference uses CUDA if available. Setup pins Torch/Torchaudio
  2.8.0 and TorchCodec 0.7.0 in a separate environment, and records installed
  dependency versions. Fish Speech/Qwen environments are not modified.
- Clean enrollment excerpts are embedded separately in approximately 3 second
  windows. Their unit vectors are averaged and normalized. A joined WAV is
  generated only for listening, not used as one artificial embedding input.
- Silero speech spans are scored using 3 second windows with a 1.5 second hop.
  These are overlapping evidence windows, **not exact speaker-turn boundaries**.
  VAD does not detect speaker changes or overlap; mixed windows can score as
  either speaker. Silero output explicitly leaves overlap status unknown.
- Pyannote mode scores windows within predicted turns and excludes windows
  intersecting a different predicted speaker. It still depends on the accuracy
  of the underlying diarization. Local cluster IDs have no cross-episode meaning.
- The configurable speech minimum now defaults to 0.5 seconds. Shorter speech is
  not embedded. Scored speech below 1.5 seconds is marked as a short estimate and
  remains uncertain under automatic calibration; the manual display threshold
  can still label it for listening review.
- Cosine similarity is not a probability. Without separate validation examples,
  the output contains scores and every identity label stays uncertain. With at
  least three manual validation excerpts per class, empirical 90th-percentile
  other-speaker and 10th-percentile Wesam score bands define the decisions.
  Overlapping/insufficiently separated bands are rejected. The middle band stays
  uncertain. These are initial thresholds, not guaranteed error rates; evaluate
  on additional clips not used to set them. Long validation excerpts are scored
  in the same short windows used for identification.
- “Other speaker” means a non-match to Wesam, not a verified guest identity.
- Selection intervals and audio hashes are stored in `identity/selections.json`.
  Reused source intervals are rejected even across enrollment/validation roles.
  Changes invalidate saved profiles and predictions shown in the page.
- `identity/wesam_reference.wav` is the compiled listening reference;
  `identity/profile_<backend>.json` stores the embedding, model name, source
  provenance and calibration scores. Predictions are stored under
  `identity/predictions/<episode>_<backend>_<vad>.json`, with absolute recording
  timestamps and flags for windows intersecting enrollment material.
- Full recordings are processed in bounded five-minute blocks with context.
  Analysis produces metadata only; no new speaker-labelled training clips or
  changed transcripts are written. Retriggering a source overwrites its prediction
  report. A five-minute trial replaces an earlier report for that same source/model/VAD.

Model references:
[WeSpeaker ResNet34](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM),
[ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb),
[WavLM Base Plus SV](https://huggingface.co/microsoft/wavlm-base-plus-sv),
[Silero VAD](https://github.com/snakers4/silero-vad).

Verification: 16 tests cover the original reviewer plus enrollment totals,
interval rejection, validation separation, stale profiles, reference assembly,
score bands, short/overlapped predictions and the reference HTTP endpoint.
JavaScript syntax and shell syntax pass. Real speaker-model/VAD inference and
live browser operation remain unverified in this network/socket-restricted
sandbox. No Wesam reference has been selected automatically.

The identification display now has an adjustable threshold, initially 0.5 based
on the user's listening observations. Scored windows above it display as Wesam,
below it as other speaker, and exactly equal/without scores as uncertain. This
is a display-only override; saved reports retain their original classification.
Refresh the page to use it with existing predictions. Missing scores now show
whether speech was below the report's configured minimum or speakers overlapped, including
older prediction reports whose existing flags supply the reason. New reports
also persist `no_score_reason`. In the inspected 146-window Silero trial, all 61
missing scores were due to short speech.


## Short speech and notebook overlap review

The 1.5 second cutoff was an application heuristic. Upstream pyannote probes the
embedding model's computational minimum (`min_num_samples`) separately; that is
not an accuracy guarantee. The cached WeSpeaker model was tested on one manually
selected enrollment excerpt at 0.25, 0.5, 0.75, 1, 1.5 and 3 seconds. All produced
finite embeddings. Scores were approximately 0.37, 0.56, 0.57, 0.63, 0.72 and 0.83.
This is a same-enrollment example, not independent validation or an error-rate
measurement. The probe is recorded in `reports/short_embedding_probe.json`.

Identification now accepts separate controls:

- `--min-speech-seconds 0.5`: score speech of at least half a second.
- `--window-seconds 3`: retain up to 3 seconds of voice evidence per embedding.
- `--hop-seconds 1.5`: advance overlapping windows by 1.5 seconds.

For finer sliding positions, use a 0.5 second hop while retaining the 3 second
window. Reducing the hop does not reduce the amount of audio used per embedding.
A shorter window is permitted, but the existing calibration bands were fitted
on longer windows and do not establish short-window accuracy. Runtime embedding
failures are reported per window without aborting all predictions.

```bash
training/wesam_review/.venv/bin/python training/wesam_review/identity.py   --backend wespeaker identify --source utrhqcji78 --vad silero   --min-speech-seconds 0.5 --window-seconds 3 --hop-seconds 1.5   --limit-seconds 300
```

The same fields are in the web reference panel. Restart the Python server and
refresh the page, then rerun identification to calculate the previously skipped
scores. Existing reports and embeddings are not regenerated automatically.

`speaker_embedings.ipynb` uses SepFormer-WHAMR16k to separate two streams and
ECAPA-TDNN to select/assign them using references. Its 3 second windows have 50%
overlap; this means overlapping time windows, not overlapping talkers. The
notebook has no 1.5 second minimum for its embedding function and sends shorter
inputs directly to ECAPA. Its low assignment-margin and high mixture-correlation
checks only warn; it still saves a stream. Correlation alone cannot prove failure,
since a genuine dominant speaker's stream can resemble the mixture. Picking the
best stream alone cannot prove it is Wesam if both streams score poorly.

For real simultaneous speakers, separation can be tested on detected overlap
intervals, followed by embedding comparison with an absolute match threshold and
a margin between streams; ambiguous outputs should be reviewed. Longer clean
turns for the same diarized speaker can also supply a cluster-level identity,
but that is inherited evidence, not a direct score for a short/overlapped segment.
The notebook is unchanged and its separator/ASR calls were not executed here.
The current reviewer still leaves detected overlaps unscored.

References:
[pyannote embedding minimum](https://github.com/pyannote/pyannote-audio/blob/main/src/pyannote/audio/pipelines/speaker_verification.py),
[overlap exclusion](https://github.com/pyannote/pyannote-audio/blob/main/src/pyannote/audio/pipelines/speaker_diarization.py),
[short-duration speaker verification](https://arxiv.org/abs/2406.02167),
[SepFormer-WHAMR16k](https://huggingface.co/speechbrain/sepformer-whamr16k).

All 19 tests pass, including configurable minimums, short-score estimates,
independent window/hop settings, and per-window embedding failure handling.
JavaScript syntax passes; live serving remains socket-restricted here.
