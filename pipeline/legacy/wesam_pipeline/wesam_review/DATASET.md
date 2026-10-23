# Preparing Wesam podcast data

## Completed word-timestamp refinement

The refined export is in `training/wesam_dataset_words/`: 7,102 clips, 20.005 hours,
including 1,061 Wesam clips (7,133.157 seconds) and 25 anonymous episode-local guest
IDs. It reuses all 23 saved GPU diarization results and their identity evidence;
no model inference or new speaker identity assignment is performed.

```bash
training/wesam_review/.venv/bin/python training/wesam_review/refine_dataset.py
training/wesam_review/.venv/bin/python training/wesam_review/audit_refinement.py
```

VTT rolling-caption history is removed, and words with identical timestamps stay
together. Word starts come from the original captions; ends are inferred from
the next distinct onset or caption end. These are not verified forced-alignment
word ends. Words whose intervals cross a speaker change or detected overlap are
excluded. Accepted neighboring words form contiguous 3–30-second clips with at
least 3 seconds of detected speech, preserving episode splits and identity IDs.

`audit.json` reports checks of every WAV's checksum, format, duration, label text,
word intervals and different-speaker intrusion (10 ms tolerance), plus 40 source
crop comparisons within PCM-16 rounding tolerance. These checks passed. They do
not establish transcript correctness or identity accuracy. Nine ambiguous or
insufficient clusters are recorded in `excluded_speakers.json`; episodes
`0j515zfr5l` and `c5bcjps40r` have no accepted Wesam identity and need review.

The standalone `review.html` supports listening, speaker/episode/text filters,
browser-local accept/reject/uncertain decisions and a downloadable decision JSON.
Restart the existing review app to enable `http://localhost:3000/dataset` and its
manifest-scoped audio routes. The page can also be served from its dataset folder.
Audio and decisions are not sent to external services. Decisions are not yet
applied to training manifests; use a reviewed subset before fine-tuning.

Forty-three regression tests pass, including rolling-caption deduplication,
shared onsets, mixed-speaker word rejection, overlap rejection, source gaps and
manifest-scoped audio serving. The original SRT export remains unchanged.

`prepare_dataset.py` uses the enrolled Wesam reference and full-episode
Community-1 diarization. It exports model-neutral mono PCM-16 WAVs and JSONL
metadata without modifying the source recordings or training a model.

```bash
cd /opt/dlami/nvme
training/wesam_review/.venv/bin/python training/wesam_review/prepare_dataset.py --check
training/wesam_review/.venv/bin/python training/wesam_review/prepare_dataset.py
```

The default destination is `training/wesam_dataset/`. A separate short pilot:

```bash
training/wesam_review/.venv/bin/python training/wesam_review/prepare_dataset.py \
  --source utrhqcji78 --limit-seconds 300 --out training/wesam_dataset_pilot
```

The default device is now CUDA. Missing CUDA fails immediately instead of silently
falling back to CPU; use `--device cpu` only for an intentional CPU run.
`--threads 4` controls PyTorch CPU threads. Both diarization and reference-matching
models are moved to CUDA, with segmentation and embedding batch sizes defaulting
to 64 (configurable via `--segmentation-batch-size` and `--embedding-batch-size`).
The isolated review environment and cached models are reused. The pipeline uses
in-memory 16 kHz input to bypass TorchCodec decoding.

For the H100, run from the host terminal:

```bash
bash /opt/dlami/nvme/training/wesam_review/run_gpu.sh
```

The launcher stops only the original CPU preparation worker, benchmarks the same
five-minute recording on CPU (batch 32) and CUDA (batch 64), records actual times,
speedup and peak allocated GPU memory in `reports/gpu_benchmark.json`, then starts
all 23 episodes in `training/wesam_dataset_gpu/`. Missing CUDA blocks the benchmark
and full run. Monitor the GPU output's `status.json` and `prepare.log`.
The old CPU outputs and pilot are preserved. The changed code and batch sizes have
a different recipe, so they require a separate output directory.

The pipeline:

1. Diarizes each whole recording so guest cluster IDs stay consistent within that
   episode. Uses regular diarization to retain detectable speaker overlap.
2. Extracts clean 3-second voice embeddings from each speaker's longest
   non-overlapping turns, up to 60 seconds. Requires at least 9 seconds of
   evidence for an identity assignment. Uses the user's 0.5 Wesam cosine threshold
   and at least 80% agreement across the sampled windows; ambiguous clusters are
   excluded. Speaker matching remains an automatic prediction.
3. Labels the matching host as `wesam_ali` across episodes. Other stable clusters
   get `guest_<episode>_<cluster>` IDs. These are anonymous episode-local speakers,
   not verified named identities or links across episodes.
4. Keeps complete aligned subtitle cues with only one detected speaker, no
   detected overlap, enough speech coverage and plausible transcript rates.
   Joins adjacent complete cues from the same speaker only when separated by at
   most 0.5 seconds with no intervening different speaker, and the combined
   duration stays below 30 seconds. Audio is one contiguous recording slice;
   distant passages are never stitched. Requires at least 3 seconds of detected
   speech as well as at least 3 seconds of clip duration.
5. Rejects mixed-speaker cues, overlaps, unknown identities, low speech coverage,
   overly long cues and music/applause markers. This exporter currently reads SRT
   cue boundaries only, so mixed cues are logged rather than attaching the entire
   transcript to one voice. All 23 original `.ar.vtt` files contain inline word
   timestamps; these are available for a subsequent word-based export, after
   removing repeated rolling captions and checking speaker-boundary timing.
6. Exports 3–30 second WAVs at native 48 kHz by default. Use `--sample-rate 24000`
   for a separate Qwen-oriented dataset or `--sample-rate 44100` for Fish Speech.
   No loudness normalization, denoising or source separation is applied.
7. Splits the 23 recordings deterministically into 19 training, 2 validation and
   2 test recordings. Clips from a recording never cross splits. The same mapping
   is used during single-source pilots and the full run.

Artifacts:

- `audio/<speaker>/<clip>.wav` and matching `.lab` transcript.
- `manifest.jsonl`, `train.jsonl`, `validation.jsonl`, `test.jsonl`.
- `episodes/<episode>.json`: speaker identity evidence, exported clips, rejected cues.
- `diarization/<episode>.json`: cached predicted turns.
- `recipe.json`, `summary.json`, `status.json`: configuration, totals and progress.

Each manifest row includes `audio`, `text`, `speaker`, `duration`, `sample_rate`,
`split`, `source_group`, `source_id`, original times and sample boundaries, subtitle
cue IDs, identity score, output audio checksum and reference fingerprint.

Rerunning the identical command reuses verified completed episode exports and
cached diarization. A changed recipe or reference requires a separate `--out`
directory so previous results remain intact. Interrupted work resumes from saved
completed episodes; an incomplete inference stage may need rerunning. An exclusive
output lock prevents two preparations from writing the same dataset simultaneously.

The files are automatic candidates pending listening review. A model-neutral
manifest does not guarantee that every model's trainer can consume it unchanged.
For a Wesam-only model, filter `speaker == "wesam_ali"`. For multiple permanent
voices, confirm guest identities and collect independent recordings for any guest
that needs recording-disjoint training and validation/test data. Episode-local
guests may occur only in one split; the Qwen runner that requires every identity
in every split will reject such a manifest until that data constraint is addressed.

Check voice identity, cross-talk, transcript accuracy and cut boundaries by
listening before treating these as a final training set. Low loss cannot establish
that the data or the resulting Arabic speech is correct.


## Original CPU run

The original CPU run was requested to stop for a GPU switch. The tool's Ctrl+C
ended its session, but the worker retained its output lock; the host launcher above
stops that worker before benchmarking. This sandbox cannot access the H100 driver,
so no measured GPU speedup is available until the host benchmark runs.

A real five-minute pilot in `training/wesam_dataset_pilot/` exported 12 guest
clips, 3.299–20.241 seconds long, totaling about 1.62 minutes. Wesam's cluster was
identified (reference cosine about 0.838; guest about 0.238), but no complete
Wesam-only subtitle cues passed the minimum-duration rules in that short pilot.
The rejected-cue reasons show why. The current exporter does not yet consume the
existing VTT word timestamps, which can help recover host turns inside mixed cues.
These original caption timestamps are not verified CTC word alignments: the
`.ar-new.srt` files have different, refined cue boundaries. Timestamp availability
for all 23 recordings is recorded in `reports/word_timestamp_availability.json`.

Twenty-eight tests pass, including mixed cues, overlaps, contiguous cue joining,
intervening-speaker rejection, minimum voiced duration, unstable identity scores,
anonymous guest IDs, recording splits and real WAV/label export.

Exact duplicate WAV content is excluded from the manifests, while episode
receipts and audio files are preserved for auditing. Train using the manifests
rather than recursively loading every WAV. Audio QC rejections are included in
per-episode rejection records. Resuming checks both WAV checksums and `.lab` text.
