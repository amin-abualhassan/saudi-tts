# Word-timed podcast dataset

Completed: 23 episodes, 7,102 mono 48 kHz PCM-16 WAVs, 20.005 hours.
Wesam: 1,061 clips, 1 hour 58 minutes 53 seconds. Guests: 25 episode-local IDs.
All exported clips are 3–30 seconds with at least 3 seconds of detected speech.

Use `manifest.jsonl` or the `train.jsonl`, `validation.jsonl`, `test.jsonl` splits.
Each WAV has a matching `.lab`. Manifest rows include original recording times,
speaker evidence, relative word intervals, caption IDs and source/output hashes.

The VTT captions contain word starts; interval ends are inferred, not verified
forced alignment. Speaker labels are automatic predictions. Listen before
fine-tuning: check speaker identity, transcript accuracy, cross-talk and boundaries.
Structural/sample verification passed; see `audit.json`. Deduplication/text-count
consistency is recorded in `transcript_consistency.json`. Identity exclusions are
in `excluded_speakers.json`.

For listening, restart the existing review server and open
`http://localhost:3000/dataset`. The generated `review.html` provides speaker and
episode filters, audio playback and downloadable browser-local review decisions.
Decisions do not automatically change manifests.

The original SRT export and GPU diarization remain in `../wesam_dataset_gpu/`.
Refinement required no further model inference. See
`../wesam_review/DATASET.md` for reproducible commands and limitations.
