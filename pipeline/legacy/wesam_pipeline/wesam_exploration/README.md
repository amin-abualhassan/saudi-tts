# Wesam Ali interview corpus exploration

Inspected local `README.md`, the complete `refrence.txt`, full recordings, aligned SRT files, extracted FLAC clips, transcript labels and Parquet metadata/name columns. No source data, training configuration or training splits were modified.

The other source is **wesamali.official**, identified by the README and the first 23 title-to-key mappings before the Talha playlist sections. These are interview episodes, including host Wesam Ali and different guests. Episode titles describe guests; they do not label the speaker of each audio clip.

| Property | Observed locally |
| --- | --- |
| Episodes | 23 / 23 present |
| Extracted clips | 6,365 |
| Extracted audio | 21.562 hours |
| Full original recordings | 31.431 hours |
| Format | FLAC, mono, 48 kHz for every clip |
| Clip duration | 5.0–30.0 seconds; median 10.299 seconds |
| Transcripts / SRT consistency | Every clip has a nonempty transcript matching an SRT cue with duration agreement within 50 ms |
| Structural candidates | 6,363 clips / 21.548 hours |
| Speaker annotations | None in the inspected ASR schema/transcripts |

**Structural candidates are not verified TTS training data.** The automatic checks do not establish a single speaker per clip, accurate acoustic transcripts, absence of overlap/music, or correct speaker identities.

## Findings requiring review

- Two clips exceed six transcript words per second: `omovwbnxb9-110` and `omovwbnxb9-208`. This is a plausibility flag, not proof the transcript is wrong.
- Thirteen clips have fewer than 0.75 transcript words per second. Some endings contain only phrases such as “الى اللقاء” across much longer audio.
- Thirty-five transcripts contain music/applause markers. These may describe non-speech or may be spoken words; listening is needed before excluding them.
- One transcript string repeats across three clips. Identical transcript text alone does not establish duplicate audio.
- Acoustic checks covered three deterministic examples per episode (69 total). All decoded to finite audio, none exceeded 0.1% full-scale samples, and every sampled waveform matched its original SRT interval under nearest/truncated sample boundaries. This is sampled verification; no manual listening or full-corpus acoustic audit was performed.
- The only inspected RTTM output is `Splitter/output/pred_rttms/UAE_1_mono.rttm`. Its input manifest names `UAE_1_mono.wav`, rather than a Wesam episode key. No applicable speaker annotations were found for these 23 episodes.

## Existing Parquet partitions

| Dataset partition | Total rows | Wesam rows | Wesam episodes |
| --- | ---: | ---: | ---: |
| Jordan-youtube-train.parquet | 43,854 | 6,303 | 23 |
| Jordan-youtube-validation.parquet | 442 | 62 | 22 |
| Jordan-youtube-telephony-train.parquet | 43,854 | 6,303 | 23 |
| Jordan-youtube-telephony-validation.parquet | 442 | 62 | 22 |

The standard 48 kHz train and validation files contain 6,303 and 62 Wesam clips respectively. **Twenty-two source episodes occur in both partitions.** This means the existing split does not measure generalization to unseen recordings. The telephony partitions have the same source-row counts and an additional `degradation_profile` column; their audio payloads were not decoded in this exploration.

## Implications for the pending Qwen training work

1. For a Wesam-only voice, first separate his speech from guest speech across episodes, verify the recurring host identity, and keep source episodes disjoint between training and evaluation.
2. For multiple permanent voices, label individual speaker turns and link identities across recordings. Remove or re-cut overlapping/mixed-speaker clips, then align each new cut to its transcript. A channel or episode ID is not a speaker ID.
3. Most named guests appear in only one title in this reference list. The current Qwen runner requires train/validation/test examples for every speaker and forbids source-recording overlap. A guest represented by one recording cannot satisfy both rules. Additional independent recordings or an explicitly different evaluation design are needed before a guest-inclusive multi-speaker manifest can be accepted.
4. The inventory intentionally has `speaker: null` and no training split. Do not feed it directly to the Qwen runner or assign every interview clip to one speaker. No fine-tuning was started.

## Per-episode inventory

| Source ID | Clips | Clip hours | Full hours | Original title |
| --- | ---: | ---: | ---: | --- |
| 2366wqj7sw | 255 | 0.895 | 1.263 | 20241224_جلال ابو مويس： انا بتعمد إستفزاز الناس والغربة سرقت مني كثير ｜ برنامج تحت الضغط مع وسام علي (Part 1) |
| 6n2esa018j | 187 | 0.667 | 0.990 | 20241227_جلال ابو مويس متأثراً： قدوتي بالحياة ما كان مليونير ｜ برنامج تحت الضغط مع وسام علي (Part 2) |
| 5flgspymra | 444 | 1.432 | 2.244 | 20250103_سلامة ： مظلوم بالسجن وبالحب وبالطلاق ｜ وهاي قصة زواجه بالسر ｜ برنامج تحت الضغط مع وسام علي |
| epabqc9ast | 254 | 0.896 | 1.357 | 20250109_ابن سوريا ： سبب اختفاؤه و موقفه من المعارضة و النظام  ｜ برنامج تحت الضغط مع وسام علي |
| utrhqcji78 | 389 | 1.265 | 1.952 | 20250116_ابو الصادق ： ببكي بحُرقة ، و البشر طلعوا مش بشر ｜ برنامج تحت الضغط مع وسام علي |
| omovwbnxb9 | 426 | 1.497 | 1.998 | 20250130_ماهركو： جربت المخدرات ، و هاي قصة اكتئابي ｜ برنامج تحت الضغط مع وسام علي |
| i2clnz70i8 | 286 | 1.016 | 1.467 | 20250206_معاذ العمري باكياً： ما في افضل من ابوي ، و ديانا كرزون افضل أُم في العالم ｜ تحت الضغط مع وسام علي |
| xs9hy6dwtn | 227 | 0.807 | 1.109 | 20250215_ضحّكني ، فَ بَكِّيته ｜ عبدالله صبيح في تحت الضغط مع وسام علي. |
| vngwvadz85 | 459 | 1.353 | 2.361 | 20250310_ممنوع تحب أكثر من الله ｜ الشيخ علاء جابر في تحت الضغط مع وسام علي. |
| c5bcjps40r | 335 | 1.075 | 1.675 | 20250509_سيلاوي： امك بتعملك سندويشتك ، أنا امي رافعة علي ٧ قضايا ｜ #تحت_الضغط مع وسام علي |
| qc67lna1qb | 323 | 1.006 | 1.626 | 20251111_عمار النجار： ضد الجنس بالكوميديا، والمجتمع ما بتقبل البنات الناصحين! |
| 690bdx2a7x | 291 | 0.991 | 1.442 | 20251120_أديب دعسان ｜ طردوني من المانيا، و صاحبي انقتل لهالسبب! |
| 0j515zfr5l | 291 | 1.031 | 1.468 | 20251205_محمد نبيل ｜ الصحابة لا يُمثلون الدين، وما أظُن إنّي تخطّيت! |
| nzc21rdbcn | 409 | 1.485 | 2.001 | 20251220_فاديا الطويل ： حاولت انتحر لهذا السبب, وتمردي كان الحل. |
| rikl276p9j | 525 | 1.714 | 2.469 | 20260101_زيد المصري： تارك الصلاة ملعون ولا يزوج..واتحدى الجن على الهواء＂ |
| w5ri83j0ly | 115 | 0.382 | 0.555 | 20260220_أدمن على ايذاء نفسه جسدياً ، و السبب كان صادم و مؤسف ! |
| 8h7r4cs81s | 186 | 0.687 | 0.921 | 20260224_ايمن يروي قصة ادمانه على الحب ، الأسباب و المشاكل و الحلول . |
| 1zm10wn4u4 | 70 | 0.247 | 0.349 | 20260228_ادمان العادة السرية ، اولها دلع و آخرها ولع. |
| piekf44qr0 | 151 | 0.518 | 0.729 | 20260303_ادمان المخدرات، بين رهن الأبناء و زنا المحارم و المجتمع . |
| j27tmdq22g | 142 | 0.497 | 0.687 | 20260313_إدمان إبر الهرمونات ، بين الضعف الجنسي والمنظر الخارجي .. |
| j3bw52pgye | 128 | 0.410 | 0.579 | 20260315_إدمان القمار الإلكتروني：اللعبة اللتي يبيعها المشاهير لمتابعينهم! |
| zmjh0hwrup | 276 | 0.951 | 1.253 | 20260317_هاي الحلقة ممكن تغير حياتك ، للأبد ! |
| zsbp3t7j3m | 196 | 0.740 | 0.936 | 20260318_وسام علي وغيـّاث محمد نوح القضاة ｜ بين الدنيا والدين ! |

## Artifacts

- `report.json`: summary, per-episode figures, acoustic sample checks, manual-review shortlist and Parquet partition evidence.
- `inventory.jsonl`: all 6,365 clip records with recovered SRT intervals and audit flags.
- `structural_candidates.jsonl`: 6,363 clips passing the structural checks; speaker identity remains unassigned.
- `explore.py`: reproducible audit script. Run from the workspace with `miniconda3/envs/fish-speech/bin/python training/wesam_exploration/explore.py`.
