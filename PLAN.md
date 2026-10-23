# PLAN — Saudi TTS (stage list from plan-v2 §11, calendar from kickoff §1)

Status: TODO | RUNNING | PASS | FAIL. A stage is PASS only with its gate number in `PROGRESS.md` or `reports/` **and** a verifier-subagent re-derivation (rule 2).
Hard stop: instance terminated from Sun 1 Nov 11:00 UTC. Final S3 sync verified by Sun 1 Nov 09:00 UTC.

| # | Stage | Target day | Gate (must pass) | Status | Evidence |
|---|-------|-----------|------------------|--------|----------|
| 0 | bootstrap | D0 Fri 23 | State files + CLAUDE.md + PROMPT.md committed and pushed; `ops/test_guard.py` all pass; `reports/s3_inventory.md` and `reports/ttsft_inventory.md` written; dashboard at https://training.aljawab.ai returns login page (200) and every data page redirects/401 without session; downloads for SADA/Omnilingual/SCC22 started; `ops/sync_s3.sh` round-trip verified | RUNNING | |
| 1 | data-real-1 (SADA step 1 + Omnilingual + SCC22 + ASWAT if present) | D1 Sat 24 | ≥ 60 h filtered real Saudi in `data/manifests/sau_real.jsonl`, two-ASR CER-agreement gate applied, global speaker clusters, eval speakers excluded; numbers in `reports/data_report.md` | TODO | |
| 2 | eval-set (`saudi_eval_v1` + eval suite + ElevenLabs references) | D0–D1 | ~1,200 items frozen with sha256 in `eval/saudi_eval_v1/MANIFEST.sha256`; 0 speaker overlap with training manifests; real-vs-real SIM ceiling and real-clip CER computed; ElevenLabs Maryam E baseline row in `reports/metrics.jsonl`; `eval/run_suite.py` runs end to end < 15 min | TODO | |
| 3 | synthetic (ElevenLabs corpus) | D1–D2 | ≥ 10 h filtered synthetic (Maryam register + ≥ 50 voices breadth) with ASR round-trip WER ≤ 0.1 and dialect check; spend ≤ cap in `elevenlabs_spend.json` | TODO | |
| 4 | data-real-2 (SADA Music recovery + MASC) | D2 Sun 25 | Real Saudi total ≥ 100 h in manifest after filters (or documented shortfall + fallback per plan §10) | TODO | |
| 5 | text front-end | D2 Sun 25 | `pipeline/textnorm` + verbaliser unit tests pass; stress-set verbalisation accuracy ≥ 95% on the 200 numeric/entity items (rule check + second LLM pass) | TODO | |
| 6 | bake-off | D1 Sat 24 | All tier-A models scored zero-shot on the 60-sentence probe (WER-O, SIM, UTMOS, dialect-ID, TTFA) next to ElevenLabs and Habibi SAU in `reports/bakeoff.md`; streaming + quality bases chosen in `DECISIONS.md` | TODO | |
| 7 | stage A (pan-Arabic warm-up) | D2 Sun 25 | Checkpoint whose dev ΔCER and SIM are no worse than base (metric row in `reports/metrics.jsonl`) | TODO | |
| 8 | stage B (Saudi adaptation, both tracks) | D3–D5 | Best checkpoint: ΔCER ≤ +1 vs real clips, unseen-speaker SIM ≥ 0.70, mean P(KSA∪Gulf) ≥ 0.8, failures ≤ 1% normal / ≤ 5% hard | TODO | |
| 9 | panel 1 | D4 Tue 27 | ≥ 60 blind pairs rated through the dashboard (Amin's votes in their own column); judge–panel ρ reported; direction check written to `reports/panel1.md` | TODO | |
| 10 | stage C + voice library | D5 Wed 28 | 3–5 frozen voices (incl. Riyadh-Najdi Maryam-like flagship + a Hijazi voice), each with reference prompt; gender/age correct; self-SIM ≥ 0.75 across 10 renders; `reports/voice_library.md` | TODO | |
| 11 | serving | D5–D6 | Container serves `POST /v1/audio/speech`, `WS /v1/audio/speech/stream`, `GET /v1/voices`, ElevenLabs shim; WER on served audio within 1 pt of offline; LiveKit/Pipecat snippets | TODO | |
| 12 | latency | D6 Thu 29 | Model-side p50 TTFA ≤ 100 ms @1 stream, p95 ≤ 200 ms @8, RTF < 0.1; ElevenLabs measured in same harness; `reports/latency_report.md` | TODO | |
| 13 | panel 2 | D7 Fri 30 | CMOS vs ElevenLabs ≥ −0.1 (bootstrap CI), dialect score ≥ ElevenLabs, MSA leakage < 5%; `reports/panel2.md` | TODO | |
| 14 | dry run (Jordanian) | D7 Fri 30 | Pipeline completes on `configs/dialects/jor.yaml` with zero code diffs (`git diff --stat` on code dirs empty); `reports/dryrun_jor.md` | TODO | |
| 15 | export | D8 Sat 31 | `reports/MODEL_CARD.md` + `reports/latency_report.md`; final checkpoint, voice library, reports synced to `s3://aljawab-asr-data/saudi-tts/` and verified by size+sha256 listing; Notion EBS-snapshot task; `DONE` | TODO | |

Other dated duties: Thu 29 Oct — `FUNDING_ASKS.md` extension recommendation + Notion task. Sat 31 Oct — Notion task: snapshot root EBS.
