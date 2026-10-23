# DECISIONS

Format: date — decision. Options · evidence · choice · how to reverse.

## 2026-10-23 — CLAUDE.md imports kickoff + PLAN, not the whole PROGRESS.md
Options: import PROGRESS.md (plan §11) vs read its tail. PROGRESS.md is append-only and will grow to thousands of lines; importing it would burn context every turn. Choice: CLAUDE.md imports docs/kickoff.md and PLAN.md; rule 1 + the SessionStart compact hook bring in `tail -40 PROGRESS.md`. Reverse: add `@PROGRESS.md` to CLAUDE.md.

## 2026-10-23 — Dataset download routes (mirrors of ledger datasets)
SADA: IEEE node 14747 has no files; the official Kaggle release `sdaiancai/sada2022` needs a Kaggle login we don't have. Using the HF mirror `khaledalganem/sada2022`, converted from that release (same 4,563 WAVs and CSVs, 77 GB). SCC22: SDAIA's official release is only links + timestamps, so audio comes from `MohamedRashad/SCC22`, plus the official CSV `SDAIANCAI/Saudilang-Code-Switch-Corpus`. MASC: `Byne/MASC` (the only mirror with country/dialect columns). The data is the ledger dataset in each case; only the transport differs. Evidence: `reports/sources_research.md`. Reverse: re-download from the official source when a login exists, and diff the CSVs.

## 2026-10-23 — Adopted from TTS-FT (credit: earlier Fish S2 Pro work on Majed-AWS; see reports/ttsft_inventory.md)
- S2 Pro LoRA trainer settings: bf16-mixed, ctx 2048, batch 1×accum 4, AdamW β(0.9,0.95) wd 0.01, constant LR after 100 warm-up steps, clip 1.0, 10 codebooks; `extract_vq --config-name modded_dac_vq` with `s2-pro/codec.pth`. Code pulled to `pipeline/legacy/` (jordanian_s2pro, talha_s2pro, wesam_pipeline, splitter, omnivoice_speaker1, chatter, speech-agent, nf4_bundle, ttsft-docs).
- **Must fix before any Slow-AR training:** upstream `BaseTransformer.embed()` skips `scale_codebook_embeddings` (÷√11, ÷√2) that `forward_generate` applies. Base step-0 loss is at chance, and Slow-AR LoRAs collapsed to near-silence in 15/15 cases. The fix gets a unit test: teacher-forced loss of the base on held-out Saudi clips must be far below chance at step 0.
- Latency gate measures **first PCM byte**, not first token: TTS-FT saw TTFT 90–125 ms but first audio 0.9–1.0 s through SGLang-Omni 0.1.3. S2 Pro's streaming advantage is unproven on this stack; the bake-off measures first PCM.
- OmniVoice full FT overfits after ~500 steps on small single-speaker data, so use LoRA or early stopping and evaluate every 250 steps.
- Not adopted: stale wheelhouse (torch 2.1.2+cu121), root-only Docker storage.
- `pipeline/legacy/speech-agent/` is kept out of git: one of its tests hardcodes LiveKit API key/secret values (redacted in our copy, never pushed or synced). Flagged to Amin in QUESTIONS Q3.

## 2026-10-23 — ElevenLabs budget is credits, not dollars (default until F1 is answered)
Creator plan, 95,419 credits left, no overage. v4 Turbo is 0.5 credit/char. Allocation: ≈30k credits for reference renders of probe + eval text (the comparison baseline), ≈60k for a Maryam E register corpus on v4 Turbo (≈2.5 h), 0 for multi-voice breadth. `elevenlabs_spend.json` tracks credits, and the guard blocks ElevenLabs commands within 500 credits of exhaustion. Reverse: if F1 is approved, raise `credits_remaining_at_start` and add the breadth corpus.
