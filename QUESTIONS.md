# QUESTIONS for Amin

Format: `## Q<n> [OPEN|RESOLVED] title` · question · default I am taking · answer (write below "Answer:").

## Q1 [OPEN] Rotate the Claude OAuth token
On 23 Oct ~21:00 UTC I printed `~/.secrets/claude.env` while listing key names (my sed pattern missed the `export ` prefix), so the CLAUDE_CODE_OAUTH_TOKEN value is now in this session's transcript (local `~/.claude/projects/` and the Remote Control session). Nothing was committed, synced or sent elsewhere.
Default: none — please run `claude setup-token` again when convenient, replace the value in `~/.secrets/claude.env`, and revoke the old token in claude.ai settings. I will not touch the file.
Answer:

## Q2 [OPEN] ElevenLabs is on the Creator plan: 95k credits left, no overage possible
The account is `creator` tier: 147,270 credits/month, 51,851 already used, resets 16 Nov (after the GPU window), and `allowed_to_extend_character_limit = false`, so the $50 cap can't be spent per character at all. eleven_v4_turbo costs 0.5 credits/char, eleven_v4 1.0. That's ≈190k Turbo characters ≈ 4 h of audio, vs the plan's 25–40 h synthetic corpus over ≥ 50 voices.
Default (no spend): (1) ≈30k credits for ElevenLabs reference renders of the probe + eval text sets (the comparison baseline every gate needs); (2) the remaining ≈60k credits for a ~2.5 h Maryam E target-register corpus on v4 Turbo; (3) no multi-voice breadth corpus — speaker breadth comes from real data (SADA, Omnilingual, MASC). Request: see FUNDING_ASKS.md F1 (upgrade to Pro for one month).
Answer:

## Q3 [OPEN] Can TTS-FT (Majed-AWS) be stopped?
From this project's side, yes: everything needed is pulled (code/configs to `pipeline/legacy/`, the Noura reference and the two selected S2 Pro adapters to S3 `saudi-tts/keep/legacy_ttsft/`), and all its raw data is byte-identical in `s3://aljawab-asr-data/Jordan/`. But its NVMe is instance store: stopping it erases every Jordanian checkpoint, the NF4 bundle and Docker images, and it currently serves the public **Noura** voice-agent demo (port 8002 via Cloudflare tunnel, SGLang studio on 3000). See `reports/ttsft_inventory.md` §4.
Default: I take nothing more from it unless asked; please confirm with whoever owns the Noura demo before stopping it.
Answer:
