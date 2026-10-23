# PROGRESS (append-only, UTC)

2026-10-23T21:01:31Z START bootstrap. Session started by Amin via /goal. Repo d3c2a64. GPU idle (H100 80GB, 0 MiB used). NVMe 3.3T free, EBS 79G free.
2026-10-23T21:01:31Z JOB pid=161530 kind=data log=/opt/dlami/nvme/inventory/s3_inv.log  S3 recursive inventory of aljawab-asr-data -> /opt/dlami/nvme/inventory/s3/
2026-10-23T21:01:31Z Subagents launched: TTS-FT read-only inventory -> reports/ttsft_inventory.md; dataset/model download-route research -> reports/sources_research.md
2026-10-23T21:05:23Z Watchdog tmux session present. Bootstrap skeleton committed db2e7fa.
2026-10-23T21:09:03Z Dashboard LIVE at https://training.aljawab.ai (Caddy 2.6.2 auto-HTTPS -> uvicorn 127.0.0.1:8090, systemd saudi-dashboard). Tests: / -> 303 /login; /login 200; /logs 303; HX 401; :80 -> 308 https; wrong pw 401; right pw shows QR once (not enrolled); bad TOTP 401; 6th failure -> 429. TOTP enrolment left for Amin's first login.
2026-10-23T21:09:35Z JOB pid=167770 kind=data log=/opt/dlami/nvme/logs/fetch_models_1.log  tier-A weights: s2-pro, OmniVoice, VoxCPM2
2026-10-23T21:10:24Z reports/s3_inventory.md written (no Saudi corpus in bucket; Jordan/ = 158 h aligned 48 kHz Jordanian, already mirrors TTS-FT data). Raw listings synced to saudi-tts/inventory/.
2026-10-23T21:10:41Z ENDED pid=167770 (import clash: pipeline/secrets.py shadowed stdlib; renamed to pipeline/keys.py). JOB pid=168642 kind=data log=/opt/dlami/nvme/logs/fetch_models_1.log  tier-A weights retry
2026-10-23T21:10:42Z reports/s3_inventory.md written (no Saudi corpus in bucket; Jordan/ = 158 h aligned 48 kHz Jordanian, mirrors TTS-FT data). Raw listings synced to saudi-tts/inventory/.
2026-10-23T21:12:19Z JOB pid=169370 kind=data log=/opt/dlami/nvme/logs/envs_tierA.log  uv envs fish/omnivoice/voxcpm
2026-10-23T21:13:12Z JOB pid=170009 kind=data log=/opt/dlami/nvme/logs/fetch_sada.log  SADA 77 GB (khaledalganem/sada2022 mirror of SDAIA Kaggle)
2026-10-23T21:13:12Z JOB pid=170010 kind=data log=/opt/dlami/nvme/logs/fetch_datasets.log  Omnilingual ars/acw/afb, SCC22 (+official CSV), MASC 134 GB
2026-10-23T21:13:12Z JOB pid=170011 kind=data log=/opt/dlami/nvme/logs/fetch_models_2.log  MOSS local+codec, Habibi SAU, eval models, tier B
