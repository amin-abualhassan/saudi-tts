# SGLang voice studio on port 3000

Verified running on the H100 on 2026-10-11 in detached screen session
`jordanian-sglang`. Both selected models are resident; all 30 paired clips,
both uploaded-voice checks and studio HTTP requests passed. Warm short requests
take about 0.7–0.8 seconds for 2.6–3.1 seconds of audio. A new reference incurs
CPU encoding overhead: first requests measured about 14–19 seconds; subsequent
requests reuse the reference. Review voice identity and pronunciation at
`http://<server-ip>:3000/sglang-compare`.
See the [live validation report](reports/sglang_live_validation_20261011.json).

The studio includes **Keep ending volume steady**, enabled by default for
generated playback and WAV downloads. It gently raises quieter speech in the
last portion (at most three seconds), with a 9 dB gain limit, gradual gain,
silence gating and peak protection. Earlier speech and duration remain intact.
Uncheck it to hear the raw output; **Original WAV** always downloads the untouched
generation. The setting also works on existing clips. Raw comparison clips and
benchmark measurements remain available at `/sglang-compare`.

This is an output volume correction; it does not change generation, checkpoints,
pronunciation or inference settings. Corrected WAVs are cached separately in
`voice/leveled/`, with source/algorithm hashes and gain receipts. Five recent
uploaded-voice clips measured approximately 2–5 dB improvement in last-second
RMS, with about 20–25 ms processing. All 59 regression checks pass. See the
[ending-volume report](reports/ending_volume_validation_20261011.json); listening
review of the correction remains necessary.

To attach to the running session:

```bash
screen -r jordanian-sglang
```

If Docker image export reports **no space left on device**, use this one command:

```bash
bash /opt/dlami/nvme/training/jordanian_s2pro/start_voice_server_sglang.sh --nvme-docker
```

It requests sudo once for storage setup, copies the existing Docker/containerd
stores and build cache to `/opt/dlami/nvme/sglang_docker_storage/`, then continues
the normal port-3000 launcher. It restarts the idle Docker and containerd services;
it refuses migration when containers or containerd tasks are running, or when
restarting Docker would start a stopped `always` container. The original Fish
voice studio continues through this storage step because it runs outside Docker.

Both stores move: Docker's `data-root` and containerd's top-level `root`.
[Docker documents these separate storage paths](https://docs.docker.com/engine/daemon/#configure-the-data-directory-location).
The original directories are retained, NVIDIA/runtime settings are preserved,
configuration backups are kept under the new storage directory, and failed
copy/configuration/restart steps restore the original configuration and services.
No images, datasets or checkpoints are pruned. Startup guards require NVMe to
be mounted before either service can write to it. A successful setup is reused
on future launches, including the ordinary command below.

To inspect the storage plan without changing services:

```bash
/usr/bin/python3 /opt/dlami/nvme/training/jordanian_s2pro/sglang_docker_storage.py
```

The host migration completed on 2026-10-11. Docker and containerd both use NVMe;
the original stores remain retained. The system filesystem has approximately
41G free and NVMe has 2.8T free after startup. The migration receipt is
`/opt/dlami/nvme/sglang_docker_storage/migration.json`. See the historical
[storage fix report](reports/sglang_storage_fix_20261011.json) for its design.

Run this one command inside your screen session on the H100 host:

```bash
bash /opt/dlami/nvme/training/jordanian_s2pro/start_voice_server_sglang.sh
```

Open `http://<server-ip>:3000`. The interface serves `fast_r32_final` and
`attention_mlp_r16_early`, the same three speaker references, and your uploaded
or recorded voice with its transcript. SGLang's two private speech APIs listen
on `127.0.0.1:3001` and `127.0.0.1:3002`; the browser uses only port 3000.

The first launch builds an isolated Docker image, verifies the original model
and selected checkpoints, checks CUDA/BF16 and validates engine configuration.
These steps happen before stopping the idle existing studio. It refuses to
switch while requests are unfinished. The original server and SGLang studio
share the same voice data and server lock, so they run one at a time to release
GPU memory when switching. A different application occupying port 3000 is left
alone. Docker with NVIDIA Container Toolkit and an H100 driver compatible with
the image's CUDA 13 runtime must already be available on the host.

The container uses SGLang-Omni 0.1.3, SGLang 0.5.16, Torch/Torchaudio 2.11.0,
and Transformers 5.12.1. Packages install into `/opt/sglang-venv` inside the
container; Python and `sgl-omni` run from that environment. It can read the base
image's preinstalled packages, while installation writes target the virtual
environment. This avoids Debian's externally managed system-Python restriction.
The launcher hashes the Dockerfile, override file, build check and Docker context
rules, and rebuilds the image automatically when any of these inputs changes;
rerun the same Bash command after a failed build. The environment setup follows
the [uv Docker guide](https://docs.astral.sh/uv/guides/integration/docker/).
The existing Fish environment is unchanged. The image
uses a versioned base tag; the resulting immutable local image ID and resolved
versions are recorded for the run. Container installation, codec CPU execution,
H100 CUDA-13/BF16 preflight and actual two-engine speech generation have passed.
The selected release's
[installation instructions](https://github.com/sgl-project/sglang-omni/blob/v0.1.3/docs/get_started/installation.md)
and [dependencies](https://github.com/sgl-project/sglang-omni/blob/v0.1.3/pyproject.toml)
are the source for these pins.

`sglang-overrides.txt` explicitly applies the release's protobuf override when
installing the packaged distribution. A regular `protobuf>=6.31.1,<7` requirement
alone conflicts with audiotools 0.7.2's obsolete `<3.20` bound; an override replaces
that bound. Dependencies continue to be resolved. The build verifies the installed
versions, protobuf major version and codec/TensorBoard imports, and runs a small
randomly initialized Descript codec forward pass on CPU before producing the image.
See the [dependency fix report](reports/sglang_protobuf_fix_20261011.json).

Both engines use BF16, FA3, Torch compilation and CUDA graphs. Each is limited
to one running request, a 4096-token context, 8192 total KV tokens, graph batch
size one and static GPU-memory fraction 0.32. Generation is serial through the
studio's existing bounded queue. Both models stay resident. Initial kernel
compilation can take substantial time; inspect the logs below for progress.
Images, exact checkpoint exports and compiler caches are reused on later runs.
The installed ServerArgs uses `cuda_graph_bs_decode` and
`cuda_graph_max_bs_decode`; the pipeline builder supplies its 4096 context
directly, so `context_length` must not be duplicated in its overrides.

## Export verification

Native HF exports have already been created under `sglang/models/`. Every LoRA
target is merged, including fast embeddings/output: 44 adapter tensors / 22
weights for rank 32, and 400 tensors / 200 weights for attention+MLP. Original
HF names, configuration, tokenizer, tied embeddings and codec are retained.
Each export has source and output SHA-256 receipts. Incomplete or modified
exports are rejected rather than silently reused. Training checkpoints are
preserved. Unchanged large artifacts are linked to the verified original base.

The audit in `sglang_validate_exports.py` loads each complete selected adapter
and native export in the Fish implementation on CPU, checks all 358 state
tensors, and compares slow and fast logits for a fixed four-frame input.
See `reports/sglang_export_validation.json` for actual results. This checks
weight preservation; it does not establish SGLang sampling or audio parity.

## Listening and timing

After each engine starts, a real voice-cloning canary verifies the full path.
On the first normal launch, 15 comparison cases per model are generated using
the original text and references. Successful matching outputs are reused on
later launches; failed cases are retried. Two fresh repeated-reference requests
and one PCM streaming probe per model measure warm generation and first audio.
An existing uploaded reference is checked if its exact transcript is available
in prior requests; otherwise custom-voice validation awaits your upload.

Open `http://<server-ip>:3000/sglang-compare` for paired original/SGLang playback.
WAVs and metrics are stored in `voice/sglang-benchmark/` and
`voice/sglang-benchmark.json`. Full-request RTF is elapsed time divided by output
duration. Original timings are historical; they are not a fresh simultaneous
baseline benchmark. Cached paired-clip timings also retain their original
measurement; the warm probes run fresh on each normal startup.
Streaming first-audio timing measures receipt of the first
4096 raw PCM bytes, including that buffering delay. The studio itself plays
completed WAVs. Whole-GPU memory is sampled every second in
`voice/sglang-gpu-memory.json`; this is an observed peak, not an allocator peak.

Listen for voice identity, Jordanian accent, missing/repeated words and stability.
SGLang sampling can differ from Fish, even with the same weights and seed. Seed
support is detected from the API schema, or the installed CreateSpeechRequest
fields when OpenAPI omits the raw request body, and reported with every result. Audio
is checked for sample rate, duration, finite values and silence, but these
checks cannot establish pronunciation or voice quality automatically.

## Separate streaming endpoint

Open `http://<server-ip>:3000/stream` for live playback and timing. The normal
studio at `/`, `/api/generate`, ending-volume checkbox and existing downloads
continue to work. Streaming uses the same selected models, saved speaker/upload
references and serial bounded queue; it does not load extra model copies.

Live H100 tests passed on both models, including an uploaded voice, normal
generation and disconnected-client recovery. All 70 regression checks pass.
Warm TTFT measured approximately 90–125 ms, with first audio arriving locally
in approximately 0.9–1.0 seconds. A new upload measured about 13 seconds of
reference encoding before its first token. Chromium 153 verified playback
starting before generation completed, finishing playback, stopping playback
and the normal studio's generation/volume control. The headless browser's
audio output was muted; human listening review remains necessary. See the
[browser report](reports/sglang_stream_browser_validation.json) and
[screenshot](reports/sglang_stream_browser.png).

The page displays queue wait, **first generated audio token (TTFT)** and
**first audio received**. TTFT is measured after S2 Pro collects its first actual
Slow-AR/Fast-AR audio frame; the container writes a timestamp only for explicitly
tagged streaming requests. It includes HTTP/backend work and reference encoding,
but excludes the studio queue. The page's first-audio time includes the studio
queue and browser/network transport. A separate decoder-only TTFT is also saved
as `engine_time_to_first_token_seconds`. Missing token telemetry is reported as
unavailable, never substituted with first-audio latency.

`POST /api/stream` accepts the same JSON fields as `/api/generate`:

```bash
curl -N http://127.0.0.1:3000/api/stream \
  -H 'Content-Type: application/json' \
  -d '{"model":"fast_r32_final","reference_id":"talha_ahmad","text":"كيفك اليوم؟","seed":42}'
```

The response is `text/event-stream`: `accepted` supplies the job ID; `started`
supplies queue wait; `token` supplies TTFT; `audio` carries base64 `pcm` chunks
in signed 16-bit little-endian mono at 44,100 Hz; `complete` supplies timing,
duration and original/volume-corrected WAV URLs. `error` terminates a failed
stream. The browser schedules PCM buffers as they arrive; it does not wait for
the completed WAV. `/api/stream/references` lists saved references, including
uploaded voices and known exact transcripts.

Live playback uses the raw generated audio. The final three seconds cannot
receive the existing full-file ending correction before the ending is known;
the corrected download remains available after completion. **Stop playback**
closes the stream/player; generation may finish in the normal studio and save
its WAV. Disconnects, slow clients and worker restarts release subscriptions.
There are at most eight subscriptions, with bounded event buffers. SSE must
remain unbuffered if a reverse proxy is placed in front of this server.

The one Bash launcher remains `start_voice_server_sglang.sh`. Token timing uses
a thin configured engine factory from the mounted workspace, not a container
package replacement. Engine startup validates the pinned runner's timing hook.
It waits for the assigned private port to be released and uses strict-port mode
to prevent SGLang from silently choosing a different address during restarts.
PCM payloads are omitted from progress logs; token/completion metrics remain.
Current verification is recorded in
[streaming validation](reports/sglang_stream_validation_20261011.json).

For a quicker startup that runs canaries and skips comparison generation:

```bash
bash /opt/dlami/nvme/training/jordanian_s2pro/start_voice_server_sglang.sh --skip-benchmark
```

`--prepare-only` builds and validates without stopping the existing studio.
`--check` verifies sources and exports on CPU without Docker or CUDA.
`--gpu N` selects a host GPU; `--port N` changes the public port and the two
loopback API ports to N+1 and N+2.

## Logs and fallback

- `voice/server.log` and `voice/status.json`: studio and worker progress.
- `sglang/fast_r32_final.log`, `sglang/attention_mlp_r16_early.log`: engine startup,
  compilation, inference and errors.
- `sglang/manifest.json`: active image, checkpoints, configuration paths and
  container preflight results. It is committed only after acquiring the voice lock.
- `reports/sglang_live_launch.log`: actual detached-screen launch and preflight.
- `reports/sglang_live_validation_20261011.json`: successful live deployment,
  measured timings, memory and linked HTTP/audio checks.
- `reports/sglang_launch_attempt.log`: historical Docker permission failure
  before host access was enabled.

Ctrl-C exits and removes this worker's containers. Retry server cleans up and
restarts the SGLang worker. Startup errors remain visible; the system does not
silently switch to eager inference or claim a speedup.

To return to the original port-7860 server once outstanding jobs have finished:

```bash
bash /opt/dlami/nvme/training/jordanian_s2pro/start_voice_server.sh --replace-voice-server
```

Both live engines use FA3 and CUDA graphs, with steady decode logs around
98–100 tokens/second. Warm request RTF is approximately 0.25–0.27, versus the
historical original decoder's approximately 1.84; these are different runs,
not a simultaneous controlled baseline. Both engines plus an unrelated resident
Talha application occupy about 53.3 GiB of the H100's reported 79.6 GiB. Live HTTP
and audio checks passed; browser microphone interaction and listening quality
still need user review.
