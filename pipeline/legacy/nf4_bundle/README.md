# attention_mlp_r16_early: packed NF4 derivative

This is the **Fish Speech + bitsandbytes fallback, not SGLang**. The packed
weights come directly from your exact merged export revision
`392cfb1cf6824476e616df3d0459986efdd268a13991737fdbb8260d200a5cd4`.
No adapter, public base quantization, retraining, or second BF16 transfer is used.
The original service, model, environments and credentials were left unchanged.

The full H100 results and limitations are in `VALIDATION.md` and `reports/`.
**RTX 4070 Laptop SM89 fit and latency are UNVERIFIED.** At the recorded 2 GB
available host RAM, there is insufficient headroom for the measured runtime.
Free memory before trying this; allow at least 6 GiB available host RAM as an
engineering allowance, not a verified laptop requirement. Keep the app's
environment separate. This runtime uses CUDA 12.8 PyTorch wheels; the existing
Windows NVIDIA driver must support that runtime. Do not replace host CUDA/Torch.

## Windows verification and extraction

Use Python's standard library; no Torch installation is needed on Windows.
Extraction was exercised on Linux Python 3.14 with Windows path-rejection tests;
execution on Windows itself remains unverified.
Download the ZIP, `verify_extract.py`, and `TRANSFER.json` into a NEW directory.
Use the ZIP SHA-256 supplied in the delivery receipt, independently of the ZIP.

```powershell
py -3 verify_extract.py --zip attention_mlp_r16_early-nf4.zip --sha256 <ZIP_SHA256> --destination D:\AljawabNF4\release --codec D:\Models\attention_mlp_r16_early\392cfb1cf6824476e616df3d0459986efdd268a13991737fdbb8260d200a5cd4\codec.pth
```

If using checksummed 1 GiB binary parts, download all listed parts and the
transfer receipt, and add `--parts-manifest TRANSFER.json`. The tool joins into
the NEW `--zip` path, verifies each part and the complete ZIP, then extracts.
Do not download both the ZIP and its parts. Input parts are never deleted.
Existing destinations, checksum errors, archive links, traversal, Windows device
names, alternate streams and case-colliding paths are rejected. Failed restores
can leave a partial NEW destination; use another NEW folder after correcting it.

`manifest.json` lists every payload file's size and SHA-256. The manifest itself
is covered by the outer ZIP checksum. The unchanged external codec is omitted
and MUST have size 1,871,099,728 and SHA-256
`74fc41c5a7151c6f350af8bd7e5d6e3accfcc7f3dfbfac23afd35af07052bb2f`.
The included Noura WAV retains SHA-256
`d3290b0274478217ddb09145e1e3c118b50eb23b76579d67ce093798bee4432f`.
Retain the original BF16 files and their original manifest unchanged.

## WSL2/Linux runtime

Use WSL2 with GPU access and Docker's NVIDIA integration already working.
Build/run these commands inside Linux, not in the Windows app virtualenv.
The image contains pinned public wheels, not an environment dump or locally
compiled H100 kernels. A build needs Internet package downloads; model inference
is offline. The base image is pinned by digest, and every Python dependency is
pinned in `requirements.lock`. Allow roughly 20 GiB free Linux disk space for
build/download/image overhead in addition to the release and existing codec.

```bash
cd /mnt/d/AljawabNF4/release
docker build -t aljawab-fish-nf4:20261017 .
export CODEC_PATH=/mnt/d/Models/attention_mlp_r16_early/392cfb1cf6824476e616df3d0459986efdd268a13991737fdbb8260d200a5cd4/codec.pth
bash launch.sh
```

The host binds **127.0.0.1:3012 only**. Container paths are `/data/model`,
`/original/codec.pth`, and `/references/noura_reference.wav`. All bind mounts are
read-only. One active request, context 2048, maximum 1024 new frames, eager
execution, no compilation or CUDA graphs, and PyTorch SDPA are used. Reference
encoding runs on CPU FP32. The codec then moves to GPU BF16 for bounded decoding.
One reference is cached; changing it incurs encoding again. The codec weights
are never quantized. A short prompt must still fit the 2048 context together
with reference codes and the requested output bound; oversized requests fail.

Run health and smoke checks in a second Linux terminal:

```bash
curl --fail http://127.0.0.1:3012/health
curl --fail http://127.0.0.1:3012/v1/models
docker exec aljawab-nf4-laptop python /opt/runtime/http_smoke.py --url http://127.0.0.1:3012 --reference /references/noura_reference.wav --reference-json /data/reference/reference.json --output /tmp/smoke
```

The smoke script sends actual WAV and raw PCM streaming requests, verifies
PCM16 little-endian mono 44.1 kHz, records first-byte times, verifies WAV/PCM
payload equality for a fixed seed, and rejects original-model identity and
out-of-root references. To retain its files, use a separate Python 3.10 Linux
venv with the pinned lock and run `runtime/http_smoke.py` on the host with
`--reference-json reference/reference.json --output /NEW/smoke-results`.
Do not install GPU dependencies in the local app environment.

The engine model ID is
`attention_mlp_r16_early-nf4-bd78f331a5a1534c`. `/v1/audio/speech` accepts:

```json
{"model":"attention_mlp_r16_early-nf4-bd78f331a5a1534c","input":"Hello, how can I help you?","references":[{"audio_path":"/references/noura_reference.wav","text":"هذا هو السؤال، فأحنا عندنا أول نظام للشركات كان بعهد الملك فيصل عشان بس نحطه في سياق من خمسين سنة بعدين صدر نظام"}],"response_format":"pcm","stream":true,"seed":42,"temperature":0.7,"top_p":0.7,"top_k":30,"max_new_tokens":768}
```

For nonstreaming use `"response_format":"wav","stream":false`.
Streaming emits decoded frame windows while AR generation is still running.
It does not send a WAV header or buffer the complete utterance. The codec uses
16-frame chunks, 8 frames of future context, and up to 128 past frames. This
bounded-window adaptation can differ from full-utterance codec decoding at
boundaries; paired samples use the same window policy on BF16 and NF4.

For CPU codec placement, replace the launch command with the same `docker run`
arguments from `launch.sh`, followed by the image name and:
`--container --model /data/model --codec /original/codec.pth --reference-root /references --port 3012 --codec-device cpu`.
This reduces GPU pressure but retains FP32 codec weights/activations in RAM and
uses the laptop CPU for every audio chunk. See the measured tradeoff in the report.

## Deliberate Aljawab app integration

The existing gateway and app pin the original BF16 checkpoint identity. They
must reject the derivative until explicitly configured. Use a separate app
copy; preserve its credentials locally and never copy them into this release.
Apply the supplied patches with the app stopped:

```powershell
py -3 D:\AljawabNF4\release\integration\apply_integration.py --app D:\AljawabAppNF4
```

This checks the four original source hashes, backs them up, and updates only
`tts/gateway.py`, `tts/model_files.py`, `app/config.py`, and `app/speech.py`.
It does not change `tts/model-artifacts.json` or disable hash checks. The new
derivative manifest is itself pinned, and every required derivative file is
hashed. Add/update only these values in that app copy's `.env`:

```dotenv
SGLANG_URL=http://127.0.0.1:3012
SGLANG_MODEL_PATH=D:/AljawabNF4/release/model
TTS_DERIVATIVE_MANIFEST=D:/AljawabNF4/release/integration/derivative-model-artifacts.json
TTS_ENGINE_MODEL_ID=attention_mlp_r16_early-nf4-bd78f331a5a1534c
TTS_CHECKPOINT_SHA256=bd78f331a5a1534cb8bfecc6c0a024db7c0c5267a6c9da806a2dde1e18dd29af
TTS_VERIFY_MODEL_HASHES=true
TTS_ENGINE_REFERENCE_MAP='{"D:/AljawabAppNF4/assets/voice":"/references/assets/voice","D:/AljawabAppNF4/tts-data/uploads":"/references/tts-data/uploads"}'
```

The names `SGLANG_URL`/`SGLANG_MODEL_PATH` are retained for the app's configuration
compatibility; the engine is explicitly Fish NF4. Before launching the runtime
for this integration, set Linux `REFERENCE_ROOT=/mnt/d/AljawabAppNF4`; it will
mount at `/references`. This maps Windows local reference paths to absolute
Linux paths. The app's existing Noura transcript/hash and other credentials
remain in its own environment. Start only this separate app copy with its
existing `python -m scripts.run_local --with-tts` command.

**Cached clips are derivative-specific.** Back up the selected app copy's
`assets/voice` before regenerating. Run its existing
`python -m scripts.generate_clips` after the NF4 runtime and patched gateway
are ready. Original cached clips retain their original identity and are
rejected by the derivative configuration; never relabel them. The delivered
greeting samples demonstrate synthesis but are not a full regenerated app cache.
To revert, restore the four `.py.bf16-original` backups and original `.env`
values, and restore the BF16 cached voice assets. No gateway hash bypass is needed.

## Exact conversion and reload

Pinned source: `groxaxo/fish-speech-int4-patch` commit
`fc4e1e24ff3b8d7d28fdd66e6789f23acb63c5bb`. `fish-source/` contains the required
source and license. `runtime/fish-source.patch` records the frame callback,
SM89-compatible attention selection, and bounded prompt check. `runtime/weights.py`
replaces the candidate's permissive/full-model loader and exporter with strict
tensor coverage, meta initialization, per-tensor loading and packed-state saving.

```bash
python3.10 -m venv /NEW/fish-nf4-env
/NEW/fish-nf4-env/bin/pip install -r requirements.lock
source /NEW/fish-nf4-env/bin/activate
bash convert.sh /exact/verified/392cfb1cf6824476e616df3d0459986efdd268a13991737fdbb8260d200a5cd4 /NEW/derivative
PYTHONPATH="$PWD/fish-source:$PWD/runtime" python runtime/kernel_probe.py --model model --report /NEW/kernel-probe.json
PYTHONPATH="$PWD/fish-source:$PWD/runtime" python runtime/validate.py --model model --codec "$CODEC_PATH" --only ar_greeting
```

The converter refuses an existing output folder and rechecks both parent
shards and codec before conversion. Reload never reads parent shards; the
kernel probe makes any attempted runtime requantization fail. The loader
requires this delivered derivative's SHA-256. A newly reproduced artifact with
a different hash must be audited and deliberately assigned a new identity.
No runtime quantization from BF16 is needed on the laptop.
