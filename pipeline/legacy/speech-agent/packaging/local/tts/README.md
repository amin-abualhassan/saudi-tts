# Exact TTS model, Noura reference, and engine contract

Required model: **attention_mlp_r16_early**. Required source checkpoint SHA-256:

```text
9dbf9c8b6ee71dc5c737e6a76f2166539e833a6aa8ab274fb192553714a294cf
```

The running deployment uses the **merged Hugging Face-style export**, revision `392cfb1cf6824476e616df3d0459986efdd268a13991737fdbb8260d200a5cd4`. `model-artifacts.json` and `model_config/export.json` record all artifact hashes. The generic S2 Pro base shares configuration/tokenizer files but has **different learned shard hashes**. Do not use it as a silent replacement.

## Transfer the exact learned artifacts

Large weights were excluded because the requested handoff is source/configuration. Three files must be transferred separately: both merged `.safetensors` shards and `codec.pth`. The codec is a symlink on the source host; **follow it**, or your copy will contain a broken AWS-only path. Copy all export files, not just the three weights.

On the original AWS host, after unpacking this bundle there:

```bash
python -m scripts.copy_model_from_aws /path/to/new/attention_mlp_r16_early
```

That helper only copies into a new directory and follows the codec symlink. It does not stop/change the existing services. Then transfer that directory to your GPU machine using your normal SSH/SCP connection.

Alternatively, from your local machine, substituting your SSH connection details:

```bash
scp -r -i /path/to/ssh-key.pem \
  ubuntu@YOUR_AWS_HOST:/path/to/new/attention_mlp_r16_early ./models/
```

The exact original export path is recorded in `model-artifacts.json`. A direct `rsync -aL` of that directory also follows the codec link. No private SSH key is included. These private fine-tuned weights are not claimed to be publicly downloadable.

Verify locally:

```bash
python -m scripts.verify_model ./models/attention_mlp_r16_early
```

The optional raw adapter checkpoint path and its training config are recorded for provenance, but you do **not** need the raw adapter when using the exact merged export. Export/merge code and Fish Speech model/LoRA implementation are included under `reference_aws/` if your AI agent needs to inspect/recreate the merge. That route additionally requires the exact base weights and raw adapter; do not guess or retrain a different model.

## SGLang is managed by you

The known working runtime was SGLang **0.5.16**, SGLang-Omni **0.1.3**, Transformers **5.12.1**, Torch **2.11.0+cu130**, and an NVIDIA H100. The original Dockerfile and source snapshot are included for reference. This app's `requirements.txt` intentionally does not install that GPU stack.

`sglang.template.json` contains the working pipeline settings. Generate a local version:

```bash
python -m scripts.render_sglang_config \
  --model-directory /absolute/path/to/models/attention_mlp_r16_early
```

It retains the original H100 FA3/bfloat16 settings. Adjust the attention backend and memory fraction for your own hardware if necessary; do not copy the H100 settings blindly onto an unsupported GPU. The renderer accepts `--attention-backend` and `--memory-fraction`. This changes serving placement, not the selected weights.

If you choose to retain the original timing factory, make `tts/sglang_telemetry.py` importable by SGLang, for example by adding the absolute `tts/` directory to that process's `PYTHONPATH`. The timing module is optional for basic speech and does not record transcript contents.

Illustrative command for your already-installed CLI (verify signatures against your installed SGLang-Omni version):

```bash
PYTHONPATH=/absolute/path/to/aljawab-local/tts \
sgl-omni serve \
  --model-path /absolute/path/to/models/attention_mlp_r16_early \
  --config /absolute/path/to/aljawab-local/tts/sglang.local.json \
  --host 127.0.0.1 --port 3002
```

The engine must advertise this exact absolute path through `GET /v1/models`. Put that value in `.env` as `SGLANG_MODEL_PATH`. The compatibility gateway verifies all required artifact hashes by default and refuses a different advertised model. Keep that check enabled unless you have independently verified the exact artifacts and understand the change.

## Request/stream contracts

The app connects to port **3000**:

- `GET /api/status`: ready status, loaded model, references, bounded jobs.
- `GET /api/stream/references`: available reference IDs.
- `POST /api/upload`: raw reference bytes, prepared by ffmpeg to mono 44.1 kHz PCM16.
- `GET /api/reference/{id}.wav`: reference audio for hash verification.
- `POST /api/generate`: non-streaming background job; `GET /api/audio/{id}.wav` returns the completed WAV.
- `POST /api/stream`: SSE accepted/started/audio/complete events. Audio events contain base64 raw **PCM16 little-endian, mono, 44.1 kHz**. HTTP chunk boundaries are not audio frame boundaries.

Both synthesis requests specify `model: attention_mlp_r16_early`, `reference_id`, the exact `reference_text`, `text`, seed forty-two, temperature/top-p 0.7, and a maximum generation length of 1024 tokens. The app's non-streaming/streaming methods validate model/checkpoint/reference identity.

The gateway forwards to your SGLang **POST `/v1/audio/speech`**, using:

```json
{
  "model": "/absolute/path/to/models/attention_mlp_r16_early",
  "input": "حيّاك الله، معك نورة.",
  "voice": "default",
  "references": [{"audio_path": "/absolute/path/to/aljawab-local/assets/voice/noura_reference.wav", "text": "THE_EXACT_TRANSCRIPT_FROM_reference.json"}],
  "response_format": "pcm",
  "stream": true,
  "temperature": 0.7,
  "top_p": 0.7,
  "top_k": 30,
  "max_new_tokens": 1024
}
```

For cached clips it requests `response_format: wav` and `stream: false`. The original engine does not advertise seed support, so the gateway does not invent seed forwarding. Noura is selected through the reference waveform/transcript, not the literal `voice: default` label. **The reference file path must exist inside the engine's filesystem/container**, with the same path, or use an equivalent supported reference input contract after verifying your installed implementation.

Source Noura audio, exact transcript, prepared audio hash, and ID are in `assets/voice/reference.json`. Both the original stereo and exact prepared mono recordings are bundled. Do not substitute another reference, infer a different transcript, or claim to have fine-tuned a separate Noura model.

## Historical sources

`reference_aws/training/jordanian_s2pro/` contains the original serving, queue/SSE, model export/validation, timing, Docker, and launcher code. Its `common.py` also imports utilities from the copied `talha_s2pro/` sibling and the copied `fish-speech-upstream/` model code. Original paths, multi-model selection, NVMe storage scripts, and campaign assumptions are retained as historical evidence; the local app uses the new single-model gateway instead. Do not run the historical launchers against a local machine without adapting/reviewing them.

The original Fish Audio research license and model README are under `base_model_reference/`; the Fish Speech code license is included with that code. Preserve upstream license terms. No upstream service account, token, or private credential is part of this bundle.
