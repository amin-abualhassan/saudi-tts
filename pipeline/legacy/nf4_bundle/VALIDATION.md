# H100 validation: attention_mlp_r16_early NF4

Status: packed conversion, strict reload, real synthesis, WAV/PCM HTTP, and app identity integration passed. Runtime is **Fish Speech/bitsandbytes NF4 fallback, not SGLang**. SM89 laptop fit/latency and human listening quality are **UNVERIFIED**.

## Parent, storage and coverage

Both supplied BF16 shard hashes/sizes and the unchanged codec match. The exact Noura prepared WAV and transcript were used; its SHA-256 is `d3290b0274478217ddb09145e1e3c118b50eb23b76579d67ce093798bee4432f`. Parent revision is `392cfb1cf6824476e616df3d0459986efdd268a13991737fdbb8260d200a5cd4`; adapter provenance is `9dbf9c8b6ee71dc5c737e6a76f2166539e833a6aa8ab274fb192553714a294cf`.

The 3,117,086,440-byte packed safetensors file has SHA-256 `bd78f331a5a1534cb8bfecc6c0a024db7c0c5267a6c9da806a2dde1e18dd29af`. All 358 learned transformer tensors are covered. All 200 merged attention AND MLP weights identified by the parent export are directly quantized. The 180 Slow-AR and 20 Fast-AR attention/MLP linear layers, plus the Fast-AR output layer, use NF4 with block size 64 and double-quantized statistics. The tied text embedding/output, shared codebook embedding, Fast-AR embedding, and all norms remain BF16. Fast input projection is Identity for this 2560-dimensional model. No learned tensor is dropped.

The external unchanged codec is not included again. Its learned tensors load strictly; six obsolete deterministic checkpoint buffers are explicitly excluded and regenerated. Codec position/mask buffers are bounded to 4096 positions; transformer context/cache is 2048. Reference encoding is CPU FP32; default audio decoding is GPU BF16. The packed loader reads tensors individually through safetensors, checks every learned shape/key, rejects extra keys, and never reads parent BF16 shards.

SGLang v0.1.3 commit `91d4359fa06d1b60834250624116318ffa448e18` was inspected: quant_config is accepted but not forwarded to the Fish linear layers. The separately pinned Fish candidate is commit `fc4e1e24ff3b8d7d28fdd66e6789f23acb63c5bb`. See `reports/source-review.json`, `model/quantization.json`, `reports/finetune-coverage.json`, and the delivered source/patches.

## Actual quantized execution

A fresh process reloaded the packed derivative with `Params4bit._quantize` replaced by an exception: reload and generation passed. All 201 quantized modules are `bitsandbytes.nn.Linear4bit` with packed uint8 `Params4bit`, NF4 state, and nested quantization statistics. The profiler recorded `kgemm_4bit_inference_naive<__nv_bfloat16,128,16>` and `kDequantizeBlockwise` kernels. Prefill can dequantize individual matrices into BF16 workspace before GEMM; weights remain packed on disk and for decode. Peak measurements include that workspace. Complete module/kernel counts are in `reports/kernel-probe.json`.

Only stock PyPI wheels are used: Torch/Torchaudio 2.8.0 CUDA 12.8 and bitsandbytes 0.48.2, with a full pinned dependency lock and digest-pinned Python 3.10 container. No H100-built kernels or compiler cache is delivered. The [bitsandbytes 0.48.2 wheel matrix](https://huggingface.co/docs/bitsandbytes/v0.48.2/installation) lists SM89 for Linux CUDA 12.8 builds. This is a software support statement, not a successful laptop test.

## Paired samples

Use `listen.html` for side-by-side playback. Each pair uses seed 42, temperature 0.7, top_p 0.7, top_k 30, max_new_tokens 768, one request, context 2048, eager execution, compilation/CUDA graphs disabled, and the same reference. No test hit the generation limit. Both parent and derivative use this same Fish decoder/sampling/window policy; these are not SGLang-vs-Fish comparisons. The company-services answers are test text, not verified product claims.

| Case | BF16 audio s | NF4 audio s | BF16 elapsed s | NF4 elapsed s | NF4 first PCM s | BF16/NF4 ASR WER |
|---|---:|---:|---:|---:|---:|---:|
| ar_greeting | 5.57 | 6.18 | 22.05 | 22.79 | 9.11 | 0.0% / 0.0% |
| ar_services | 13.47 | 13.47 | 23.73 | 36.24 | 2.96 | 0.0% / 3.2% |
| en_greeting | 3.53 | 3.62 | 6.33 | 9.73 | 2.96 | 0.0% / 15.4% |
| en_services | 13.93 | 15.84 | 25.10 | 43.47 | 2.99 | 0.0% / 0.0% |

First Arabic requests include first-reference CPU encoding and kernel warmup; subsequent samples reuse reference codes. Production services remained resident. CPU ASR/dependency/image work overlapped parts of the paired-run wall-clock timing, so do not treat these runs as controlled hardware benchmarks. The isolated restored-container test below is the cleaner cold/warm runtime measurement.

Text for every case is recorded in both JSON reports and printed below:

- **ar_greeting**: حيّاك الله، معك نورة، المساعدة الصوتية للجواب. وش أقدر أساعدك فيه؟
- **ar_services**: أكيد، في الجواب نساعد الشركات في خدمة العملاء والرد على الاستفسارات وتنظيم المواعيد. نقدر نفهم احتياجك ونوضح لك الخيارات المناسبة، وإذا احتجت تفاصيل أكثر أحولك للفريق المختص. وش الخدمة اللي تهمك؟
- **en_greeting**: Hello, I'm Noura, Aljawab's voice assistant. How can I help you?
- **en_services**: At Aljawab, we help companies answer customer questions, organize appointments, and provide clear information about their services. Tell me what your company needs, and I can explain the available options or connect you with the right team. Which service would you like to discuss first?

## Quality observations and limits

Automatic multilingual Whisper large-v3-turbo CPU/int8 ASR recovered all four BF16 texts with zero normalized word error. NF4 recovered the Arabic greeting and longer English answer with zero normalized WER. Its Arabic services transcript changed the final `تهمك` to `تهمت` (1 of 31 normalized words); English greeting transcription rendered “Noura” as “Nora” and “Aljawab” as “Aljoab” (2 of 13 words). These may reflect ASR spelling and/or pronunciation differences; no human judgment of pronunciation is claimed. All eight clips are finite, non-silent, have no clipped samples at the chosen threshold, and have zero adjacent repeated trigrams in ASR output. There was no generation-limit truncation. These checks cannot rule out audible clicks, prosody changes, accent errors, or nonverbal artifacts.

Speaker embedding cosine to the exact Noura reference (WeSpeaker ResNet34-LM) is a diagnostic, not a calibrated voice-identity acceptance test:

| Case | BF16/reference | NF4/reference | BF16/NF4 |
|---|---:|---:|---:|
| ar_greeting | 0.637 | 0.626 | 0.779 |
| ar_services | 0.658 | 0.715 | 0.894 |
| en_greeting | 0.466 | 0.615 | 0.694 |
| en_services | 0.569 | 0.671 | 0.817 |

Human listening for Saudi accent, brand/name pronunciation, repetitions, artifacts, and Noura similarity remains required. NF4 is lossy; do not assume BF16 parity. The bounded decoder emits 16 new codec frames per chunk with 8 future and up to 128 past frames. It may create boundary differences from full-utterance codec decoding; both paired samples use that same policy. No loudness normalization or ending-volume recovery was applied.

## H100 memory and latency

These are H100 SM90 measurements, with existing production services left running. No allocator cap was used to simulate 8 GB. Device memory is sampled through nvidia-smi approximately every 0.5 s (0.25 s sleep plus query overhead); short peaks between samples can be missed. Allocated/reserved peaks use PyTorch counters. Host RAM is process RSS including file-backed pages, not complete Windows+WSL+Docker+app consumption. Cold means a fresh process/CUDA context; filesystem page cache was not flushed because production was preserved. Storage-cold laptop I/O is unmeasured.

| Configuration | Device baseline GiB | Whole-device peak GiB | Increment GiB | CUDA peak allocated GiB | CUDA peak reserved GiB | Host RSS peak GiB |
|---|---:|---:|---:|---:|---:|---:|
| BF16 complete pipeline | 52.814 | 64.076 | 11.262 | 10.046 | 10.582 | 1.962 |
| NF4 complete pipeline | 52.814 | 58.498 | 5.684 | 4.407 | 5.004 | 3.665 |
| NF4 CPU codec short probe | 52.814 | 56.910 | 4.096 | 3.315 | 3.428 | 3.796 |

The complete four-case NF4 run reached 5.684 GiB incremental whole-device use, below the 6 GiB engineering target in this test only. Warm first PCM was 2.96–2.99 s and RTF 2.69–2.74: slower than real time on H100. First cold-reference Arabic PCM was 9.11 s. This is a runnable memory-oriented fallback, not a low-latency replacement for the production SGLang service.

CPU codec probe: 3.62 s audio took 24.87 s including first-reference encoding (RTF 6.87), first PCM 9.73 s, peak incremental GPU 4.096 GiB, host RSS 3.796 GiB. It reduces GPU memory but raises CPU/RAM cost. It is a short cold probe on the H100 host CPU, not a laptop benchmark or a warm controlled comparison.

At the laptop’s recorded 6.4 GiB free VRAM, measured H100 usage leaves limited margin and does not prove SM89 fit. At its recorded 2 GB available host RAM, the measured process alone exceeds available RAM. Free host memory before attempting; Windows/WSL paging can make latency much worse. No equivalent 8 GB SM89 hardware was available. **Laptop memory fit and latency remain UNVERIFIED; no guarantee is given.**

## Protocol, identity and restore

Container HTTP checks passed: WAV duration 3.62 s, warm streaming first PCM 3.07 s, stream completion 10.11 s, PCM/WAV bytes identical for a fixed seed. First bytes arrived before the complete response. `reports/http/http-smoke.json` contains network arrival times. Runtime frame-level timestamps additionally prove first PCM preceded AR completion; this is actual incremental synthesis, not playback chunks from a completed waveform.

OpenAI-style `/v1/models` and `/v1/audio/speech` are supplied by `runtime/server.py`. Requests must use the explicit derivative ID, absolute Linux reference audio path and exact transcript. PCM is mono 44.1 kHz signed 16-bit little-endian. A loopback-published container and Windows-to-Linux reference-path adapter are provided. `integration/` deliberately verifies this derivative, updates app/job/cache identity, retains original hashes, rejects a tampered derivative manifest and rejects original cached clips. A real isolated patched gateway SSE request passed. Existing app cached assets must be regenerated; do not relabel BF16 audio.

Clean extracted-release container cold test PASSED. Only restored derivative/reference directories and the single unchanged codec file were mounted; no BF16 shards were accessible. Fresh launch-to-ready: 16.71 s; model/codec load: 5.07 s. Cold first WAV request: 16.51 s for 3.62 s audio. Subsequent real PCM stream first bytes: 3.14 s; total: 10.35 s. Peak incremental whole-device memory including cold load/reference/codec/cache/workspace: 5.123 GiB. Peak container host RSS: 3.619 GiB. PyTorch peak allocated/reserved: 4.216/4.445 GiB. Raw traces and exact mounts/commands are in `reports/clean-restore/`.

The standard-library restore tests reject checksum corruption, unsafe paths, links, Windows collisions, existing destinations, and corrupt parts without deleting inputs. See `reports/security-tests.log`. The final ZIP has an external transfer receipt with archive/part hashes. The final archive is independently extracted and rehashed before handoff.

The production containers were neither restarted nor modified. Validation containers use separate names/ports and are stopped after tests. No model hub/public upload was performed. No secrets, environment/cache dumps, parent BF16 shards, or locally compiled H100 artifacts are included.
