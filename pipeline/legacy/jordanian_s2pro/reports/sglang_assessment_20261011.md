# SGLang assessment for the selected Jordanian S2 Pro models

Assessed 2026-10-11. This is a source/configuration review, not a GPU benchmark.
The restored voice server and launchers were not changed or restarted.

SGLang-Omni is a suitable candidate for an isolated inference experiment. Its
[S2-Pro cookbook](https://github.com/sgl-project/sglang-omni/blob/main/docs/cookbook/fishaudio_s2_pro.md)
documents a dedicated speech endpoint, reference-audio cloning and streaming.
Our pinned Fish checkout also recommends its SGLang acceleration path in
`docs/en/inference.md`.

## What could improve

The [Fish S2 paper, section 5](https://arxiv.org/html/2603.08823v1#S5), describes
paged KV cache, continuous batching, CUDA graphs and audio-aware prefix reuse.
It reports RTF 0.195 and approximately 100 ms time to first audio on an H200 in
its production setup. These are published results, not predictions for this
H100 or this public engine version. Our previous live baseline was approximately
12 generated frames/second and RTF 1.84. First playback latency and complete
generation time need separate measurements.

The current [S2-Pro engine builder](https://github.com/sgl-project/sglang-omni/blob/main/sglang_omni/models/fishaudio_s2_pro/engine_builder.py)
selects `fa3` for SM90 (H100), uses BF16 and a 4096-token context, and enables
codebook Torch compilation on SM90. Its CUDA defaults allow 64 running requests
and a static memory fraction of 0.85. Initial experiments should explicitly
bound concurrency, graph batch sizes and memory use, especially when sharing
the GPU with the existing server. Startup still involves warmup/graph work.

## Compatibility findings

The local original base contains the expected nested text/audio configuration:
36 slow layers, four fast layers, dimension 2560 and ten codebooks. Its 358 native
safetensors keys include `text_model.model.*` and `audio_decoder.*`. This matches
the layout expected by the inspected
[SGLang model loader](https://github.com/sgl-project/sglang-omni/blob/main/sglang_omni/models/fishaudio_s2_pro/sglang_model.py).
This establishes structural compatibility; actual loading remains untested.

Serve both listening-selected adapters as separately merged native checkpoints:
`fast_r32_final` and `attention_mlp_r16_early`. The current local `merge.py` calls
an exporter that writes legacy `dual_ar` configuration and `model.pth`; it should
not be assumed compatible with this backend. An exporter must retain the native
HF configuration, tokenizer and safetensors key layout, merge every trained LoRA
target into its matching original tensor, and retain the original codec. Rank-32
fast embedding/output adapters must be included along with fast attention/MLP.
No further training is needed for the proposed export.

The inspected SGLang implementation chooses fast codebooks with argmax. Our
current decoder samples these codebooks using temperature/top-p/top-k. Semantic
repetition handling also differs. Matching weights and seed therefore does not
ensure identical output or unchanged Jordanian quality. Listening tests are
required before adoption.

## Recommended experiment

1. Pin a verified SGLang-Omni release/commit and compatible CUDA stack in a separate
   container/environment. The [installation guide](https://github.com/sgl-project/sglang-omni/blob/main/docs/get_started/installation.md)
   recommends Docker for NVIDIA. The inspected main dependency file differs
   substantially from the existing Torch 2.8.0 / Transformers 4.57.3 environment;
   neither SGLang package is installed there.
2. Export and verify `fast_r32_final` first in a new directory, preserving source
   checkpoint/base hashes. Compare merged weights/logits against the current
   model before testing the new backend. Then repeat for the attention+MLP model.
3. Use an isolated API on 7861, BF16, context 4096, maximum one running request
   and graph batch size one initially. Check GPU headroom before loading; separate
   ports/environments do not isolate GPU memory. Schedule GPU timing without
   simultaneous production generation. Bound generation at the existing 1024
   tokens with temperature/top-p 0.7, top-k 30 and seed 42 where supported.
4. Compare the existing 15 cases per model across Talha, Wesam and the single
   speaker; add an uploaded voice and repeated-reference requests. Record cold
   startup, warm token speed, full-request RTF, first-audio latency, peak GPU
   memory and voice/accent/stability ratings. Score sampling differences explicitly.
5. Integrate the successful backend with the existing cloning UI after the
   comparison. Keep the original launcher available for immediate fallback.

Not performed: installing SGLang, merging/exporting weights, starting a GPU
engine, replacing port 7860, or measuring accelerated synthesis. The sandbox
cannot access CUDA or control the live host process.
