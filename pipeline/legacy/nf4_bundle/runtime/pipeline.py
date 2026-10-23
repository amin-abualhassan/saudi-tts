"""Serial eager TTS, CPU reference encoding, bounded incremental codec windows."""
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.nn.attention import SDPBackend, sdpa_kernel

from fish_speech.models.text2semantic.inference import generate_long, decode_one_token_ar, encode_audio
from weights import load, audit, sha, PARENT

class Pipeline:
    def __init__(self, model_path, codec_path, parent=False, codec_device='cuda'):
        torch.set_num_threads(8)
        self.start = time.perf_counter()
        self.model = load(model_path, parent=parent)
        self.model_path = str(Path(model_path).resolve())
        self.quantized_modules = audit(self.model) if not parent else {}
        codec_path = Path(codec_path)
        if codec_path.stat().st_size != PARENT['codec.pth'][0] or sha(codec_path) != PARENT['codec.pth'][1]:
            raise ValueError('Unchanged codec SHA-256 mismatch')
        import fish_speech
        cfg = OmegaConf.load(str(Path(next(iter(fish_speech.__path__)))/'configs/modded_dac_vq.yaml'))
        with torch.device('meta'):
            self.codec = instantiate(cfg)
        state = torch.load(codec_path, map_location='cpu', mmap=True, weights_only=True)
        if 'state_dict' in state:
            state = state['state_dict']
        if any(k.startswith('generator.') for k in state):
            state = {k[len('generator.'):]:v for k,v in state.items() if k.startswith('generator.')}
        # Older codec checkpoints persist these deterministic buffers. They are
        # not learned tensors; reject every other unexpected key. Bound the
        # regenerated buffers for <=30 s references and <=160-frame decode.
        from fish_speech.models.dac.modded_dac import Transformer, precompute_freqs_cis
        allowed_derived = set()
        for name, module in self.codec.named_modules():
            if isinstance(module, Transformer):
                for field in ('freqs_cis', 'causal_mask'):
                    allowed_derived.add(name+'.'+field)
                module.freqs_cis = precompute_freqs_cis(4096, module.config.head_dim, module.config.rope_base)
                module.causal_mask = torch.ones(4096,4096,dtype=torch.bool).tril_()
        state = {k:v for k,v in state.items() if k not in allowed_derived}
        self.codec.load_state_dict(state, strict=True, assign=True)
        self.codec.eval()
        # CPU FP32 encoding avoids the long-reference GPU activation peak.
        self.codec.to(device='cpu', dtype=torch.float32)
        self.codec_device = codec_device
        self.reference_cache = None
        self.load_seconds = time.perf_counter() - self.start

    @torch.inference_mode()
    def reference(self, path, text):
        path = Path(path)
        if not path.is_absolute() or not path.is_file():
            raise ValueError('Reference audio_path must be an existing absolute Linux path')
        import soundfile as sf
        info = sf.info(path)
        if not 3 <= info.duration <= 30:
            raise ValueError('Reference must be between 3 and 30 seconds')
        key = (sha(path), text)
        if self.reference_cache and self.reference_cache[0] == key:
            return self.reference_cache[1]
        self.codec.to(device='cpu', dtype=torch.float32)
        codes = encode_audio(path, self.codec, 'cpu').cpu()
        if codes.shape[1] > 650:
            raise ValueError('Reference exceeds 30 seconds')
        self.reference_cache = (key, codes)
        self.codec.to(device=self.codec_device,
                      dtype=torch.bfloat16 if self.codec_device == 'cuda' else torch.float32)
        gc.collect()
        torch.cuda.empty_cache()
        return codes

    @torch.inference_mode()
    def synthesize(self, text, reference_path, reference_text, emit, *, seed=42,
                   temperature=0.7, top_p=0.7, top_k=30, max_new_tokens=768):
        if not 1 <= len(text) <= 1000 or not reference_text or not 1 <= max_new_tokens <= 1024:
            raise ValueError('Text, transcript or generation bound invalid')
        started = time.perf_counter()
        prompt = self.reference(reference_path, reference_text)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        frames = []
        emitted = 0
        first_frame = None
        first_pcm = None
        chunks = []
        frame_length = self.codec.frame_length
        # 16 new frames per chunk, 8 future frames, 128 past frames. The decoder
        # operates while AR generation is still active; no full-audio buffering.
        def decode(final=False):
            nonlocal emitted, first_pcm
            end = len(frames) if final else len(frames) - 8
            if end <= emitted:
                return
            left = max(0, emitted - 128)
            # RVQ downsamples by 4; align each decode window to a multiple of 4.
            left = left // 4 * 4
            codes = torch.cat(frames[left:], dim=1).to(self.codec_device)
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                wav = self.codec.from_indices(codes[None])[0, 0]
            raw = wav[(emitted-left)*frame_length:(end-left)*frame_length].float().cpu().numpy()
            if not np.isfinite(raw).all() or not raw.size:
                raise ValueError('Invalid incremental codec output')
            pcm = (np.clip(raw, -1, 1)*32767).astype('<i2').tobytes()
            if first_pcm is None:
                first_pcm = time.perf_counter()-started
            chunks.append(dict(at_seconds=time.perf_counter()-started, bytes=len(pcm), frames=end-emitted))
            emit(pcm)
            emitted = end

        def callback(token):
            nonlocal first_frame
            semantic = int(token[0, 0])
            if semantic == self.model.tokenizer.get_token_id('<|im_end|>'):
                return
            if first_frame is None:
                first_frame = time.perf_counter()-started
            frames.append(token[1:].detach().cpu().clone())
            if len(frames)-emitted >= 24:
                decode()

        self.model._audio_frame_callback = callback
        try:
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                for response in generate_long(model=self.model, device='cuda', decode_one_token=decode_one_token_ar,
                    text=text, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
                    top_k=top_k, compile=False, iterative_prompt=False, prompt_text=[reference_text],
                    prompt_tokens=[prompt]):
                    pass
            generation_done = time.perf_counter()-started
            decode(final=True)
        finally:
            self.model._audio_frame_callback = None
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        duration = emitted*frame_length/44100
        if not duration:
            raise ValueError('No audio generated')
        return dict(elapsed_seconds=elapsed, duration_seconds=duration, rtf=elapsed/duration,
                    first_frame_seconds=first_frame, first_pcm_seconds=first_pcm,
                    ar_done_seconds=generation_done, chunks=chunks,
                    incremental_before_ar_done=bool(first_pcm and first_pcm < generation_done),
                    frames=emitted, hit_token_limit=len(frames)>=max_new_tokens-1,
                    cuda_allocated=torch.cuda.memory_allocated(), cuda_reserved=torch.cuda.memory_reserved(),
                    cuda_peak_allocated=torch.cuda.max_memory_allocated(), cuda_peak_reserved=torch.cuda.max_memory_reserved())
