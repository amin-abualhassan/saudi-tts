"""CPU audit of the actual selected adapters versus their native HF exports."""
import gc
from pathlib import Path
import sys

from common import HERE, UPSTREAM, atomic_json, read_json
from sglang_export import prepare_exports
from voice_server import selected_bundle


def main():
    import torch
    sys.path.insert(0, str(UPSTREAM))
    from fish_speech.models.text2semantic.llama import DualARTransformer
    from voice_worker import load_adapter_model
    torch.set_num_threads(8)
    bundle = selected_bundle()
    paths = prepare_exports(bundle)
    report = {"device": "CPU", "dtype": "bfloat16", "models": {},
              "sglang_gpu_tested": False, "note": "Parity with native Fish model, not SGLang sampling"}
    for name, entry in bundle["models"].items():
        print(f"Checking actual merged weights and slow/fast logits: {name}", flush=True)
        adapter = load_adapter_model(entry, "cpu", torch.bfloat16)
        merged = DualARTransformer.from_pretrained(paths[name], load_weights=True,
                                                   max_length=4096).to(torch.bfloat16).eval()
        expected = {n: p for n, p in adapter.state_dict().items() if "lora_" not in n}
        actual = merged.state_dict()
        if set(expected) != set(actual):
            raise ValueError("Merged state keys differ")
        for key in expected:
            torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
        torch.manual_seed(42)
        tokens = torch.randint(0, 4096, (1, 11, 4))
        tokens[:, 0] += adapter.config.semantic_begin_id
        with torch.inference_mode():
            a, b = adapter(tokens, labels=tokens), merged(tokens, labels=tokens)
        torch.testing.assert_close(a.token_logits, b.token_logits, rtol=0, atol=0)
        torch.testing.assert_close(a.codebook_logits, b.codebook_logits, rtol=0, atol=0)
        receipt = read_json(Path(paths[name]) / "export.json")
        report["models"][name] = {"export": paths[name], "checkpoint_sha256": entry["checkpoint_sha256"],
            "global_step": receipt["global_step"], "adapter_tensors": receipt["adapter_tensors"],
            "merged_weights": len(receipt["merged_weights"]), "verified_state_tensors": len(actual),
            "weights_exact": True, "slow_logits_exact": True, "fast_logits_exact": True,
            "sample_input": {"seed": 42, "shape": [1, 11, 4], "semantic_frames": 4}}
        atomic_json(HERE / "reports/sglang_export_validation.json", report)
        del adapter, merged, expected, actual, a, b
        gc.collect()
    print("Both actual model exports passed exact CPU weight and logit parity", flush=True)


if __name__ == "__main__":
    main()
