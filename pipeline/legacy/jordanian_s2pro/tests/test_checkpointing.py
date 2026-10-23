"""Exercise actual DualARTransformer backward with frozen input embeddings."""
from pathlib import Path
import sys
import unittest
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import UPSTREAM
sys.path.insert(0, str(UPSTREAM))
import torch
import fish_speech.models.text2semantic.llama as llama
from fish_speech.models.text2semantic.lora import LoraConfig, setup_lora
from training_compat import non_reentrant_checkpointing


class CheckpointingTests(unittest.TestCase):
    def tiny_model(self, targets):
        torch.manual_seed(7)
        config = llama.DualARModelArgs(vocab_size=64, n_layer=2, n_head=2,
                    n_local_heads=2, head_dim=16, dim=32, intermediate_size=64,
                    max_seq_len=32, codebook_size=16, num_codebooks=2,
                    semantic_begin_id=16, semantic_end_id=31,
                    n_fast_layer=2, fast_dim=32, fast_n_head=2,
                    fast_n_local_heads=2, fast_head_dim=16, fast_intermediate_size=64,
                    use_gradient_checkpointing=True)
        model = llama.DualARTransformer(config)
        setup_lora(model, LoraConfig(r=2, lora_alpha=2, target_modules=targets))
        model.train()
        return model

    def loss(self, model):
        inp = torch.tensor([[[17, 18, 19, 20], [1, 2, 3, 4], [2, 3, 4, 5]]])
        labels = torch.tensor([[[18, 19, 20, 21], [2, 3, 4, 5], [3, 4, 5, 6]]])
        result = model(inp, labels=labels)
        return (torch.nn.functional.cross_entropy(result.token_logits.reshape(-1, 64), labels[:, 0].reshape(-1))
                + torch.nn.functional.cross_entropy(result.codebook_logits.reshape(-1, 16),
                         labels[:, 1:].permute(0, 2, 1).reshape(-1)))

    def test_reproduces_original_attention_failure(self):
        model = self.tiny_model(["attention"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loss = self.loss(model)
        self.assertFalse(loss.requires_grad)
        with self.assertRaisesRegex(RuntimeError, "does not require grad"):
            loss.backward()

    def test_attention_and_mlp_adapters_receive_gradients_in_both_transformers(self):
        for targets in (["attention"], ["attention", "mlp"]):
            with self.subTest(targets=targets):
                model = self.tiny_model(targets)
                self.assertFalse(model.embeddings.weight.requires_grad)
                self.assertFalse(model.fast_embeddings.weight.requires_grad)
                before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
                with non_reentrant_checkpointing():
                    for _ in range(2):
                        optimizer.zero_grad(set_to_none=True)
                        loss = self.loss(model)
                        self.assertTrue(loss.requires_grad)
                        loss.backward()
                        for n, p in model.named_parameters():
                            if p.requires_grad:
                                self.assertIsNotNone(p.grad, n)
                                self.assertTrue(torch.isfinite(p.grad).all(), n)
                        optimizer.step()
                changed = {n for n, p in model.named_parameters() if n in before and not torch.equal(p, before[n])}
                self.assertTrue(any(n.startswith("layers.") for n in changed))
                self.assertTrue(any(n.startswith("fast_layers.") for n in changed))
                self.assertEqual(changed, set(before))

    def test_forward_values_match_and_runtime_override_is_restored(self):
        model = self.tiny_model(["attention"])
        model.config.use_gradient_checkpointing = False
        plain = self.loss(model).detach()
        model.config.use_gradient_checkpointing = True
        original = llama.checkpoint
        with non_reentrant_checkpointing():
            patched = self.loss(model).detach()
            torch.testing.assert_close(plain, patched)
        self.assertIs(llama.checkpoint, original)


if __name__ == "__main__":
    unittest.main()
