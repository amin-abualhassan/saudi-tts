"""Finite-gradient checks, preserved early checkpoint and adapter-update proof."""
from pathlib import Path
import lightning as L
import torch
from common import atomic_json


class TrainingGuard(L.Callback):
    def __init__(self, early_step=200):
        self.early_step = early_step
        self.saved = False
        self.initial = None

    def on_train_start(self, trainer, module):
        trainable = {n: p for n, p in module.named_parameters() if p.requires_grad}
        if not trainable or any("lora_" not in n for n in trainable):
            raise RuntimeError("Expected only LoRA parameters to be trainable")
        self.initial = {n: p.detach().cpu().clone() for n, p in trainable.items()}
        self.start_step = trainer.global_step
        self.root = Path(trainer.default_root_dir)
        atomic_json(self.root / "trainable.json", {"tensors": len(trainable),
                    "parameters": sum(p.numel() for p in trainable.values()), "only_lora": True})

    def on_before_optimizer_step(self, trainer, module, optimizer):
        for name, p in module.named_parameters():
            if p.requires_grad and p.grad is None:
                raise RuntimeError(f"Trainable adapter has no gradient: {name}")
            if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all():
                raise RuntimeError(f"Nonfinite gradient: {name}")

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if trainer.global_step == self.early_step and not self.saved:
            trainer.save_checkpoint(self.root / "checkpoints/early.ckpt")
            self.saved = True

    def on_train_end(self, trainer, module):
        changed = 0
        for name, p in module.named_parameters():
            if name in self.initial:
                if not torch.isfinite(p).all():
                    raise RuntimeError(f"Nonfinite adapter: {name}")
                changed += int(not torch.equal(self.initial[name], p.detach().cpu()))
        if changed == 0 and trainer.global_step > self.start_step:
            raise RuntimeError("No adapter updated during this run")
        atomic_json(self.root / "adapter_updates.json", {"changed_tensors": changed,
                    "trainable_tensors": len(self.initial), "step": trainer.global_step})
