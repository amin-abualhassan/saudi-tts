"""Worker-local checkpointing compatible with LoRA and frozen embeddings."""
from contextlib import contextmanager


@contextmanager
def non_reentrant_checkpointing():
    import fish_speech.models.text2semantic.llama as llama
    from torch.utils.checkpoint import checkpoint

    original = llama.checkpoint

    def checkpoint_layer(function, *args, **kwargs):
        # The pinned slow and fast forward paths explicitly pass True. That
        # variant drops adapter gradients if the frozen input has no grad_fn.
        kwargs["use_reentrant"] = False
        return checkpoint(function, *args, **kwargs)

    llama.checkpoint = checkpoint_layer
    try:
        yield
    finally:
        llama.checkpoint = original
