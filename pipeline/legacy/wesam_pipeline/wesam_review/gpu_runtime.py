"""Explicit device selection and batched diarization inference."""


def resolve_device(requested, torch):
    if requested == 'cpu':
        return 'cpu'
    if not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA is unavailable in this process. Run from a host terminal with '
            'H100 access; use --device cpu only for an intentional CPU run.'
        )
    # Force initialization now, before loading models or starting an episode.
    torch.empty(1, device='cuda')
    return 'cuda'


def configure_pipeline(pipeline, torch, device, segmentation_batch_size,
                       embedding_batch_size):
    if segmentation_batch_size < 1 or embedding_batch_size < 1:
        raise ValueError('Batch sizes must be positive')
    pipeline.to(torch.device(device))
    pipeline.segmentation_batch_size = segmentation_batch_size
    pipeline.embedding_batch_size = embedding_batch_size
    return pipeline
