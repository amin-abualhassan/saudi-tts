from unittest import TestCase
from unittest.mock import Mock

from gpu_runtime import configure_pipeline, resolve_device
from stop_cpu import HERE, matches


class GpuRuntimeTests(TestCase):
    def test_unavailable_cuda_never_falls_back_to_cpu(self):
        torch = Mock()
        torch.cuda.is_available.return_value = False
        for requested in ('cuda', 'auto'):
            with self.assertRaisesRegex(RuntimeError, 'CUDA is unavailable'):
                resolve_device(requested, torch)
        torch.empty.assert_not_called()

    def test_explicit_cpu_is_supported(self):
        torch = Mock()
        self.assertEqual(resolve_device('cpu', torch), 'cpu')
        torch.cuda.is_available.assert_not_called()

    def test_cuda_initializes_before_work(self):
        torch = Mock()
        torch.cuda.is_available.return_value = True
        self.assertEqual(resolve_device('cuda', torch), 'cuda')
        torch.empty.assert_called_once_with(1, device='cuda')

    def test_models_and_batches_use_requested_configuration(self):
        torch, pipeline = Mock(), Mock()
        configure_pipeline(pipeline, torch, 'cuda', 64, 128)
        pipeline.to.assert_called_once_with(torch.device.return_value)
        torch.device.assert_called_once_with('cuda')
        self.assertEqual(pipeline.segmentation_batch_size, 64)
        self.assertEqual(pipeline.embedding_batch_size, 128)

    def test_stop_matches_only_original_cpu_worker(self):
        argv = ['python', str(HERE / 'prepare_dataset.py'), '--device', 'cpu']
        self.assertTrue(matches(argv, HERE))
        self.assertFalse(matches(argv + ['--out', '/tmp/another-dataset'], HERE))
        self.assertFalse(matches(['python', str(HERE / 'prepare_dataset.py'),
                                  '--device', 'cuda'], HERE))
        self.assertFalse(matches(['python', '/tmp/prepare_dataset.py',
                                  '--device', 'cpu'], HERE))
