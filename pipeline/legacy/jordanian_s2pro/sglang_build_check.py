"""Build-time interpreter, version and codec smoke checks, without GPU or weights."""
import importlib.metadata as metadata
import os
from pathlib import Path
import shutil
import sys


def main():
    import audiotools
    from audiotools.ml import BaseModel
    import dac
    from dac.model.base import CodecMixin
    from dac.nn.quantize import ResidualVectorQuantize
    import google.protobuf
    import torch
    import torchaudio

    environment = os.environ["VIRTUAL_ENV"]
    if sys.prefix != environment or sys.prefix == sys.base_prefix:
        raise RuntimeError("Build check is not running in the container virtual environment")
    entrypoint = shutil.which("sgl-omni")
    if not entrypoint or not Path(entrypoint).is_relative_to(Path(environment) / "bin"):
        raise RuntimeError("SGLang entry point is outside the virtual environment")
    expected = {"torch": "2.11.0", "torchaudio": "2.11.0", "sglang": "0.5.16",
                "sglang-omni": "0.1.3", "transformers": "5.12.1",
                "descript-audiotools": "0.7.2", "descript-audio-codec": "1.0.0"}
    for package, version in expected.items():
        actual = metadata.version(package).split("+")[0]
        if actual != version:
            raise RuntimeError(f"Unexpected {package}: {actual}; expected {version}")
    if not google.protobuf.__version__.startswith("6."):
        raise RuntimeError("The modern protobuf override was not applied")
    # Import the codec's actual utility/model/quantizer interfaces. This also
    # exercises audiotools' TensorBoard imports under the installed protobuf.
    assert BaseModel and CodecMixin and ResidualVectorQuantize
    torch.set_num_threads(2)
    torch.manual_seed(42)
    signal = audiotools.AudioSignal(torch.randn(1, 1, 256) * .05, sample_rate=44100)
    model = dac.DAC(encoder_dim=8, encoder_rates=[2, 2], latent_dim=16,
        decoder_dim=16, decoder_rates=[2, 2], n_codebooks=2,
        codebook_size=16, codebook_dim=4, sample_rate=44100).eval()
    with torch.inference_mode():
        audio = model(signal.audio_data)["audio"]
    if audio.shape != signal.audio_data.shape or not torch.isfinite(audio).all():
        raise RuntimeError("Descript codec CPU forward check failed")
    print("Container interpreter, pinned versions, protobuf and codec CPU checks passed", flush=True)


if __name__ == "__main__":
    main()
