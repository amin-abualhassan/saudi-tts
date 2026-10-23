from omegaconf import OmegaConf
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
import os

# 1. Load the config file
cfg = OmegaConf.load("/workspace/Splitter/diar_config.yaml")

# 2. Initialize the Diarizer
# This will now work because 'cfg' contains all the necessary nested defaults
msdd_model = NeuralDiarizer(cfg=cfg)

# 3. Run
msdd_model.diarize()
