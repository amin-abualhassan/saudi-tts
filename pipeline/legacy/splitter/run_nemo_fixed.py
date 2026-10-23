import os
from omegaconf import OmegaConf
from nemo.collections.asr.models.msdd_models import NeuralDiarizer

# 1. Load the official config
config_path = "/workspace/nemo/examples/speaker_tasks/diarization/conf/inference/diar_infer_telephonic.yaml"
cfg = OmegaConf.load(config_path)

# ... (keep your existing config overrides here) ...

cfg.diarizer.manifest_filepath = "/workspace/Splitter/input_manifest.json"
cfg.diarizer.out_dir = "/workspace/Splitter/output"
cfg.diarizer.num_speakers = 2

# FORCE num_workers to 0 to avoid Shared Memory Bus Errors
cfg.diarizer.vad.parameters.num_workers = 0 
# If you run into issues later in the clustering stage, you may need:
# cfg.diarizer.clustering.parameters.num_workers = 0

# 2. Initialize and Run
try:
    print("Initializing NeuralDiarizer...")
    msdd_model = NeuralDiarizer(cfg=cfg)
    print("Diarizer initialized successfully!")
    msdd_model.diarize()
    print("Diarization complete.")
except Exception as e:
    print(f"FAILED: {e}")
