import numpy as np, torch
print(f"numpy : {np.__version__}")
print(f"torch : {torch.__version__}")
print(f"CUDA  : {torch.cuda.is_available()}")
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
print("NeuralDiarizer: OK")