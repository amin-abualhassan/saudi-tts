# Source this file: source /opt/dlami/nvme/training/activate_fish.sh
export CONDA_REGISTER_ENVS=false
export CONDA_PKGS_DIRS=/opt/dlami/nvme/miniconda3/pkgs,/opt/dlami/nvme/conda/pkgs
export CONDA_ENVS_PATH=/opt/dlami/nvme/miniconda3/envs
export PIP_CACHE_DIR=/opt/dlami/nvme/pip_cache
source /opt/dlami/nvme/miniconda3/etc/profile.d/conda.sh
conda activate /opt/dlami/nvme/miniconda3/envs/fish-speech
