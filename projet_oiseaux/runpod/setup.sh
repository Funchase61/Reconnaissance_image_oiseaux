#!/bin/bash
# Installation sur le pod RunPod (à lancer UNE fois, depuis le dossier projet_oiseaux) :
#     bash runpod/setup.sh
# Durée : environ 10 minutes.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=/workspace/hf_cache   # les modèles téléchargés restent sur le disque du pod

echo "== 1/3 Bibliothèques Python"
# Ubuntu 24.04 protège le Python du système ; sur les images RunPod, PyTorch y est
# installé, donc on autorise pip à y ajouter nos bibliothèques.
export PIP_BREAK_SYSTEM_PACKAGES=1
pip install -q timm==1.0.31 open_clip_torch==3.3.0 huggingface_hub pandas pyarrow

echo "== 2/3 Téléchargement du dataset depuis Hugging Face (environ 2 Go)"
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("yashikota/birds-525-species-image-classification", repo_type="dataset",
                  local_dir="/workspace/hf_birds", allow_patterns=["data/*"])
PY

echo "== 3/3 Rangement des images en dossiers (même script que sur le PC)"
python scripts/prepare_data.py --src /workspace/hf_birds/data --out .

python -c "import torch; print('Carte graphique :', torch.cuda.get_device_name(), '| PyTorch', torch.__version__)"
echo "Installation terminée. Lancer ensuite : nohup bash runpod/run_all.sh > outputs/run_all.log 2>&1 &"
