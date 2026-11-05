#!/bin/bash
# Toute la partie GPU d'un coup (depuis le dossier projet_oiseaux) :
#     mkdir -p outputs && nohup bash runpod/run_all.sh > outputs/run_all.log 2>&1 &
#     tail -f outputs/run_all.log        (pour suivre ; Ctrl+C arrête seulement l'affichage)
# Si le pod s'arrête en route, relancer la même commande : chaque étape reprend où elle en était.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=/workspace/hf_cache
export PYTHONUNBUFFERED=1
WORKERS=$(( $(nproc) > 12 ? 12 : $(nproc) ))

echo "== Étapes 4-5 : comparaison des backbones (linear probe) — $(date)"
python scripts/compare_backbones.py --workers "$WORKERS"

BEST=$(python -c "import pandas as pd; print(pd.read_csv('outputs/compare/summary.csv').sort_values('val_acc').iloc[-1].backbone)")
echo "== Étape 6 : fine-tuning de $BEST — $(date)"
python scripts/finetune.py --model "$BEST" --workers "$WORKERS"

echo "== Étape 7 : évaluation sur la validation — $(date)"
python scripts/evaluate.py --ckpt "outputs/finetune_$BEST/best.pt" --split val --workers "$WORKERS"

echo "== Terminé — $(date)"
echo "Le test n'a PAS été utilisé. Quand tout est validé :"
echo "    python scripts/evaluate.py --ckpt outputs/finetune_$BEST/best.pt --split test"
echo "puis : bash runpod/pack_results.sh"
