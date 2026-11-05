#!/bin/bash
# Rassemble les résultats à télécharger dans un seul fichier : resultats_runpod.tar
set -euo pipefail
cd "$(dirname "$0")/.."
BEST=$(python -c "import pandas as pd; print(pd.read_csv('outputs/compare/summary.csv').sort_values('val_acc').iloc[-1].backbone)")
tar --exclude='last.pt*' -cf resultats_runpod.tar \
    outputs/run_all.log outputs/compare/summary.csv outputs/compare/probe_*.pt \
    outputs/compare/features_"$BEST"_*.npy \
    "outputs/finetune_$BEST"
ls -lh resultats_runpod.tar
echo "À télécharger depuis Jupyter (clic droit > Download). Ensuite, on peut supprimer le pod."
