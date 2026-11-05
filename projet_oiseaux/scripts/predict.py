"""Livrable : donner l'espèce d'oiseaux sur n'importe quelle photo.

python scripts/predict.py --ckpt outputs/finetune_bioclip2/best.pt photo1.jpg photo2.jpg
python scripts/predict.py --ckpt ... mon_dossier_de_photos/

Fonctionne aussi sans carte graphique (environ 1 seconde par photo sur un portable).
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from common import device_and_amp, eval_transform, load_classifier

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def predict(model, ck, image: Image.Image, k: int = 5):
    """Renvoie les k espèces les plus probables : [(espèce, probabilité), ...]."""
    tf = eval_transform(ck["img_size"], model.backbone.mean, model.backbone.std)
    x = tf(image.convert("RGB"))[None].to(next(model.parameters()).device)
    with torch.no_grad():
        # Sans TTA : c'est le choix figé sur la validation (99,43 % sans, 99,35 % avec).
        p = F.softmax(model(x).float(), 1)
    top = p[0].topk(k)
    return [(ck["classes"][i], v) for v, i in zip(top.values.tolist(), top.indices.tolist())]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("images", nargs="+", type=Path)
    args = p.parse_args()

    device, _ = device_and_amp()
    model, ck = load_classifier(args.ckpt, device)
    files = []
    for path in args.images:
        files += sorted(f for f in path.rglob("*") if f.suffix.lower() in EXTENSIONS) if path.is_dir() else [path]
    for f in files:
        with Image.open(f) as im:
            top = predict(model, ck, im)
        print(f"\n{f.name}")
        for name, prob in top:
            print(f"  {prob:6.1%}  {name}")
        if top[0][1] < 0.5:
            print("  (peu sûr : l'oiseau n'est peut-être pas une des 525 espèces, ou la photo est difficile)")


if __name__ == "__main__":
    main()
