"""Étapes 7 et 8 : mesurer le modèle fine-tuné et sauvegarder tout ce qu'il faut pour l'analyse.

Pour le tas choisi (val, ou test UNE SEULE FOIS à la toute fin) :
  - accuracy et top-5, avec leur marge d'erreur ;
  - avec et sans TTA (on fait aussi lire l'image retournée et on moyenne les deux avis) ;
  - predictions_<tas>.csv : pour chaque image, la vraie espèce, la réponse, la confiance, le top-5 ;
  - recall_<tas>.csv : le score de chaque espèce ;
  - features_<tas>.npy : le vecteur de caractéristiques de chaque image (pour les quasi-doublons).

Le test est protégé : la première évaluation crée le fichier TEST_DEJA_UTILISE.txt
et le script refuse ensuite de recommencer (sauf --force), pour qu'on ne soit pas
tenté de « régler » le modèle sur le test.

python scripts/evaluate.py --ckpt outputs/finetune_bioclip2/best.pt --split val
python scripts/evaluate.py --ckpt outputs/finetune_bioclip2/best.pt --split test
"""

import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import BirdImages, device_and_amp, eval_transform, load_classifier


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true", help="réévaluer le test malgré tout")
    args = p.parse_args()

    out = args.ckpt.parent
    marker = out / "TEST_DEJA_UTILISE.txt"
    if args.split == "test" and not args.smoke:
        if marker.exists() and not args.force:
            raise SystemExit(f"Le test a déjà été utilisé ({marker.read_text().strip()}). "
                             "Les résultats sont dans predictions_test.csv.")
        marker.write_text(f"Test évalué le {datetime.datetime.now():%Y-%m-%d %H:%M} avec {args.ckpt}\n")

    device, amp = device_and_amp()
    model, ck = load_classifier(args.ckpt, device)
    classes = ck["classes"]
    bb = model.backbone
    ds = BirdImages(args.root, args.split, eval_transform(ck["img_size"], bb.mean, bb.std), args.smoke)
    dl = DataLoader(ds, args.batch, num_workers=args.workers, pin_memory=device.type == "cuda")

    probs, probs_tta, feats = [], [], []
    with torch.no_grad():
        for x, _, _ in dl:
            x = x.to(device, non_blocking=True)
            with amp:
                logits, f = model(x, return_features=True)
                logits_flip = model(torch.flip(x, dims=[3]))  # l'image vue dans un miroir
            p1, p2 = F.softmax(logits.float(), 1), F.softmax(logits_flip.float(), 1)
            probs.append(p1.cpu())
            probs_tta.append(((p1 + p2) / 2).cpu())
            feats.append(f.float().cpu())
    probs, probs_tta, feats = torch.cat(probs), torch.cat(probs_tta), torch.cat(feats)
    y = ds.labels

    n = len(y)
    print(f"Modèle : {ck['model']} (epoch {ck['epoch']}, poids {ck['weights']}) | tas : {args.split} ({n} images)")
    for label, pr in (("sans TTA", probs), ("avec TTA", probs_tta)):
        acc = (pr.argmax(1) == y).float().mean().item()
        top5 = (pr.topk(5, 1).indices == y[:, None]).any(1).float().mean().item()
        margin = 1.96 * (acc * (1 - acc) / n) ** 0.5
        print(f"  {label} : accuracy {acc:.2%} (± {margin:.2%})   top-5 {top5:.2%}   erreurs : {int(round((1 - acc) * n))}")

    pr = probs_tta
    conf, pred = pr.max(1)
    top5 = pr.topk(5, 1).indices
    df = pd.DataFrame({
        "chemin": ds.meta.chemin,
        "vraie": [classes[i] for i in y],
        "predite": [classes[i] for i in pred],
        "correct": (pred == y).numpy(),
        "confiance": conf.numpy().round(4),
        "top5": [" | ".join(classes[j] for j in row) for row in top5.tolist()],
    })
    df.to_csv(out / f"predictions_{args.split}.csv", index=False)
    df.groupby("vraie")["correct"].mean().rename("recall").sort_values().to_csv(out / f"recall_{args.split}.csv")
    np.save(out / f"features_{args.split}.npy", feats.half().numpy())
    (out / f"scores_{args.split}.json").write_text(json.dumps({
        "accuracy": float((pred == y).float().mean()), "n": n, "tta": True}, indent=1))
    print(f"Fichiers écrits dans {out}")


if __name__ == "__main__":
    main()
