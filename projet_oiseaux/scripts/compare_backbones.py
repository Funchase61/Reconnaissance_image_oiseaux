"""Étapes 4 et 5 : comparer plusieurs backbones pré-entraînés par « linear probe ».

Pour chaque backbone :
  1. on le GÈLE (on ne touche à aucun de ses poids) ;
  2. on lui fait lire toutes les images une fois, et on garde le vecteur de
     caractéristiques qu'il produit pour chacune (feature extraction) ;
  3. on entraîne seulement une couche linéaire sur ces vecteurs (linear probe).

Le score de validation dit quel backbone « comprend » le mieux les oiseaux. Le
meilleur sera fine-tuné (finetune.py), en partant de la couche linéaire apprise ici.

Les vecteurs du test sont aussi calculés et sauvegardés, mais AUCUN score n'est
calculé sur le test : ils serviront seulement à chercher les quasi-doublons
entre train et test pendant l'analyse.

Sur RunPod :   python scripts/compare_backbones.py
Test sur PC :  python scripts/compare_backbones.py --smoke --models resnet18 smoke_vit smoke_clip
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import BirdImages, device_and_amp, eval_transform, load_backbone


@torch.no_grad()
def extract(backbone, loader, device, amp):
    feats = []
    for x, _, _ in loader:
        with amp:
            feats.append(backbone(x.to(device, non_blocking=True)).float().cpu())
    return torch.cat(feats)


def train_probe(Xtr, ytr, Xva, yva, n_classes, device, epochs, wd):
    """Régression logistique (une couche linéaire + softmax) entraînée sur GPU."""
    mean, std = Xtr.mean(0), Xtr.std(0) + 1e-6  # on centre-réduit chaque dimension
    Xtr_n, Xva_n = ((Xtr - mean) / std).to(device), ((Xva - mean) / std).to(device)
    ytr, yva = ytr.to(device), yva.to(device)
    lin = torch.nn.Linear(Xtr.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=wd)
    steps_per_epoch = (len(Xtr_n) + 1023) // 1024
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * steps_per_epoch)
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr_n), device=device)
        for k in range(0, len(perm), 1024):
            idx = perm[k:k + 1024]
            loss = F.cross_entropy(lin(Xtr_n[idx]), ytr[idx], label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    with torch.no_grad():
        logits = lin(Xva_n)
        acc = (logits.argmax(1) == yva).float().mean().item()
        top5 = (logits.topk(5, 1).indices == yva[:, None]).any(1).float().mean().item()
    # On « replie » la normalisation dans les poids, pour que la couche s'applique
    # directement aux vecteurs bruts du backbone (utile pour démarrer le fine-tuning).
    W = lin.weight.detach().cpu() / std
    b = lin.bias.detach().cpu() - W @ mean
    return acc, top5, W, b


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--models", nargs="+", default=["resnet18", "dinov2_l", "eva02_l", "bioclip2"])
    p.add_argument("--out", type=Path, default=Path("outputs/compare"))
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--probe-epochs", type=int, default=40)
    p.add_argument("--smoke", action="store_true", help="mini-test sur le PC")
    args = p.parse_args()

    device, amp = device_and_amp()
    args.out.mkdir(parents=True, exist_ok=True)
    classes = json.loads((args.root / "classes.json").read_text(encoding="utf-8"))
    print(f"Appareil : {device}  ({torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU'})", flush=True)

    summary_path = args.out / "summary.csv"
    rows = pd.read_csv(summary_path).to_dict("records") if summary_path.exists() and not args.smoke else []
    done = {r["backbone"] for r in rows}

    for name in args.models:
        if name in done:
            print(f"\n{name} : déjà fait, on passe (supprimer summary.csv pour refaire)", flush=True)
            continue
        print(f"\n=== {name} ===", flush=True)
        t0 = time.time()
        bb = load_backbone(name, args.img_size).to(device).eval()
        n_params = sum(p.numel() for p in bb.parameters()) / 1e6
        tf = eval_transform(args.img_size, bb.mean, bb.std)
        feats = {}
        for split in ("train", "val", "test"):
            ds = BirdImages(args.root, split, tf, smoke=args.smoke)
            dl = DataLoader(ds, batch_size=args.batch, num_workers=args.workers, pin_memory=device.type == "cuda")
            t1 = time.time()
            feats[split] = (extract(bb, dl, device, amp), ds.labels)
            print(f"  {split:5s} : {len(ds):6d} images -> vecteurs de taille {feats[split][0].shape[1]}"
                  f"  ({time.time() - t1:.0f} s)", flush=True)
            np.save(args.out / f"features_{name}_{split}.npy", feats[split][0].half().numpy())
        del bb
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # On essaie 3 forces de régularisation et on garde la meilleure SUR LA VALIDATION.
        (Xtr, ytr), (Xva, yva) = feats["train"], feats["val"]
        best = None
        for wd in (1e-4, 1e-2, 1e-1):
            acc, top5, W, b = train_probe(Xtr, ytr, Xva, yva, len(classes), device, args.probe_epochs, wd)
            print(f"  linear probe (weight decay {wd:g}) : val acc {acc:.2%}  top-5 {top5:.2%}", flush=True)
            if best is None or acc > best[0]:
                best = (acc, top5, W, b, wd)
        acc, top5, W, b, wd = best
        torch.save({"weight": W, "bias": b, "backbone": name}, args.out / f"probe_{name}.pt")
        rows.append({"backbone": name, "params_M": round(n_params, 1), "val_acc": round(acc, 4),
                     "val_top5": round(top5, 4), "weight_decay": wd, "minutes": round((time.time() - t0) / 60, 1)})
        pd.DataFrame(rows).to_csv(summary_path, index=False)

    table = pd.DataFrame(rows).sort_values("val_acc", ascending=False)
    print("\nRésultat (trié du meilleur au moins bon) :\n" + table.to_string(index=False), flush=True)
    print(f"\nMeilleur backbone : {table.iloc[0].backbone}", flush=True)


if __name__ == "__main__":
    main()
