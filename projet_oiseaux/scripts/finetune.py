"""Étape 6 : fine-tuning complet du meilleur backbone.

Cette fois, on modifie TOUS les poids du réseau, très doucement :
  - départ : le backbone pré-entraîné + la couche linéaire apprise par le linear probe
    (on ne part pas d'une tête au hasard, qui « secouerait » le réseau au début) ;
  - AdamW, avec un pas d'apprentissage plus petit pour les couches basses
    (layer-wise lr decay) : elles savent déjà voir les bords et les textures ;
  - 1 epoch d'échauffement (warmup), puis le pas diminue en cosinus ;
  - régularisation : label smoothing, drop path, weight decay, augmentation douce ;
  - EMA : une copie « moyennée » des poids, souvent un peu meilleure et plus stable ;
  - calcul en bfloat16 sur la carte graphique.

À chaque epoch : score de validation, sauvegarde de last.pt (pour reprendre si le
pod s'arrête) et de best.pt (le meilleur modèle selon la validation).
Le test n'est JAMAIS utilisé ici.

Sur RunPod :   python scripts/finetune.py --model bioclip2
Test sur PC :  python scripts/finetune.py --model smoke_vit --smoke --epochs 2 --batch 8
"""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import (BirdImages, Classifier, count_blocks, device_and_amp, eval_transform,
                    layer_id, load_backbone, train_transform)


def param_groups(model, lr, head_lr, layer_decay, wd):
    """Un groupe de paramètres par (couche, avec/sans weight decay)."""
    n = count_blocks(model.backbone)
    groups = {}
    for name, prm in model.named_parameters():
        if name.startswith("head."):
            gid = ("head", head_lr)
        else:
            lid = layer_id(name, n)
            gid = (f"layer{lid}", lr * layer_decay ** (n + 1 - lid))
        # Pas de weight decay sur les biais, normalisations et tokens (vecteurs 1-D).
        decay = 0.0 if prm.ndim <= 1 or "token" in name or "embedding" in name else wd
        key = (gid[0], decay)
        groups.setdefault(key, {"params": [], "lr": gid[1], "weight_decay": decay, "name": gid[0]})
        groups[key]["params"].append(prm)
    for g in groups.values():
        g["base_lr"] = g["lr"]
    return list(groups.values()), n


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    loss = correct = top5 = n = 0
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device)
        with amp:
            logits = model(x).float()
        loss += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        top5 += (logits.topk(5, 1).indices == y[:, None]).any(1).sum().item()
        n += len(y)
    return loss / n, correct / n, top5 / n


@torch.no_grad()
def ema_update(ema, model, decay):
    for e, m in zip(ema.state_dict().values(), model.state_dict().values()):
        if e.dtype.is_floating_point:
            e.mul_(decay).add_(m.detach(), alpha=1 - decay)
        else:
            e.copy_(m)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="bioclip2")
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--probe", type=Path, default=None, help="par défaut outputs/compare/probe_<model>.pt")
    p.add_argument("--out", type=Path, default=None, help="par défaut outputs/finetune_<model>")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-5, help="pas du bloc le plus haut du backbone")
    p.add_argument("--head-lr", type=float, default=5e-4)
    p.add_argument("--layer-decay", type=float, default=0.85)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup", type=float, default=1.0, help="en epochs")
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--ema", type=float, default=0.9998)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--smoke", action="store_true", help="mini-test sur le PC")
    args = p.parse_args()

    out = args.out or Path("outputs") / f"finetune_{args.model}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    device, amp = device_and_amp()
    torch.manual_seed(0)
    classes = json.loads((args.root / "classes.json").read_text(encoding="utf-8"))

    # ---- le modèle
    model = Classifier(load_backbone(args.model, args.img_size, args.drop_path), len(classes))
    probe = args.probe or Path("outputs/compare") / f"probe_{args.model}.pt"
    if probe.exists():
        pr = torch.load(probe, map_location="cpu")
        model.head.weight.data.copy_(pr["weight"])
        model.head.bias.data.copy_(pr["bias"])
        print(f"Tête initialisée avec le linear probe ({probe})", flush=True)
    else:
        print("Pas de linear probe trouvé : tête initialisée au hasard", flush=True)
    model.to(device)
    ema = copy.deepcopy(model).eval()
    for q in ema.parameters():
        q.requires_grad_(False)

    # ---- les données
    bb = model.backbone
    train_ds = BirdImages(args.root, "train", train_transform(args.img_size, bb.mean, bb.std), args.smoke)
    val_ds = BirdImages(args.root, "val", eval_transform(args.img_size, bb.mean, bb.std), args.smoke)
    pin = device.type == "cuda"
    train_dl = DataLoader(train_ds, args.batch, shuffle=True, drop_last=True, num_workers=args.workers,
                          pin_memory=pin, persistent_workers=args.workers > 0)
    val_dl = DataLoader(val_ds, args.batch * 2, num_workers=args.workers, pin_memory=pin)

    # ---- optimiseur et calendrier du pas d'apprentissage
    groups, n_blocks = param_groups(model, args.lr, args.head_lr, args.layer_decay, args.wd)
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999))
    steps_per_epoch = len(train_dl)
    total, warm = args.epochs * steps_per_epoch, int(args.warmup * steps_per_epoch)

    def lr_factor(step):  # montée linéaire, puis descente en cosinus jusqu'à ~0
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    print(f"{len(train_ds)} images de train, {len(val_ds)} de validation, {steps_per_epoch} pas par epoch", flush=True)
    print(f"{sum(q.numel() for q in model.parameters()) / 1e6:.0f} M paramètres, {n_blocks} blocs ; "
          f"pas d'apprentissage : tête {args.head_lr:g}, haut {args.lr:g}, "
          f"bas {args.lr * args.layer_decay ** (n_blocks + 1):.1e}", flush=True)

    # ---- reprise après une coupure
    start_epoch, history, best_acc = 0, [], -1.0
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        start_epoch, history, best_acc = ck["epoch"], ck["history"], ck["best_acc"]
        print(f"Reprise après l'epoch {start_epoch}", flush=True)

    if start_epoch == 0:  # vérification : au départ, on doit retrouver à peu près le score du linear probe
        _, acc0, _ = evaluate(model, val_dl, device, amp)
        print(f"Accuracy de validation AVANT fine-tuning : {acc0:.2%}", flush=True)

    step = start_epoch * steps_per_epoch
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, run_loss, run_n = time.time(), 0.0, 0
        for i, (x, y, _) in enumerate(train_dl):
            f = lr_factor(step)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * f
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with amp:
                loss = F.cross_entropy(model(x).float(), y, label_smoothing=args.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            # Au tout début, l'EMA suit vite (sinon elle resterait collée aux poids de départ).
            ema_update(ema, model, min(args.ema, (1 + step) / (10 + step)))
            step += 1
            run_loss, run_n = run_loss + loss.item() * len(y), run_n + len(y)
            if i % 200 == 0:
                print(f"  epoch {epoch + 1} pas {i}/{steps_per_epoch} loss {loss.item():.3f} "
                      f"({(time.time() - t0) / 60:.1f} min)", flush=True)

        v_loss, v_acc, v_top5 = evaluate(model, val_dl, device, amp)
        e_loss, e_acc, e_top5 = evaluate(ema, val_dl, device, amp)
        row = {"epoch": epoch + 1, "train_loss": run_loss / run_n, "val_loss": v_loss, "val_acc": v_acc,
               "val_top5": v_top5, "ema_val_loss": e_loss, "ema_val_acc": e_acc, "ema_val_top5": e_top5,
               "minutes": (time.time() - t0) / 60}
        history.append(row)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print(f"epoch {epoch + 1:2d} | train loss {row['train_loss']:.3f} | val acc {v_acc:.2%} "
              f"(EMA {e_acc:.2%}) top-5 {v_top5:.2%} | {row['minutes']:.1f} min", flush=True)

        # Meilleur modèle selon la VALIDATION (poids normaux ou EMA, le meilleur des deux).
        for which, net, acc in (("normal", model, v_acc), ("ema", ema, e_acc)):
            if acc > best_acc:
                best_acc = acc
                torch.save({"model": args.model, "img_size": args.img_size, "classes": classes,
                            "weights": which, "epoch": epoch + 1, "val_acc": acc,
                            "state_dict": {k: v.half() if v.is_floating_point() else v
                                           for k, v in net.state_dict().items()}},
                           out / "best.pt")
                print(f"  -> nouveau meilleur modèle ({which}, {acc:.2%}) sauvegardé", flush=True)
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch + 1, "history": history, "best_acc": best_acc}, out / "last.pt.tmp")
        (out / "last.pt.tmp").replace(out / "last.pt")

    print(f"\nTerminé. Meilleure accuracy de validation : {best_acc:.2%}  ->  {out / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
