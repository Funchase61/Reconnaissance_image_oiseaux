"""Outils partagés par les scripts GPU (comparaison, fine-tuning, évaluation, prédiction).

On y trouve :
  - BACKBONES : les modèles pré-entraînés qu'on sait charger ;
  - load_backbone : charge un modèle sans sa tête de classification ;
  - BirdImages : le dataset (lit metadata.csv et les JPEG de data/) ;
  - les transformations d'images (augmentation pour l'entraînement, propre pour l'évaluation) ;
  - layer_id : le numéro de « couche » de chaque paramètre (pour le layer-wise lr decay).
"""

import re
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms as T

# Nom court -> (bibliothèque, identifiant). Les identifiants « openclip: » viennent
# d'open_clip (modèles CLIP), les autres de timm.
BACKBONES = {
    # Petit CNN pré-entraîné sur ImageNet : le point de comparaison « transfer learning simple ».
    "resnet18": ("timm", "resnet18.a1_in1k"),
    # Vision Transformer Large entraîné SANS étiquettes sur 142 millions d'images (Meta).
    "dinov2_l": ("timm", "vit_large_patch14_dinov2.lvd142m"),
    # Vision Transformer Large entraîné façon CLIP sur 2 milliards d'images avec légendes.
    "eva02_l": ("timm", "eva02_large_patch14_clip_224.merged2b"),
    # CLIP Large spécialisé dans le vivant (entraîné sur 214 millions d'images d'organismes).
    "bioclip2": ("openclip", "hf-hub:imageomics/bioclip-2"),
    # Petits modèles réservés aux tests rapides sur le PC (mode --smoke).
    "smoke_vit": ("timm", "vit_small_patch14_dinov2.lvd142m"),
    "smoke_clip": ("openclip", "ViT-B-32/openai"),
}


class Backbone(nn.Module):
    """Enveloppe commune : image -> vecteur de caractéristiques de taille `dim`."""

    def __init__(self, net: nn.Module, dim: int, mean, std, kind: str):
        super().__init__()
        self.net, self.dim, self.mean, self.std, self.kind = net, dim, tuple(mean), tuple(std), kind

    def forward(self, x):
        return self.net(x).float()


def load_backbone(name: str, img_size: int = 224, drop_path: float = 0.0, pretrained: bool = True) -> Backbone:
    lib, ident = BACKBONES[name]
    if lib == "timm":
        import timm
        kwargs = {"pretrained": pretrained, "num_classes": 0}
        if "vit" in ident or "eva" in ident:
            kwargs["img_size"] = img_size  # DINOv2 est prévu pour 518 px : on l'adapte à nos 224 px
            kwargs["drop_path_rate"] = drop_path
        net = timm.create_model(ident, **kwargs)
        cfg = timm.data.resolve_data_config({}, model=net)
        return Backbone(net, net.num_features, cfg["mean"], cfg["std"], "timm")

    import open_clip
    if ident.startswith("hf-hub:"):
        model, _, _ = open_clip.create_model_and_transforms(ident, pretrained=pretrained or None)
    else:
        arch, tag = ident.split("/")
        model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=tag if pretrained else None)
    visual = model.visual  # on ne garde que la partie « image » de CLIP
    del model
    mean = getattr(visual, "image_mean", None) or (0.48145466, 0.4578275, 0.40821073)
    std = getattr(visual, "image_std", None) or (0.26862954, 0.26130258, 0.27577711)
    return Backbone(visual, visual.output_dim, mean, std, "openclip")


class Classifier(nn.Module):
    """Backbone pré-entraîné + une couche linéaire qui donne un score par espèce."""

    def __init__(self, backbone: Backbone, n_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.dim, n_classes)

    def forward(self, x, return_features=False):
        f = self.backbone(x)
        return (self.head(f), f) if return_features else self.head(f)


# ---------------------------------------------------------------- données

def train_transform(size, mean, std):
    """Augmentation « douce » : on change le cadrage et le sens, jamais les couleurs
    en profondeur (la couleur du plumage est un vrai indice pour l'espèce)."""
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)], p=0.5),
        T.ToTensor(),
        T.Normalize(mean, std),
        T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def eval_transform(size, mean, std):
    """Pas de hasard en évaluation. Toutes les images sont mises en size x size
    (les 210 PLUSH CRESTED JAY ne sont pas en 224 : la taille ne doit pas servir d'indice)."""
    return T.Compose([
        T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


class BirdImages(Dataset):
    """Les images d'un tas (train, val ou test), lues d'après metadata.csv."""

    def __init__(self, root: Path, split: str, transform, smoke: bool = False):
        meta = pd.read_csv(root / "metadata.csv")
        meta = meta[meta.partition == split].reset_index(drop=True)
        if smoke:  # mini-version pour tester le code sur le PC : 10 espèces, peu d'images
            meta = meta[meta.label < 10].groupby("label").head(12 if split == "train" else 3).reset_index(drop=True)
        self.root, self.meta, self.transform = root, meta, transform
        self.labels = torch.tensor(meta.label.values)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        with Image.open(self.root / self.meta.chemin[i]) as im:
            img = im.convert("RGB")
        return self.transform(img), int(self.meta.label[i]), i


# ---------------------------------------------------------------- layer-wise lr decay

def layer_id(param_name: str, n_blocks: int) -> int:
    """0 = l'entrée du réseau (découpage en patchs), 1..n_blocks = les blocs du
    transformer, n_blocks + 1 = la sortie. Les couches basses (bords, textures)
    sont déjà bonnes : on les modifiera moins que les couches hautes."""
    m = re.search(r"(?:^|\.)(?:blocks|resblocks)\.(\d+)\.", param_name)
    if m:
        return int(m.group(1)) + 1
    stem = ("patch_embed", "cls_token", "pos_embed", "reg_token", "mask_token",
            "class_embedding", "positional_embedding", "ln_pre", "rope")
    if any(s in param_name for s in stem) or re.search(r"(?:^|\.)conv1\.", param_name) and n_blocks:
        return 0
    return n_blocks + 1


def count_blocks(model: nn.Module) -> int:
    ids = [int(m.group(1)) for n, _ in model.named_parameters()
           if (m := re.search(r"(?:^|\.)(?:blocks|resblocks)\.(\d+)\.", n))]
    return max(ids) + 1 if ids else 0


def device_and_amp():
    """Sur carte graphique : calcul en bfloat16 (2 fois plus rapide, assez précis)."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.device("cuda"), torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.device("cpu"), torch.autocast("cpu", enabled=False)


def load_classifier(ckpt_path: Path, device="cpu") -> tuple[Classifier, dict]:
    """Recharge un modèle fine-tuné sauvegardé par finetune.py (fichier best.pt)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    bb = load_backbone(ck["model"], ck["img_size"], pretrained=False)
    clf = Classifier(bb, len(ck["classes"]))
    clf.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()})
    return clf.to(device).eval(), ck
