"""Étape 0 : sortir les images des fichiers parquet et les ranger en dossiers.

Résultat (structure de la slide 80 du cours, lisible par ImageFolder) :
    data/train/<ESPECE>/<n>.jpg
    data/val/<ESPECE>/<n>.jpg
    data/test/<ESPECE>/<n>.jpg
    metadata.csv  : largeur, hauteur, chemin, label, espece, partition, octets, sha1
    classes.json  : la liste des 525 espèces, dans l'ordre des labels (0 à 524)

Les images sont déjà des JPEG dans les parquet : on recopie leurs octets tels
quels, sans les décompresser ni les recompresser (aucune perte de qualité).

Utilisation (depuis le dossier projet_oiseaux) :
    python scripts/prepare_data.py --src ../../birds-525-species-image-classification/data
"""

import argparse
import hashlib
import io
import json
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

# Nom du tas dans les fichiers parquet  ->  nom du dossier chez nous
SPLITS = {"train": "train", "validation": "val", "test": "test"}


def class_names(parquet_file: Path) -> list[str]:
    """Lit la liste des espèces rangée par Hugging Face dans les métadonnées du parquet."""
    meta = pq.read_schema(parquet_file).metadata[b"huggingface"]
    info = json.loads(meta)["info"]["features"]["label"]
    return info["names"]


def safe_folder(name: str) -> str:
    """Un nom de dossier valide sous Windows (on retire les caractères interdits)."""
    return re.sub(r'[<>:"/\\|?*]', "_", name).strip().rstrip(".")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True, help="dossier des fichiers .parquet")
    parser.add_argument("--out", type=Path, default=Path("."), help="dossier du projet")
    args = parser.parse_args()

    files = sorted(args.src.glob("*.parquet"))
    if not files:
        raise SystemExit(f"Aucun fichier .parquet dans {args.src}")

    raw_names = class_names(files[0])

    # Correction d'un défaut du dataset : la même espèce apparaît sous deux noms
    # qui ne diffèrent que par un espace en double ("PARAKETT  AUKLET" n°380,
    # utilisé par train et test, et "PARAKETT AUKLET" n°381, utilisé par la
    # validation). Sans correction, ces 5 images de validation seraient
    # impossibles à réussir. On fusionne les noms identiques à l'espace près et
    # on renumérote les espèces de 0 à 524.
    names, remap = [], {}
    for old, n in enumerate(raw_names):
        clean = " ".join(n.split())
        if clean not in names:
            names.append(clean)
        remap[old] = names.index(clean)
    merged = len(raw_names) - len(names)
    print(f"{len(raw_names)} noms dans le dataset, {merged} doublon(s) fusionné(s) -> {len(names)} espèces")

    folders = [safe_folder(n) for n in names]
    assert len(names) == 525, len(names)
    assert len(set(folders)) == len(folders), "deux espèces auraient le même dossier"

    data_dir = args.out / "data"
    rows = []
    for split, folder in SPLITS.items():
        split_files = [f for f in files if f.name.startswith(split + "-")]
        counter = 0
        for f in split_files:
            pf = pq.ParquetFile(f)
            # Lecture par morceaux (row groups) pour ne pas charger 500 Mo d'un coup
            for g in range(pf.num_row_groups):
                table = pf.read_row_group(g).to_pylist()
                for row in table:
                    label = remap[row["label"]]
                    data = row["image"]["bytes"]
                    counter += 1
                    rel = Path("data", folder, folders[label], f"{counter:06d}.jpg")
                    path = args.out / rel
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                    with Image.open(io.BytesIO(data)) as im:  # lit seulement l'en-tête
                        width, height = im.size
                    rows.append({
                        "largeur": width,
                        "hauteur": height,
                        "chemin": rel.as_posix(),
                        "label": label,
                        "espece": names[label],
                        "partition": folder,
                        "octets": len(data),
                        # Empreinte du fichier : repère les doublons EXACTS entre tas
                        "sha1": hashlib.sha1(data).hexdigest(),
                    })
            print(f"{f.name} lu ({counter} images {folder} jusqu'ici)", flush=True)

    meta = pd.DataFrame(rows)
    meta.to_csv(args.out / "metadata.csv", index=False)
    (args.out / "classes.json").write_text(
        json.dumps(names, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\nImages par tas :")
    print(meta["partition"].value_counts().to_string())
    print(f"\nÉcrit dans {data_dir.resolve()}")


if __name__ == "__main__":
    main()
