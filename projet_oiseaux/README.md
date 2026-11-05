# Projet oiseaux : reconnaître 525 espèces sur une photo

Projet final du cours AI/ML (jour 3, slides 70 à 88). Le but est un classifieur d'images qui donne l'espèce d'un oiseau, avec une démarche complète : baseline, petit CNN, transfer learning, fine-tuning, analyse des erreurs.

L'explication complète, pour débutant, de tout ce qui a été fait et pourquoi est dans [DE_A_A_Z.md](DE_A_A_Z.md). Le plan de départ est dans [EXPLICATIONS.md](EXPLICATIONS.md).

## Le dataset

- **Source** : « BIRDS 525 SPECIES » (Kaggle, gpiosenka), version Hugging Face en fichiers parquet, dans `../../birds-525-species-image-classification/data`.
- **Contenu** : des images JPEG de 224 × 224 en couleur, avec en général un oiseau centré.
- **Effectifs** : 84 635 images de train, 2 625 de validation, 2 625 de test (5 par espèce en validation et en test).
- **Correction apportée** : le fichier d'origine déclare 526 noms, car « PARAKETT AUKLET » y apparaît deux fois (avec un espace en double). Le train et le test utilisent l'un des deux noms, la validation l'autre. `prepare_data.py` fusionne les deux, ce qui donne bien 525 espèces numérotées de 0 à 524.
- **Points d'attention** : 211 images ne font pas 224 × 224, dont les 210 de PLUSH CRESTED JAY (toutes redimensionnées en 224 × 224 au chargement, pour que la taille ne serve pas d'indice). Certaines photos portent un filigrane. Aucun doublon exact entre fichiers.
- **Déséquilibre du train** : de 130 à 263 images par espèce (161 en moyenne).

## Organisation

```
projet_oiseaux/
  EXPLICATIONS.md               tout le projet expliqué pas à pas
  README.md                     ce fichier
  RUNPOD.md                     guide pas à pas pour la partie GPU
  runpod_bundle.zip             les scripts à envoyer sur le pod
  runpod/                       installation, lancement, récupération des résultats sur le pod
  scripts/prepare_data.py       étape 0 : parquet -> data/<tas>/<ESPECE>/*.jpg
  scripts/common.py             outils partagés (modèles, images, augmentation)
  scripts/compare_backbones.py  étapes 4-5 : linear probe de 4 backbones
  scripts/finetune.py           étape 6 : fine-tuning complet
  scripts/evaluate.py           étapes 7-8 : scores, TTA, prédictions, test protégé
  scripts/predict.py            livrable : espèce d'oiseau sur une photo
  data/train|val|test/          les images (non versionnées, se recréent avec le script)
  metadata.csv                  chemin, label, espèce, tas, taille, empreinte sha1
  classes.json                  les 525 espèces, dans l'ordre des labels
  01_exploration_baseline.ipynb étapes 1 et 2
  02_petit_cnn.ipynb            étape 3 : petit CNN partant de zéro
  outputs/                      poids des modèles et cache (non versionnés)
```

## Recréer les données

Depuis le dossier `projet_oiseaux` :

```
python scripts/prepare_data.py --src ../../birds-525-species-image-classification/data
```

Il faut environ 2 Go d'espace disque.

## Journal des essais

| Étape | Modèle | Accuracy de validation | Remarque |
|---|---|---|---|
| 2 | Baseline (classe majoritaire) | 0,19 % | Toujours « RUFOUS TREPE » ; autant que le hasard (1/525) |
| 3 | Petit CNN (4 blocs conv, 0,5 M paramètres, 64 × 64, 12 epochs, CPU) | 78,8 % ± 1,6 (top-5 92,5 %) | 65 min sur portable ; pas de sur-apprentissage, plafonne ; 2 étiquettes suspectes repérées (TEAL DUCK, APAPANE) |
| 4-5 | ResNet18 gelé + linear probe | 92,50 % (top-5 98,63 %) | Pré-entraîné ImageNet ; 14 points de plus que le petit CNN |
| 4-5 | EVA-02 Large gelé + linear probe | 98,90 % (top-5 100 %) | |
| 4-5 | BioCLIP 2 gelé + linear probe | 99,12 % (top-5 99,96 %) | |
| 4-5 | **DINOv2 ViT-L gelé + linear probe** | **99,28 % ± 0,3** (top-5 99,92 %) | Gagnant ; écarts avec BioCLIP 2 et EVA-02 dans la marge d'erreur |
| 6 | DINOv2 ViT-L fine-tuné (12 epochs, L40S) | en cours | |
