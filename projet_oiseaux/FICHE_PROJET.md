# Fiche projet : reconnaître l'espèce d'un oiseau sur une photo

Fiche demandée par la slide 74 du cours, remplie avec les résultats obtenus le 5 novembre 2026.

## Usage

**Nous voulons reconnaître** l'espèce d'un oiseau sur une photo, parmi 525 espèces du monde entier, **pour** aider un observateur amateur à identifier un oiseau qu'il a photographié. Le programme propose les 5 espèces les plus probables, avec la confiance du modèle pour chacune, et prévient quand il est peu sûr de lui.

## Classes

- **525 espèces**, une classe par espèce. La liste complète est dans `classes.json`.
- **Règle d'appartenance** : l'étiquette fournie par le dataset. Une photo montre un seul oiseau, en général centré.
- **Cas ambigus repérés** :
  - deux noms pour la même espèce (« PARAKETT AUKLET », avec ou sans espace en double), fusionnés en une seule classe ;
  - des espèces très proches (PURPLE FINCH et HOUSE FINCH, deux touracos, deux tragopans) ;
  - des étiquettes probablement fausses (TEAL DUCK, BALTIMORE ORIOLE, MCKAYS BUNTING, APAPANE, EUROPEAN GOLDFINCH) ;
  - une classe incohérente (AUSTRAL CANASTERO).
- **Hors du périmètre** : les espèces absentes de la liste, les photos avec plusieurs oiseaux, les oiseaux non visibles.

## Collecte

- **Source** : dataset public « BIRDS 525 SPECIES » (Kaggle, version Hugging Face `yashikota/birds-525-species-image-classification`). Il contient 89 885 photos JPEG en couleur, de 224 × 224 pixels.
- **Écart avec le sujet** : le cours propose de collecter ses propres photos (80 à 150 par classe). Nous avons utilisé un dataset existant. **Question posée au professeur.**
- **Variations couvertes** : beaucoup d'espèces, de fonds et de lumières. Peu de jeunes, de femelles, d'oiseaux en vol ou de photos floues : ce sont surtout des photos « de catalogue ».
- **Nettoyage** : le nom en double a été fusionné. Aucun doublon exact (empreinte sha1). Toutes les images sont ramenées à la même taille, car celles de PLUSH CRESTED JAY avaient une taille différente qui aurait pu servir d'indice.

## Évaluation

| Tas | Photos | Part | Par espèce | Rôle |
|---|---|---|---|---|
| Train | 84 635 | 94,2 % | 130 à 263 | Apprendre les poids |
| Validation | 2 625 | 2,9 % | 5 | Choisir le modèle et les réglages |
| Test | 2 625 | 2,9 % | 5 | Bilan final, consulté **une seule fois** |

- **Séparation** : la séparation officielle du dataset, fixée avant tout essai.
- **Métriques** : accuracy, avec sa marge d'erreur à 95 %, et top-5. Le dataset est presque équilibré, donc l'accuracy est une bonne métrique. On regarde aussi le score de chaque espèce.
- **Indépendance** : le dataset ne dit pas d'où vient chaque photo. La recherche de quasi-doublons a trouvé 131 photos de test (5 %) presque identiques à une photo du train. Le score n'en est pas gonflé : 99,5 % sur les photos sans « jumeau ».

## Point de départ et résultats

| Étape | Modèle | Validation | Test |
|---|---|---|---|
| Baseline | Toujours l'espèce la plus fréquente | 0,19 % | — |
| Petit CNN | 4 blocs de convolution, partant de zéro, 64 × 64, sur portable | 78,8 % ± 1,6 | — |
| Transfer learning (gelé) | DINOv2 ViT-L + linear probe (meilleur de 4 backbones) | 99,28 % | — |
| **Fine-tuning** | **DINOv2 ViT-L fine-tuné, 12 epochs, GPU loué** | **99,43 % ± 0,29** | **99,43 % ± 0,29** |

**Choix justifié par le coût de calcul** (slide 82) :
- le petit CNN a tourné sur le portable, sans carte graphique, en 65 minutes ;
- les gros modèles ont tourné sur une carte graphique louée (RunPod, NVIDIA L40S), pour environ 3 h 20 et 3,50 $.

## Ce qui reste fragile, et la prochaine expérience

- Environ **6 des 15 erreurs du test viennent du dataset** (étiquettes fausses, classe incohérente), le reste d'espèces très proches ou de photos inhabituelles.
- Le score sur de **vraies photos d'amateur** reste à mesurer : il sera plus bas.
- **Prochaine expérience** (une modification justifiée) :
  1. nettoyer le dataset : retirer les doublons entre train et test, corriger les étiquettes fausses, vérifier AUSTRAL CANASTERO ;
  2. réentraîner ;
  3. mesurer sur un nouveau test indépendant et sur nos propres photos.
