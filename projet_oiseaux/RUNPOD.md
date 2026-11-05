# Faire tourner la partie GPU sur RunPod, pas à pas

Ce guide fait les étapes 4 à 7 du projet sur une carte graphique louée :

- comparer 4 backbones pré-entraînés (linear probe) ;
- fine-tuner le meilleur ;
- le mesurer sur la validation, puis une seule fois sur le test.

Tout est automatique. Tu lances une commande et tu reviens environ 2 heures plus tard.

**Coût prévu** : environ 2 h 30 avec une L40S à 1,09 $/h, soit **3 $ environ**, plus quelques centimes de disque.

Les scripts ont été testés sur le PC avec de petits modèles et 10 espèces : tout fonctionne de bout en bout. Seuls les gros modèles et le GPU n'ont pas pu être testés sur le PC.

## 0. Avant de louer : ce qu'il faut avoir

- Le fichier **`runpod_bundle.zip`**, dans ce dossier. Il contient tous les scripts et ne pèse que quelques Ko.
- Un compte RunPod avec au moins 5 $ de crédit.

Les images ne sont pas à envoyer : le pod les télécharge lui-même depuis Hugging Face, beaucoup plus vite que depuis ta connexion.

## 1. Créer le pod

1. RunPod → **Pods** → **Deploy**.
2. Choisis **L40S** (48 Go de mémoire, 1,09 $/h). Une A40 ou une RTX A6000 (0,59 $/h) marchent aussi, environ 2 fois plus lentement : à peu près le même prix au total.
3. Template : **RunPod PyTorch** (n'importe quelle version 2.x).
4. Clique sur **Edit Template** et règle :
   - *Container Disk* : **30 Go** ;
   - *Volume Disk* : **60 Go**. C'est le disque `/workspace` : les modèles téléchargés (environ 7 Go), les images (2 Go) et les sauvegardes (environ 15 Go).
5. Coche **Start Jupyter Notebook**, puis **Deploy**.
6. Quand le pod est prêt, clique sur **Connect** → **Jupyter Lab**.

À partir de là, le compteur tourne. Voici pourquoi chaque réglage compte :

- La L40S a assez de mémoire pour entraîner un ViT-Large avec des paquets de 64 images.
- Le *Volume* garde les fichiers si le pod redémarre.

## 2. Envoyer les scripts et installer

Dans Jupyter Lab :

1. Glisse `runpod_bundle.zip` dans le panneau de fichiers, à gauche. Il arrive dans `/workspace`.
2. Ouvre un terminal : **File → New → Terminal**.
3. Tape :

```bash
cd /workspace
unzip -o runpod_bundle.zip
cd projet_oiseaux
bash runpod/setup.sh
```

Ça prend environ 10 minutes. À la fin, le script doit afficher `Carte graphique : NVIDIA L40S`.

## 3. Lancer tout le calcul

Toujours dans le terminal, dans `projet_oiseaux` :

```bash
mkdir -p outputs
nohup bash runpod/run_all.sh > outputs/run_all.log 2>&1 &
tail -f outputs/run_all.log
```

- `nohup … &` lance le calcul « en arrière-plan ». Tu peux fermer l'onglet, ou même ton PC : le calcul continue sur le pod.
- `tail -f` affiche le journal en direct. **Ctrl+C** arrête seulement l'affichage, pas le calcul. Pour revoir le journal plus tard : `tail -f outputs/run_all.log`.

Voici ce qui va défiler :

| Étape | Durée estimée | À surveiller |
|---|---|---|
| Comparaison des 4 backbones | environ 20 à 30 min | une ligne `linear probe … val acc` par modèle et par réglage |
| Fine-tuning du meilleur (12 epochs) | environ 1 h à 1 h 30 | `Accuracy de validation AVANT fine-tuning` doit être proche du score du linear probe ; puis une ligne par epoch |
| Évaluation sur la validation | 1 min | accuracy sans et avec TTA |

**Si le pod s'arrête en route**, relance exactement la même commande `nohup …`. Les backbones déjà comparés sont sautés, et le fine-tuning reprend à la dernière epoch terminée, grâce au fichier `last.pt`.

**Si une erreur apparaît**, copie-moi les 30 dernières lignes du journal. Pour les obtenir : `tail -30 outputs/run_all.log`.

## 4. Le test, une seule fois

Quand le journal affiche `== Terminé`, regarde le score de validation et envoie-le-moi. Si tout est cohérent, lance le test **une seule fois**. Le journal affiche la commande exacte, par exemple :

```bash
python scripts/evaluate.py --ckpt outputs/finetune_bioclip2/best.pt --split test
```

Le script refuse de réévaluer le test une deuxième fois. C'est volontaire : le test doit rester l'examen final, pas un outil de réglage.

## 5. Récupérer les résultats, puis supprimer le pod

```bash
bash runpod/pack_results.sh
```

Ce script crée `resultats_runpod.tar`, d'environ 1 Go. Il contient :

- le modèle final `best.pt` ;
- les journaux ;
- les scores ;
- les prédictions de chaque image ;
- les vecteurs qui serviront à chercher les quasi-doublons.

Dans Jupyter, fais un clic droit sur `resultats_runpod.tar` puis **Download**. Mets le fichier dans `projet_oiseaux` sur le PC.

Ensuite, dans RunPod : **Stop**, puis **Terminate** (la poubelle). **Ne l'oublie pas.** Un pod arrêté mais pas supprimé continue de coûter le prix de son disque, et un pod allumé coûte 1,09 $ par heure, même s'il ne calcule rien.

## Ce que fait chaque script

| Fichier | Rôle |
|---|---|
| `runpod/setup.sh` | installe les bibliothèques, télécharge et range les images |
| `runpod/run_all.sh` | enchaîne la comparaison, le fine-tuning et l'évaluation sur la validation |
| `runpod/pack_results.sh` | rassemble les résultats à télécharger |
| `scripts/common.py` | outils partagés : modèles, images, augmentation |
| `scripts/compare_backbones.py` | étapes 4 et 5 : feature extraction et linear probe pour chaque backbone |
| `scripts/finetune.py` | étape 6 : fine-tuning complet, avec reprise automatique |
| `scripts/evaluate.py` | étapes 7 et 8 : scores, TTA, prédictions détaillées, protection du test |
| `scripts/predict.py` | livrable : l'espèce d'oiseau sur tes propres photos (marche aussi sur le PC) |
