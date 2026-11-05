# Projet oiseaux de A à Z : tout ce qui a été fait, et pourquoi

5 novembre 2026 · copie du document Claude [Projet oiseaux de A à Z](https://claude.ai/artifact/4obQ3hQCeciJNkLtZX9RWY)

## Le but et la carte du voyage

Le but : un programme qui regarde la photo d'un oiseau et dit de quelle espèce il s'agit, parmi 525 espèces. Ce document raconte tout ce qui a été fait, dans l'ordre, et pourquoi chaque choix a été fait.

On avance comme un explorateur : on part de l'idée la plus bête possible, puis on monte d'un cran à chaque fois, en mesurant le progrès. Voici les étapes, avec où on en est le 5 novembre 2026 :

1. **Ranger les photos** dans des dossiers, une espèce par dossier. *Fait.*
2. **Explorer** : compter, regarder, chercher les pièges. *Fait.*
3. **Baseline** : le score de l'idée la plus bête (0,19 %). *Fait.*
4. **Petit réseau parti de zéro** sur ton ordinateur (78,8 %). *Fait.*
5. **Comparer 4 gros modèles déjà entraînés**, sans les modifier (linear probe). *Fait : DINOv2 gagne avec 99,3 %.*
6. **Ajuster le meilleur** des quatre sur nos oiseaux (fine-tuning). *Fait : 99,43 % en validation.*
7. **Mesurer** sur la validation, puis **une seule fois** sur le test. *Fait : 99,43 % ± 0,29 sur le test.*
8. **Comprendre les erreurs** et écrire les limites. *Fait (notebook 03).*
9. **Livrer** : le modèle, un programme qui l'utilise, les notebooks, la présentation. *En partie fait.*

## Ce que demande le sujet, et comment on le suit

Le sujet (slides 70 à 88 du cours) est suivi pour toute la méthode, avec trois écarts expliqués plus bas. Le cours le dit lui-même : *« la qualité de la démarche compte davantage qu'un classement entre scores »*.

**Est-ce que le sujet autorise un modèle déjà entraîné ?** Oui. La slide 82 propose trois points de départ : une baseline, un petit CNN, et le *transfer learning* (« réutiliser un backbone préentraîné et apprendre une nouvelle tête »). La slide 88 répète : « un modèle préentraîné peut servir de point de départ ». On fait les trois, dans cet ordre.

| Ce que demande le cours | Slide | Ce qu'on a fait | État |
|---|---|---|---|
| Classes claires, labels cohérents, doublons retirés | 78 | 525 espèces ; un nom en double trouvé et corrigé ; aucun doublon exact | Fait |
| Chercher les faux indices (décor, qualité) | 77 | Taille d'image et filigranes repérés ; toutes les images mises à la même taille | Fait |
| Séparer train, validation, test avant les essais | 79 | Séparation officielle du dataset gardée, test jamais regardé | Fait |
| Dossiers data/train, val, test + metadata.csv + README | 80 | Exactement cette structure | Fait |
| ImageFolder, mêmes classes partout, afficher un batch | 81 | Vérifié dans le notebook 01 | Fait |
| Augmentations seulement sur le train | 81 | Oui, partout | Fait |
| Baseline, puis petit CNN, puis transfer learning | 82 | 0,19 %, puis 78,8 %, puis 99,3 % (modèle gelé), puis 99,43 % fine-tuné | Fait |
| Examiner les erreurs de validation | 83 | Notebook 02 : 6 erreurs commentées, 2 étiquettes fausses trouvées | Fait (CNN et modèle final) |
| Test consulté une seule fois | 85 | Le programme refuse une deuxième évaluation | Fait (99,43 %) |
| Modèle rechargeable avec preprocessing et classes | 86 | Fichier best.pt + programme predict.py | Fait |
| Journal des essais | 86 | Tableau dans le README | Fait |
| Fiche projet | 74 | FICHE_PROJET.md | Fait |
| Analyse, limites, prochaine expérience | 86 | Notebook 03 : erreurs classées, quasi-doublons, limites, prochaine expérience | Fait |
| Présentation avec effectifs en pourcentages | 87 | Prévue | À faire |

**Les trois écarts, et pourquoi on les garde :**

1. **Le dataset n'est pas collecté par nous.** Le cours demande de constituer son propre dataset (environ 80 à 150 photos par classe, 2 ou 3 classes pour commencer). On utilise un dataset existant de 525 espèces. Il faut demander au professeur si c'est accepté. Si non, on ajoute une collecte à nous.
2. **Le fine-tuning change plusieurs réglages à la fois.** Le cours conseille de tester une modification justifiée à la fois. On applique une recette standard complète, car la retirer ferait perdre de la qualité. Chaque marche (baseline, CNN, modèle gelé, modèle ajusté) reste mesurée séparément.
3. **On ne peut pas garantir des groupes indépendants.** Le cours veut garder ensemble les photos d'une même séance. Le dataset ne dit pas d'où vient chaque photo, et la séparation est 94 / 3 / 3 % au lieu de 70 / 15 / 15 %. On compense en cherchant les photos presque identiques entre train et test.

## Les bases : comment un ordinateur voit et apprend

Un ordinateur ne voit pas un oiseau : il voit un grand tableau de nombres, et un réseau de neurones apprend à transformer ces nombres en un nom d'espèce.

**Une image, ce sont des nombres.** Une photo de 224 × 224 pixels, c'est 224 × 224 = 50 176 petits carrés. Chaque carré a trois nombres : combien de rouge, de vert et de bleu (de 0 à 255). Au total, environ 150 000 nombres par photo.

**Un réseau de neurones, c'est une machine avec des millions de boutons.** Les nombres de l'image entrent d'un côté. Ils sont mélangés, multipliés, additionnés par couches successives. De l'autre côté sortent 525 scores, un par espèce. Le plus grand score, c'est la réponse. Les « boutons » s'appellent les **poids** : ce sont eux qui décident du résultat.

**Apprendre, c'est tourner les boutons dans le bon sens.** Au début, les boutons sont au hasard et les réponses sont fausses. On montre une photo dont on connaît la réponse. On mesure à quel point le réseau s'est trompé : c'est la **loss** (la « perte », une note d'erreur, plus elle est basse mieux c'est). Puis on calcule, pour chaque bouton, dans quel sens le tourner pour que l'erreur baisse un peu : c'est la **descente de gradient**. On recommence des millions de fois.

**Le pas d'apprentissage** (*learning rate*) dit de combien on tourne les boutons à chaque fois. Trop grand : on dépasse et le réseau devient fou. Trop petit : il n'apprend presque rien.

**Une epoch**, c'est un tour complet de toutes les photos d'entraînement. On en fait plusieurs, et on ne les montre pas une par une mais par paquets (des **batchs** de 64 ou 128 photos) : c'est beaucoup plus rapide.

**Un CNN** (réseau à convolutions) est un réseau spécial pour les images. Il fait glisser de petites loupes sur l'image. Les premières couches repèrent des bords et des couleurs, les suivantes des formes (un œil, une plume), les dernières des objets entiers (une tête de mésange). **Un Vision Transformer** (ViT) fait la même chose autrement : il découpe l'image en petits carrés et laisse chaque carré « regarder » tous les autres.

**Le grand danger : apprendre par cœur.** Un réseau assez gros peut mémoriser les photos d'entraînement sans rien comprendre, comme un élève qui apprend les réponses du contrôle blanc. C'est le **sur-apprentissage**. Toute la méthode qui suit sert à l'éviter et à le détecter.

## Étape 0 : récupérer et ranger les photos

Les 89 885 photos ont été sorties de leurs fichiers compressés et rangées dans un dossier par espèce, et un défaut du dataset a été corrigé au passage.

**D'où viennent les photos.** Le dataset « BIRDS 525 SPECIES », publié sur Kaggle puis sur Hugging Face. Il arrive sous forme de fichiers *parquet* : de grosses boîtes qui contiennent les photos (en JPEG) et leur étiquette (un numéro d'espèce).

**Ce que fait le programme `prepare_data.py`.** Il ouvre chaque boîte, et pour chaque photo :

- il la copie telle quelle dans `data/<tas>/<ESPÈCE>/`, sans la recompresser (aucune perte de qualité) ;
- il note dans `metadata.csv` son chemin, son espèce, son tas, sa taille en pixels et son **empreinte** (sha1). L'empreinte est un code calculé à partir du contenu du fichier : deux fichiers identiques ont la même empreinte. Elle sert à trouver les doublons.

**Pourquoi ranger en dossiers ?** C'est la structure demandée par le cours (slide 80). L'outil ImageFolder de PyTorch comprend tout seul que le nom du dossier est l'étiquette.

**Le bug trouvé : une espèce en double.** Le dataset annonce 526 noms au lieu de 525. Le nom « PARAKETT AUKLET » existe deux fois, une fois avec un espace en trop. Le train et le test utilisent l'un, la validation l'autre. Sans correction, le modèle n'aurait jamais pu réussir ces 5 photos de validation : il n'aurait jamais vu d'exemple avec cette étiquette-là. Le programme fusionne les deux noms et renumérote les espèces de 0 à 524. C'est exactement le « contrôle des labels » de la slide 78.

## Étape 1 : explorer le dataset et chercher les pièges

Avant d'entraîner quoi que ce soit, on regarde les données : on a trouvé un déséquilibre léger, un faux indice de taille et des filigranes, mais aucun doublon exact. Tout est dans le notebook `01_exploration_baseline.ipynb`.

**Pourquoi regarder d'abord ?** Un modèle apprend tout ce qui l'aide à réussir, même les mauvaises raisons. Si toutes les photos d'une espèce ont un détail commun qui n'a rien à voir avec l'oiseau, il apprendra ce détail. Le cours appelle ça un **faux indice** (slide 77).

| Ce qu'on a vérifié | Ce qu'on a trouvé | Ce qu'on en fait |
|---|---|---|
| Nombre de photos par espèce (train) | De 130 à 263, 161 en moyenne | Déséquilibre faible : on le garde, et on regarde le score de chaque espèce |
| Validation et test | Exactement 5 photos par espèce | Parfait pour comparer les espèces entre elles |
| Taille des images | 211 images ne font pas 224 × 224, dont les 210 de PLUSH CRESTED JAY | Faux indice : le modèle pourrait reconnaître l'espèce à sa taille. On met toutes les images à la même taille. |
| Couleurs | Toutes les images en couleur (RVB) | Rien à faire |
| Filigranes | Un logo « alamy » visible sur certaines photos (par ex. VISAYAN HORNBILL) | À surveiller dans l'analyse des erreurs |
| Doublons exacts | Aucun (empreintes toutes différentes) | Il reste à chercher les photos *presque* identiques |

**Le contrôle de la slide 81.** On charge les dossiers avec ImageFolder, on vérifie que les 525 classes sont dans le même ordre dans les trois tas, et on affiche un paquet de 64 photos avec leur étiquette. Tout correspond.

## Train, validation, test : les trois tas

Les photos sont séparées en trois tas qui ne se mélangent jamais, et le tas de test n'est ouvert qu'une seule fois, à la toute fin.

Pense à un élève qui prépare un examen :

| Tas | Photos | Part | Comme pour l'élève | À quoi ça sert |
|---|---|---|---|---|
| **Train** | 84 635 | 94,2 % | Le cahier d'exercices corrigés | Le modèle tourne ses boutons en regardant ces photos |
| **Validation** | 2 625 | 2,9 % | Les contrôles blancs | Choisir le meilleur modèle et les réglages |
| **Test** | 2 625 | 2,9 % | Le vrai examen | Donner la note finale, une seule fois |

**Pourquoi ne pas tout mélanger ?** Si l'élève a vu les questions de l'examen à l'avance, sa note ne dit plus rien de ce qu'il sait. Pareil pour le modèle : une photo de test vue pendant l'entraînement donnerait un score trop beau et faux.

**Pourquoi le test une seule fois ?** Si on regarde le test, puis qu'on change un réglage, puis qu'on regarde à nouveau, on finit par régler le modèle *pour* le test. C'est une triche discrète. Le programme d'évaluation refuse donc de le faire deux fois (slide 85).

**Pourquoi garder la séparation du dataset (94 / 3 / 3 %) et pas 70 / 15 / 15 % ?** C'est la séparation officielle, utilisée par tous ceux qui publient sur ce dataset : nos scores restent comparables aux leurs. Le train est très grand, ce qui aide le modèle. Et la validation et le test ont exactement 5 photos par espèce : c'est équitable.

**Le défaut : la marge d'erreur.** 2 625 photos, c'est peu pour mesurer un très bon modèle. Si on refaisait le test avec 2 625 autres photos, le score bougerait un peu, par hasard. Cette marge (à 95 % de confiance) dépend du score :

| Score mesuré | Marge d'erreur | Ce que ça veut dire |
|---|---|---|
| 50 % | ± 1,9 point | Le vrai score est entre 48,1 et 51,9 % |
| 80 % | ± 1,6 point | Le cas du petit CNN |
| 95 % | ± 0,8 point | |
| 99 % | ± 0,4 point | Deux modèles à 99,2 et 99,5 % ne peuvent pas être départagés |

C'est pour ça que chaque score de ce document est donné avec son « ± ».

## Étape 2 : la baseline

La baseline, la règle la plus bête possible, obtient **0,19 %** : c'est le plancher que tout vrai modèle doit dépasser.

La règle : répondre toujours l'espèce qui a le plus de photos dans le train (RUFOUS TREPE), sans même regarder l'image. Comme la validation a 5 photos par espèce, cette règle réussit 5 photos sur 2 625. C'est autant que tirer une espèce au hasard (1 chance sur 525, soit 0,19 %).

**Pourquoi mesurer quelque chose d'aussi bête ?** Pour savoir ce que veut dire un score. Si un dataset avait 90 % de photos d'une seule espèce, la baseline ferait 90 %, et un modèle à 91 % ne vaudrait presque rien. Ici, le plancher est tout en bas : chaque point gagné est un vrai progrès (slide 82).

## Étape 3 : le petit CNN parti de zéro

Un petit réseau entraîné **à partir de zéro** sur ton ordinateur atteint **78,8 % (± 1,6)** en validation, et la bonne espèce est dans ses 5 premières propositions 92,5 % du temps. Tout est dans `02_petit_cnn.ipynb`.

**« À partir de zéro »** veut dire : ses boutons commencent au hasard. Il n'a jamais rien vu avant nos oiseaux. C'est le « petit CNN » de la slide 82, comme au TP du cours.

**Comment il est construit.** 4 blocs qui se suivent. Chaque bloc fait glisser des loupes de 3 × 3 pixels sur l'image (convolution), règle le niveau des signaux (BatchNorm), garde ce qui est positif (ReLU), puis divise l'image par deux (MaxPool). Les loupes passent de 32 à 256. À la fin, une couche donne les 525 scores. En tout : 523 821 boutons, environ 0,5 million.

**Les réglages, et pourquoi.**

- **Images réduites à 64 × 64 pixels.** Ton ordinateur n'a pas de carte graphique : en 224 × 224, une epoch aurait pris plus d'une heure.
- **12 epochs**, par paquets de 128 photos, avec l'optimiseur Adam. Le pas d'apprentissage descend doucement jusqu'à zéro (calendrier « cosinus »).
- **Augmentation** : à chaque passage, la photo est parfois retournée comme dans un miroir et légèrement décalée. Le modèle voit donc des variantes, et a plus de mal à apprendre par cœur.
- **Dropout** : pendant l'entraînement, on éteint au hasard 30 % des signaux de la dernière couche. Le modèle ne peut pas compter sur un seul détail.

**Ce que montrent les courbes.** L'erreur de validation baisse jusqu'au bout et ne remonte jamais : **pas de sur-apprentissage**. Mais les dernières epochs ne gagnent presque plus rien (78,3, puis 78,9, puis 78,8 %). Le modèle a atteint sa limite : il est trop petit et ses images trop floues. Il manque de connaissances, pas d'entraînement.

**Ce que montrent ses erreurs.** On a regardé les 6 erreurs où il était le plus sûr de lui :

- Il confond des oiseaux qui ont les **mêmes couleurs** mais rien d'autre en commun (Scarlet Tanager et Coq-de-roche, tous deux rouge et noir). En 64 × 64, la forme du bec disparaît.
- Il confond des espèces **vraiment proches**, que même un humain confond (Purple Finch et House Finch).
- **Deux photos semblent mal étiquetées dans le dataset** : celle marquée TEAL DUCK montre un colvert (MALLARD DUCK), et celle marquée APAPANE a le long bec courbé d'un IIWI. Sur ces photos, c'est peut-être le modèle qui a raison. Aucun modèle ne pourra jamais « réussir » une photo mal étiquetée : c'est un plafond pour tout le monde.

## Partir de zéro, ou partir d'un modèle qui sait déjà voir

Le modèle final ne part **pas** de zéro : il part d'un grand modèle déjà entraîné sur des centaines de millions d'images, et on lui apprend seulement les oiseaux. C'est le *transfer learning*, prévu par le cours (slides 82 et 88).

**L'image de l'enfant et de l'adulte.** Apprendre à un enfant à reconnaître 525 oiseaux alors qu'il n'a jamais rien vu de sa vie, c'est très long. Apprendre la même chose à un adulte qui sait déjà ce qu'est un bec, une plume, une aile, c'est beaucoup plus rapide. Le petit CNN, c'est l'enfant. Un modèle pré-entraîné, c'est l'adulte.

**Le vocabulaire.**

- **Backbone** (la « colonne vertébrale ») : la grosse partie du réseau qui transforme une image en un résumé de quelques centaines de nombres. Ce résumé s'appelle un **vecteur de caractéristiques** (*features*). Deux oiseaux qui se ressemblent ont des vecteurs proches.
- **Tête** (*head*) : une petite couche posée au bout, qui transforme le résumé en 525 scores. La tête d'origine du backbone ne connaît pas nos espèces : on en met une neuve.
- **Pré-entraîné** : les boutons du backbone ont déjà été réglés par d'autres (Meta, Microsoft, des universités) sur d'énormes collections d'images. On télécharge ces réglages.

**Deux façons de s'en servir, qu'on fait l'une après l'autre :**

1. **Feature extraction + linear probe** (étapes 4 et 5) : on **gèle** le backbone (on ne touche à aucun de ses boutons) et on n'entraîne que la tête. C'est rapide et ça mesure ce que le backbone sait déjà.
2. **Fine-tuning** (étape 6) : on dégèle tout et on ajuste **tous** les boutons, très doucement, pour les spécialiser sur nos oiseaux. C'est plus long, et ça donne le meilleur score.

**La preuve que ça marche.** Le plus petit des backbones testés, ResNet18, déjà entraîné sur ImageNet, gelé, avec une simple tête entraînée en quelques secondes, fait **92,5 %**. C'est 14 points de plus que notre petit CNN entraîné pendant une heure.

## Pourquoi une carte graphique louée, et comment elle a été installée

Les gros modèles demandent une carte graphique (GPU) : on en loue une, une NVIDIA L40S, sur RunPod à 1,09 $ de l'heure, pour environ 3 h 20 de pod en tout, soit environ 3,50 $.

**Processeur ou carte graphique ?** Le processeur (CPU) de ton ordinateur est comme 6 cuisiniers très malins : ils savent tout faire, mais un par un. Une carte graphique, c'est des milliers de commis qui font tous la même opération simple en même temps. Or un réseau de neurones, c'est surtout des millions de multiplications toutes pareilles. Le GPU va des dizaines de fois plus vite. Un modèle de 300 millions de boutons prendrait des jours sur ton ordinateur ; sur la L40S, environ une heure.

**Pourquoi le petit CNN est resté sur ton ordinateur.** Il ne prenait qu'une heure, gratuitement. Louer et installer un pod pour lui aurait coûté autant de temps qu'on en gagnait.

**Comment le pod a été installé.**

1. **Toi** : création du pod sur le site RunPod (L40S, image PyTorch 2.8 avec CUDA 12.8, disque système de 30 Go, disque de travail de 60 Go). Tu gardes la main sur tout ce qui coûte de l'argent.
2. **L'accès** : une clé SSH créée sur ton ordinateur et ajoutée à ton compte RunPod. Elle permet de piloter ce pod-là depuis ton ordinateur, sans donner accès au reste de ton compte. (Le connecteur officiel RunPod a d'abord été essayé, mais sa page de connexion avait un bug.)
3. **L'envoi des programmes**, dans un petit fichier zip de quelques Ko. Les photos ne sont pas envoyées depuis ton ordinateur : le pod les télécharge directement depuis Hugging Face, puis les range avec le même programme que sur ton ordinateur (mêmes 84 635 / 2 625 / 2 625 photos).
4. **Un souci corrigé** : Ubuntu 24.04 interdit par défaut d'ajouter des bibliothèques au Python du système, alors que c'est là que PyTorch est installé sur ces pods. On l'a autorisé explicitement.
5. **Le lancement** du calcul en arrière-plan : il continue même si ta connexion coupe. Si le pod redémarre, le programme reprend à la dernière étape terminée.

**Tous les programmes ont d'abord été testés sur ton ordinateur**, avec de petits modèles et 10 espèces seulement, pour ne pas payer une carte graphique à attendre qu'on corrige un bug.

**« Pourquoi ça n'utilise pas toutes les performances ? »** Le tableau de bord montrait la carte graphique à 100 % et 314 W sur 350 W : elle est utilisée à fond. Sa mémoire n'était qu'à 10 % parce que, pendant la comparaison, les modèles lisent seulement les images sans apprendre, ce qui demande peu de mémoire. Le processeur à 12 % prépare les photos plus vite que la carte ne les traite : c'est le bon sens, la carte graphique n'attend jamais.

## Étapes 4 et 5 : comparer 4 backbones sans les modifier

Les 4 backbones pré-entraînés ont été mis à l'épreuve dans les mêmes conditions, et le gagnant est DINOv2 : **gelé, il atteint 99,3 %**, contre 78,8 % pour le petit CNN.

**Les 4 candidats, et pourquoi eux.**

| Backbone | Taille | Appris sur | Pourquoi le tester |
|---|---|---|---|
| ResNet18 | 11 millions de boutons | 1,3 million de photos ImageNet, avec étiquettes | Le classique du cours : le point de comparaison simple |
| DINOv2 ViT-L | 300 millions | 142 millions de photos, **sans** étiquettes (Meta) | Très fort pour les détails fins (plumes, motifs) |
| EVA-02 Large | 300 millions | 2 milliards de photos avec leur légende (façon CLIP) | Un des meilleurs modèles généralistes |
| BioCLIP 2 | 300 millions | 214 millions de photos d'êtres vivants, avec leur nom d'espèce | Spécialiste du vivant : mon favori sur le papier |

**Comment on les compare : le linear probe.** Pour chaque backbone :

1. On le **gèle**, puis on lui fait lire chacune des 89 885 photos une seule fois. Pour chaque photo, on garde son vecteur de caractéristiques (son résumé). C'est la *feature extraction*.
2. Sur ces résumés, on entraîne une seule couche linéaire (la tête), en 40 epochs de quelques secondes. On essaie 3 forces de régularisation et on garde la meilleure *sur la validation*.
3. On note le score de validation.

**Pourquoi c'est une comparaison juste.** Seule la tête apprend, et c'est la même pour tous. La différence de score vient donc uniquement de la qualité du résumé que fait chaque backbone. C'est rapide (quelques minutes par modèle), et ça évite de payer quatre fine-tunings complets.

**Et le test ?** Les résumés des photos de test sont calculés et sauvegardés, mais **aucun score n'est calculé dessus**. Ils serviront seulement à chercher les photos presque identiques entre train et test.

**Résultats** (validation, 2 625 photos) :

| Rang | Backbone | Accuracy de validation | Top-5 | Erreurs sur 2 625 | Temps de lecture des photos |
|---|---|---|---|---|---|
| 1 | **DINOv2 ViT-L** | **99,28 %** | 99,92 % | environ 19 | 2 min 30 |
| 2 | BioCLIP 2 | 99,12 % | 99,96 % | environ 23 | 2 min 40 |
| 3 | EVA-02 Large | 98,90 % | 100,00 % | environ 29 | 5 min 15 |
| 4 | ResNet18 | 92,50 % | 98,63 % | environ 197 | 20 s |
| pour comparer | Petit CNN parti de zéro | 78,8 % | 92,5 % | environ 557 | 65 min d'entraînement |

**Comment lire ce résultat.** Les trois gros modèles sont tous au-dessus de 98,9 %, et leurs écarts (0,4 point au plus) sont de l'ordre de la marge d'erreur (± 0,4 point). On ne peut donc pas affirmer que DINOv2 est *vraiment* meilleur que BioCLIP 2 : ils sont à égalité, à la précision près de notre mesure. La règle fixée à l'avance était de prendre le meilleur score de validation : c'est DINOv2. Mon favori sur le papier, BioCLIP 2, n'a pas gagné, et c'est très bien ainsi : on choisit sur la mesure, pas sur l'intuition.

**Pourquoi DINOv2 est si fort ici.** Il a appris sans étiquettes, en devant reconnaître qu'un détail d'une image et l'image entière parlent de la même chose. Il a donc appris à faire très attention aux détails fins : motifs de plumes, forme du bec, couleur de l'œil. C'est exactement ce qui distingue deux espèces proches.

Le fine-tuning de DINOv2 a démarré à 14 h 59 (heure du serveur). Avant toute modification, la tête issue du linear probe donne bien 99,2 % : le départ est sain.

*(Le graphique de toutes les marches est dans le document en ligne et dans le notebook 03.)* Le vrai saut vient du passage à un modèle pré-entraîné (+ 14 points dès ResNet18). Au-dessus de 98,9 %, les différences entre modèles sont de l'ordre de la marge d'erreur.

## Étape 6 : le fine-tuning, la recette ingrédient par ingrédient

Le meilleur backbone de l'étape 5 est dégelé, et tous ses boutons sont ajustés très doucement pendant 12 epochs, avec une recette standard des meilleurs résultats publiés. Le programme est `scripts/finetune.py`, et il est choisi et lancé automatiquement.

Le principe : le modèle sait déjà très bien voir. On veut le spécialiser sur nos oiseaux **sans lui faire oublier** ce qu'il sait. Chaque ingrédient sert à ça.

| Ingrédient | Réglage | Explication simple |
|---|---|---|
| Départ de la tête | La tête apprise au linear probe | Une tête au hasard enverrait de grosses erreurs au début et « secouerait » tout le réseau. On part d'une tête qui fait déjà 99 %. Le programme vérifie ce score avant de commencer. |
| Optimiseur | AdamW | La méthode qui décide comment tourner les boutons. Adam adapte le pas à chaque bouton ; le « W » ajoute un petit frein (weight decay de 0,05) qui empêche les boutons de partir trop loin. |
| Pas d'apprentissage | 5e-5 en haut du réseau, 5e-4 pour la tête | Très petit pour le backbone (on ajuste, on ne réapprend pas), plus grand pour la tête (elle est neuve et petite). |
| Layer-wise lr decay | 0,85 par couche | Plus une couche est basse, moins on la touche. Les couches basses voient des bords et des textures : c'est pareil pour tous les oiseaux du monde, elles sont déjà bonnes. La couche du bas bouge environ 50 fois moins que celle du haut. |
| Échauffement (warmup) | 1 epoch | Le pas monte doucement de 0 à sa valeur, comme on s'échauffe avant de courir. |
| Calendrier cosinus | Jusqu'à l'epoch 12 | Le pas redescend ensuite doucement jusqu'à presque zéro : grandes corrections au début, retouches fines à la fin. |
| Augmentation | Recadrage aléatoire (50 à 100 % de l'image), miroir, petites variations de lumière, petits carrés effacés | Le modèle voit chaque photo un peu différente à chaque fois. **Pas** de changement de teinte ni de retournement haut-bas : la couleur du plumage est un vrai indice, et un oiseau n'est jamais à l'envers. |
| Label smoothing | 0,1 | On dit au modèle « c'est à 90 % cette espèce » plutôt que « 100 % ». Il devient moins arrogant, et une étiquette fausse fait moins de dégâts. |
| Drop path | 0,1 | Pendant l'entraînement, on saute parfois une couche entière au hasard. Le réseau ne dépend jamais d'un seul chemin. (Pas disponible pour BioCLIP 2.) |
| EMA | Moyenne glissante des boutons (0,9998) | En plus du modèle normal, on garde une copie « moyennée » dans le temps, plus calme et souvent un peu meilleure. On garde la meilleure des deux. |
| Paquets | 64 photos | Assez pour des calculs stables, et ça tient dans la mémoire de la carte. |
| Précision des calculs | bfloat16 | Des nombres un peu moins précis, mais deux fois plus rapides. La différence ne se voit pas sur le résultat. |
| Gradient clipping | 1,0 | Si une correction est anormalement grande, on la limite : pas d'accident. |

**Pourquoi 12 epochs ?** Un modèle pré-entraîné fait l'essentiel de ses progrès dans les 3 à 5 premières epochs. Au-delà, il risque d'apprendre par cœur et d'oublier ce qu'il savait. 12 est la valeur habituelle pour ce type de modèle et de dataset.

**Comment on choisit la version finale.** À la fin de chaque epoch, on mesure le score de validation (modèle normal et copie EMA). La meilleure version jusque-là est sauvegardée dans `best.pt`. Si le meilleur moment est l'epoch 8, c'est l'epoch 8 qu'on garde.

**Sécurité.** À chaque epoch, l'état complet est aussi sauvegardé (`last.pt`). Si le pod s'arrête, on relance la même commande et ça reprend.

**Ce qui a changé par rapport au premier plan.** Le premier document prévoyait un pas de 1e-5 et un decay de 0,75. Avec 24 couches, ça donnait aux couches du bas un pas d'environ 0,000000008, plus de 100 fois plus petit qu'avec la recette retenue : elles n'auraient presque pas bougé. On a pris 5e-5 et 0,85, les valeurs habituelles pour un ViT-Large.

**Ce qui s'est passé (DINOv2, 12 epochs de 8 min 30 sur la L40S).**

| Epoch | Modèle normal | Copie EMA |
|---|---|---|
| 1 | 97,22 % | 98,97 % |
| 2 | 98,67 % | 99,01 % |
| 4 | 98,86 % | 99,24 % |
| 6 | 99,16 % | 99,35 % |
| **8** | **99,43 %** | 99,39 % |
| 10 | 99,43 % | 99,43 % |
| 12 | 99,31 % | 99,35 % |

- **Epoch 1 : une secousse.** Le modèle tombe sous le score du modèle gelé, parce qu'on commence à modifier tout le réseau avec le pas le plus grand. La copie EMA, elle, encaisse la secousse.
- **Epochs 2 à 8 : remontée régulière**, jusqu'à 99,43 % (15 erreurs sur 2 625).
- **Epochs 9 à 12 : plateau.** Le programme a gardé l'epoch 8 automatiquement.
- **Gain du fine-tuning : + 0,15 point** sur le modèle gelé (4 erreurs de moins), dans la marge d'erreur. L'essentiel du travail était déjà fait par le pré-entraînement.

## Étapes 7 et 8 : mesurer, puis le test une seule fois

Le modèle final est mesuré d'abord sur la validation, puis une seule et unique fois sur le test, et c'est ce score de test qui sera la note officielle, même s'il déçoit (slide 85).

**Ce que fait le programme `evaluate.py`.**

- Il recharge le modèle sauvegardé, avec exactement le même prétraitement qu'à l'entraînement (même taille d'image, mêmes réglages de couleurs).
- Il calcule l'**accuracy** (la part de bonnes réponses), le **top-5** (la part de photos où la bonne espèce est dans les 5 premières propositions) et la marge d'erreur.
- Il le fait **sans et avec TTA**. La TTA (*test-time augmentation*) consiste à montrer au modèle la photo normale *et* son reflet dans un miroir, puis à faire la moyenne de ses deux avis. Comme demander deux fois à quelqu'un en lui montrant l'image sous deux angles.
- Il écrit, pour chaque photo, la vraie espèce, la réponse, la confiance et le top-5. C'est la matière première de l'analyse des erreurs.
- Il calcule le score de chaque espèce séparément, pour trouver celles qui posent problème.

**Le verrou du test.** La première évaluation sur le test crée un fichier `TEST_DEJA_UTILISE.txt`. Ensuite, le programme refuse de recommencer. Ce n'est pas qu'on ne pourrait pas tricher : c'est un rappel, écrit dans le code, que le test est un examen et pas un outil de réglage.

**Ordre des opérations.** On fige tout d'abord (le modèle, le prétraitement, le choix TTA ou non, décidé sur la validation). Ensuite seulement, on ouvre le test.

**Ce qui a été figé avant d'ouvrir le test** (fichier `CHOIX_FIGE.txt`) : les poids de l'epoch 8, **sans TTA**. Sur la validation, la TTA faisait 99,35 % contre 99,43 % sans : elle n'aidait pas.

| Tas | Accuracy | Marge (95 %) | Top-5 | Erreurs |
|---|---|---|---|---|
| Validation | 99,43 % | ± 0,29 | 100 % | 15 sur 2 625 |
| **Test (une seule fois)** | **99,43 %** | **± 0,29** | **100 %** | **15 sur 2 625** |

Le score de test est identique à celui de validation : le modèle n'a pas été réglé « pour » la validation, il se comporte pareil sur des photos jamais vues. Rechargé sur ton ordinateur, sans carte graphique, le modèle donne les mêmes réponses (environ 1 minute de chargement, puis environ 1 seconde par photo).

## Analyse des erreurs et limites

Un score de 99 % ne veut rien dire tout seul : l'analyse doit expliquer où le modèle se trompe, pourquoi, et dans quelles conditions il ne marcherait plus (slides 83, 86 et 87). Elle est faite dans le notebook `03_resultats.ipynb`, sur ton ordinateur, en regardant chacune des 30 erreurs (15 en validation, 15 en test).

**Ce qu'on a trouvé sur les 15 erreurs du test :**

| Type d'erreur | Nombre | Exemples | Ce que ça veut dire |
|---|---|---|---|
| Étiquette du dataset probablement fausse | 3 | « TEAL DUCK » qui montre un canard d'Amérique ; « BALTIMORE ORIOLE » dont **la même photo** est dans le train sous un autre nom ; « MCKAYS BUNTING » (un bruant blanc) qui montre un oiseau brun rayé | Le modèle a probablement raison. Personne ne peut « réussir » ces photos. |
| Classe mal constituée | 3 | AUSTRAL CANASTERO : 3 photos de test sur 5 ratées, et ses photos de train mélangent des oiseaux d'allures très différentes | C'est la classe qui est incohérente, pas le modèle. |
| Espèces très proches | 6 | Deux touracos confondus dans les deux sens, deux tragopans, deux martins-pêcheurs, deux trogons, deux colibris | Des erreurs « honnêtes », qu'un amateur ferait aussi. |
| Photo difficile | 3 | Un jeune encore duveteux, un oiseau en vol vu de dessous, un cormoran au plumage inhabituel | Des poses et des âges rares dans le train, qui montre surtout des adultes posés de profil. |

Environ **6 erreurs sur 15 viennent du dataset lui-même**, pas du modèle. Sans elles, le score serait proche de 99,7 %. En validation, on retrouve 3 étiquettes probablement fausses (APAPANE, TEAL DUCK, EUROPEAN GOLDFINCH) et des espèces proches (PURPLE FINCH confondu 3 fois avec HOUSE FINCH).

**Les quasi-doublons entre train et test.** Pour chaque photo de test, on a cherché la photo du train la plus ressemblante.

- **Il y a de vraies fuites.** 131 photos de test (5 %) ont dans le train **la même photo**, juste recadrée ou décalée. Le dataset ne respecte donc pas la règle des groupes de la slide 79.
- **Mais elles ne gonflent pas le score.** Sur ces photos, l'accuracy n'est pas meilleure (96,9 %). Sur toutes les photos sans « jumeau » proche, elle reste à **99,5 %**. Le modèle réussit aussi bien sur des photos vraiment nouvelles : le score est crédible.

**Le modèle sait-il quand il ne sait pas ?** Plutôt, oui. Sa confiance moyenne est de 89 % quand il a raison, et de 53 % quand il se trompe. Quand il est sûr à plus de 50 %, il a raison dans 99,7 % des cas.

**La prochaine expérience** (une seule modification justifiée, comme le demande le cours) : nettoyer le dataset (retirer les doublons train/test, corriger les étiquettes fausses, vérifier la classe AUSTRAL CANASTERO), puis réentraîner et mesurer sur un nouveau test indépendant, et sur nos propres photos.

**Ce qu'on a regardé.**

1. **Les erreurs les plus sûres d'elles.** Comme pour le petit CNN : la photo, la réponse, et une photo de l'espèce répondue. Est-ce une espèce proche ? Une étiquette fausse ? Un décor trompeur ?
2. **Les étiquettes fausses.** On compte combien d'erreurs viennent en réalité d'une photo mal étiquetée (on en connaît déjà 2). Elles fixent un plafond que personne ne peut dépasser.
3. **Les photos presque identiques entre train et test.** Avec les résumés calculés à l'étape 5, on cherche les paires de photos qui se ressemblent à plus de 97 %. Si beaucoup de photos de test ont un « jumeau » dans le train, le score est trop optimiste : le modèle les reconnaît de mémoire. C'est le problème des groupes de la slide 79.
4. **Les biais du dataset.** Beaucoup de photos montrent des mâles, souvent plus colorés que les femelles. Le modèle pourrait mal reconnaître une femelle ou un jeune.
5. **Les faux indices.** Le filigrane « alamy », le fond, la taille : le modèle s'en sert-il ?
6. **Tes propres photos.** Le vrai test de la vie réelle : des photos prises par toi ou trouvées ailleurs, d'oiseaux pas forcément centrés, flous, lointains. C'est là que le score baissera, et c'est normal : ce dataset contient surtout de belles photos, bien cadrées.

**Limites déjà connues.**

- **525 espèces seulement.** Un oiseau d'une autre espèce recevra quand même l'un des 525 noms. Le programme prévient quand il est peu sûr de lui (moins de 50 %).
- **Des photos « de catalogue ».** Un seul oiseau, centré, net. Une photo de mangeoire avec trois oiseaux flous est un autre problème.
- **Le pré-entraînement a peut-être déjà vu certaines photos.** Les gros modèles ont appris sur des centaines de millions d'images d'Internet, et ce dataset vient lui aussi d'Internet. Si certaines photos de test y étaient, le score serait un peu flatté. On ne peut pas le vérifier complètement : on le dit honnêtement.

## Les livrables et ce qui reste à faire

Tous les livrables techniques et la fiche projet sont faits ; il reste la présentation, la question au professeur et le test sur tes propres photos (slide 86).

| Livrable demandé | Fichier | État |
|---|---|---|
| Dataset documenté | `README.md`, `metadata.csv`, `classes.json` | Fait |
| Notebook exécutable : exploration et baseline | `01_exploration_baseline.ipynb` | Fait |
| Notebook exécutable : petit CNN | `02_petit_cnn.ipynb` | Fait |
| Notebook des résultats GPU (comparaison, fine-tuning, test, analyse) | `03_resultats.ipynb` | Fait |
| Journal des essais | Tableau dans `README.md` | Fait |
| Modèle rechargeable (poids, prétraitement, classes) | `best.pt` | Fait (600 Mo, sur ton ordinateur seulement : trop gros pour GitHub) |
| Programme qui reconnaît un oiseau sur une photo | `scripts/predict.py` | Prêt, testé |
| Fiche projet | `FICHE_PROJET.md` | Fait |
| Présentation, avec effectifs et pourcentages | à créer | À faire |

**Prochaines actions, dans l'ordre :**

- [x] Attendre la fin du fine-tuning (environ 1 h 50)
- [x] Lire le score de validation, puis lancer le test une seule fois
- [ ] Rapatrier les résultats sur ton ordinateur et supprimer le pod
- [ ] Demander au professeur si un dataset déjà existant est accepté
- [x] Écrire la fiche projet
- [x] Faire l'analyse des erreurs et le notebook 03
- [ ] Tester le modèle sur tes propres photos
- [ ] Préparer la présentation

## Le petit dictionnaire

| Mot | Ce que ça veut dire |
|---|---|
| Accuracy | La part de photos où le modèle donne la bonne espèce |
| Augmentation | Modifier un peu les photos à chaque passage (recadrer, retourner) pour que le modèle n'apprenne pas par cœur |
| Backbone | La grosse partie du réseau qui résume une image en quelques centaines de nombres |
| Baseline | Le score de la règle la plus simple, qui sert de plancher |
| Batch | Un paquet de photos traitées ensemble |
| CNN | Réseau à convolutions : fait glisser de petites loupes sur l'image |
| Epoch | Un tour complet de toutes les photos d'entraînement |
| EMA | Une copie des boutons moyennée dans le temps, plus calme |
| Feature extraction | Faire lire les photos à un backbone gelé et garder ses résumés |
| Fine-tuning | Ajuster doucement tous les boutons d'un modèle pré-entraîné |
| GPU | Carte graphique : des milliers de petits calculs en même temps |
| Label smoothing | Dire « c'est à 90 % cette espèce » plutôt que « 100 % » |
| Learning rate | De combien on tourne les boutons à chaque correction |
| Linear probe | N'entraîner qu'une couche sur les résumés d'un backbone gelé |
| Loss | La note d'erreur du modèle, plus elle est basse mieux c'est |
| Poids | Les « boutons » du réseau, réglés pendant l'apprentissage |
| Pré-entraîné | Déjà entraîné par d'autres sur une énorme collection d'images |
| Sur-apprentissage | Apprendre les photos par cœur au lieu de comprendre |
| Tête (head) | La petite couche finale qui donne un score par espèce |
| Top-5 | La part de photos où la bonne espèce est dans les 5 premières propositions |
| Transfer learning | Réutiliser un modèle pré-entraîné comme point de départ |
| TTA | Faire la moyenne des avis du modèle sur la photo et son reflet |
| ViT | Vision Transformer : découpe l'image en carrés qui se regardent les uns les autres |
