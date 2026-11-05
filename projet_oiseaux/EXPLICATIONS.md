# Projet BirdsAI : tout comprendre, étape par étape

5 novembre 2026

## Le but et la carte du voyage

On va fabriquer un programme qui regarde la photo d'un oiseau et dit de quelle espèce il s'agit, parmi 525 espèces, en visant environ 99,5 % de bonnes réponses.

Imagine un enfant à qui on montre des milliers de cartes d'oiseaux avec leur nom au dos. Au début il devine au hasard. À force de voir les cartes et de se faire corriger, il finit par reconnaître les oiseaux tout seul. Un modèle d'intelligence artificielle apprend exactement comme ça : en voyant des exemples, en se trompant et en se corrigeant.

Le voyage se fait en 10 étapes, toujours dans le même ordre :

1. **Préparer les images** : les sortir de leurs gros fichiers et les ranger dans des dossiers, un dossier par espèce.
2. **Explorer** : regarder les images et compter, pour savoir à quoi on a affaire.
3. **La baseline** : mesurer le score d'une méthode « bête », pour avoir un point de comparaison.
4. **Le petit CNN maison** : entraîner un petit réseau à partir de zéro, comme dans le TP7, pour voir ses limites.
5. **Comparer des cerveaux déjà entraînés** : tester plusieurs modèles célèbres qui savent déjà « voir », et garder le meilleur pour les oiseaux.
6. **Louer un GPU** : un ordinateur très puissant sur RunPod, pour quelques heures.
7. **Le fine-tuning** : réentraîner doucement le meilleur cerveau pour qu'il devienne un expert des oiseaux.
8. **Le test final** : mesurer le score une seule fois, sur des images jamais utilisées.
9. **Comprendre les erreurs** : regarder où et pourquoi le modèle se trompe.
10. **Livrer** : un modèle qu'on peut recharger, une fonction qui donne la réponse, un notebook et un README.

Chaque étape est expliquée en détail dans la suite, avec le *pourquoi*. Les mots techniques en **gras** sont tous repris dans le petit dictionnaire à la fin.

## Les bases : comment une machine apprend

Une machine n'apprend pas comme par magie : elle ajuste des millions de petits boutons jusqu'à ce que ses réponses soient justes.

### Une image, pour un ordinateur, c'est un tableau de nombres

Une photo est découpée en petits carrés appelés **pixels**. Nos images font 224 pixels de large sur 224 de haut. Chaque pixel a 3 nombres de 0 à 255 : la quantité de rouge, de vert et de bleu. Ce sont les 3 **canaux**.

Une image d'oiseau, c'est donc 224 × 224 × 3 = 150 528 nombres. L'ordinateur ne voit jamais un « oiseau » : il voit seulement ces nombres. Dans PyTorch, on range une image sous la forme `[3, 224, 224]` (canaux, hauteur, largeur), et un paquet de 64 images sous la forme `[64, 3, 224, 224]`. C'est ce que montre la slide 5 du cours.

### Un modèle, c'est une machine à calculer avec des boutons

Un **modèle** prend les 150 528 nombres en entrée et sort 525 nombres, un par espèce. Plus le nombre d'une espèce est grand, plus le modèle pense que c'est elle. Ces 525 nombres bruts s'appellent des **logits**. Une fonction appelée **softmax** les transforme en pourcentages qui font 100 % au total.

Entre l'entrée et la sortie, le modèle fait des multiplications et des additions avec ses **poids** (en anglais *weights*, ou *parameters*). Imagine une table de mixage avec des millions de boutons. Si les boutons sont bien réglés, la sortie est la bonne espèce. Au départ, ils sont réglés au hasard : le modèle répond n'importe quoi.

### Apprendre, c'est tourner les boutons dans le bon sens

L'apprentissage (l'**entraînement**, ou *training*) répète une petite boucle des milliers de fois :

1. On montre une image au modèle. Il donne sa réponse.
2. On calcule à quel point il s'est trompé. Ce chiffre s'appelle la **loss** (la « perte »). Une loss à 0, c'est parfait. Une grosse loss, c'est une grosse erreur. Pour nous, ce sera la **CrossEntropyLoss**, celle du TP7 : elle punit fort quand le modèle est sûr de lui et se trompe.
3. On calcule, pour chaque bouton, dans quel sens le tourner pour que la loss baisse un peu. C'est le **gradient**. PyTorch le calcule tout seul avec `loss.backward()`.
4. On tourne chaque bouton d'un tout petit peu dans ce sens. C'est la **descente de gradient** (*gradient descent*), faite par l'**optimiseur** (`opt.step()`).

Pense à quelqu'un perdu dans le brouillard sur une montagne, qui veut descendre dans la vallée. Il ne voit rien, mais il sent la pente sous ses pieds. À chaque pas, il avance un peu vers le bas. La vallée, c'est la loss la plus basse possible. La taille de ses pas, c'est le **learning rate** : trop grands, il saute par-dessus la vallée ; trop petits, il met une éternité.

### Batch et epoch

On ne montre pas les images une par une, ce serait trop lent. On les montre par paquets, appelés **batchs** (par exemple 64 images). Le modèle fait la moyenne de ses erreurs sur le paquet, puis tourne ses boutons une fois.

Quand le modèle a vu toutes les images d'entraînement une fois, c'est une **epoch**. Avec 84 635 images et des batchs de 64, une epoch = environ 1 323 petits ajustements. On fait en général 10 à 15 epochs : le modèle revoit chaque image 10 à 15 fois.

## Les données : 525 espèces, trois tas séparés

Le dataset contient 89 885 photos d'oiseaux de 525 espèces, déjà réparties en trois tas qu'on ne mélangera jamais.

Il vient de Kaggle (« BIRDS 525 SPECIES »). Chaque photo fait 224 × 224 pixels en couleur, avec en général un seul oiseau, bien centré, qui prend une bonne partie de l'image. Les images sont stockées dans 6 fichiers `.parquet` (environ 2 Go) : c'est un format qui range beaucoup de données dans peu de fichiers, comme une grosse boîte d'archives.

| Tas | Nombre d'images | Par espèce | À quoi il sert |
| --- | --- | --- | --- |
| **Train** | 84 635 | 130 à 263 | Apprendre : le modèle tourne ses boutons en regardant ces images |
| **Validation** | 2 625 | 5 | Choisir : comparer les modèles et les réglages |
| **Test** | 2 625 | 5 | Juger : le score final, mesuré une seule fois |

### Pourquoi trois tas et pas un seul ?

Prends un élève qui prépare un examen :

- Le **train**, ce sont ses exercices avec les corrigés. Il apprend dessus.
- La **validation**, ce sont les examens blancs. Ils lui disent quelle méthode de révision marche le mieux. Il peut en passer plusieurs.
- Le **test**, c'est le vrai examen. On ne le passe qu'une fois, à la fin.

Si l'élève avait les sujets du vrai examen pendant ses révisions, il pourrait les apprendre par cœur et avoir 20/20 sans rien comprendre. Sa note ne voudrait plus rien dire. C'est pareil pour un modèle : s'il a vu une image pendant l'entraînement, le réussir sur cette image ne prouve rien.

Même chose pour la validation : à force de choisir les réglages qui marchent le mieux dessus, on finit par « s'adapter » un peu à elle. C'est pour ça qu'on garde un test qu'on n'a jamais regardé. Le cours le dit à la slide 85 : « le test se consulte une fois : c'est ce qui rend son résultat crédible ».

### Ce qui rend ce dataset difficile, et ce qui le rend facile

- **Difficile** : 525 espèces, c'est beaucoup, et certaines se ressemblent énormément (plusieurs moineaux, plusieurs canards, plusieurs colibris). On appelle ça une classification **fine** (*fine-grained*). Un modèle qui répond au hasard aurait 1 chance sur 525, soit 0,19 %.
- **Facile** : les photos sont propres. L'oiseau est grand, centré, net. Dans la vraie vie, un oiseau est souvent petit, caché dans les branches, flou ou à contre-jour. Le score qu'on obtiendra sera donc meilleur que ce qu'on aurait sur des photos prises au hasard. On le vérifiera à la fin avec tes propres photos.

## Étapes 0 et 1 : préparer et explorer les données

Avant d'entraîner quoi que ce soit, on range les images dans des dossiers et on les regarde, parce qu'une erreur dans les données gâche tout le reste.

### Étape 0 : sortir les images des fichiers parquet

Chaque ligne d'un fichier parquet contient une image (sous forme de fichier JPEG compressé) et son numéro d'espèce (de 0 à 524). Un petit script va :

1. Lire les 6 fichiers parquet avec la bibliothèque `pandas`.
2. Écrire chaque image dans un dossier au nom de son espèce, par exemple `data/train/BALD EAGLE/00001.jpg`.
3. Créer un fichier `metadata.csv` qui liste, pour chaque image : son chemin, son espèce et son tas.

Pourquoi des dossiers ? Parce que c'est la structure demandée par le cours (slide 80), et que l'outil `ImageFolder` de PyTorch la comprend tout seul (slide 81) : il devine l'espèce d'une image grâce au nom de son dossier. C'est aussi plus facile pour toi d'ouvrir un dossier et de regarder les photos.

### Étape 1 : explorer

On regarde les données avant de leur faire confiance. Concrètement :

- **Compter les images par espèce.** Le train est déséquilibré : certaines espèces ont environ 130 images, d'autres près du double. Un modèle a tendance à favoriser les espèces qu'il a vues le plus souvent. Il faut le savoir pour interpréter les erreurs.
- **Afficher des exemples** de plusieurs espèces, pour voir à quoi ressemblent les photos et repérer des bizarreries (une image vide, un mauvais label, un oiseau coupé).
- **Vérifier les dimensions** : que chaque batch a bien la forme `[64, 3, 224, 224]` et que les numéros d'espèces vont bien de 0 à 524.
- **Vérifier que les trois tas ont les mêmes 525 espèces**, dans le même ordre. Sinon, le numéro 12 pourrait vouloir dire « aigle » dans un tas et « canard » dans l'autre.

C'est ce que le cours appelle « des labels cohérents » (slide 78) : une erreur de label, c'est une mauvaise consigne donnée au modèle.

## Étape 2 : la baseline, le point de départ le plus bête

La baseline répond toujours la même espèce et obtient environ 0,19 % : c'est le score à battre, et il montre ce que vaut « ne rien savoir ».

Une **baseline**, c'est la méthode la plus simple qu'on puisse imaginer. Ici : « répondre toujours l'espèce qui a le plus d'images dans le train ». Comme la validation et le test ont 5 images par espèce, cette réponse unique est juste 5 fois sur 2 625, soit 0,19 %.

Pourquoi perdre du temps avec ça ? Parce qu'un score n'a de sens que comparé à quelque chose. Dire « mon modèle a 60 % » ne veut rien dire tout seul. Avec 2 espèces, 60 % serait médiocre (le hasard donne 50 %). Avec 525 espèces, 60 % serait déjà 300 fois mieux que le hasard.

La baseline sert aussi de **détecteur de bug**. Si un vrai modèle fait à peu près le même score que la baseline, c'est presque toujours qu'il y a une erreur dans le code (labels mélangés, images mal chargées), pas que le modèle est nul. Le cours la demande explicitement (slides 71 et 82).

## Étape 3 : le petit CNN fait maison

On entraîne un petit réseau à partir de zéro, comme au TP7, et on s'attend à un score modeste : c'est voulu, ça prouve pourquoi on a besoin de l'étape suivante.

### La convolution : une loupe qu'on promène sur l'image

Un **CNN** (*Convolutional Neural Network*) regarde l'image par petits morceaux. Imagine une loupe carrée de 3 × 3 pixels qu'on fait glisser partout sur la photo. À chaque position, la loupe se demande : « est-ce que je vois mon motif ici ? ». Cette loupe, c'est un **filtre** (ou *kernel*), et l'opération s'appelle une **convolution** (slides 10 à 13 du cours).

Le point clé : on utilise **la même loupe partout**. Si elle sait reconnaître un bord de plume en haut à gauche, elle le reconnaît aussi en bas à droite. C'est pour ça qu'un CNN a besoin de beaucoup moins de boutons qu'un MLP, et qu'il résiste mieux quand l'oiseau est décalé (slides 8, 9 et 24).

On empile plusieurs couches de convolutions :

- Les premières couches repèrent des choses simples : des bords, des taches de couleur.
- Les suivantes combinent ces bords en textures : des rayures, des plumes, des écailles.
- Les dernières combinent les textures en parties : un bec, un œil, une aile (slide 34).

Entre les couches, le **pooling** rétrécit l'image : il garde le plus important de chaque zone de 2 × 2 pixels. Ainsi, les couches profondes voient une partie plus grande de l'oiseau.

### Pourquoi il va mal marcher ici

- **Pas assez d'exemples pour partir de zéro.** Avec environ 160 photos par espèce, un réseau qui ne sait même pas ce qu'est un bord doit tout découvrir seul : les bords, les plumes, puis la différence entre 525 espèces. C'est comme apprendre à lire et passer un examen d'ornithologie le même jour.
- **Ton ordinateur est lent pour ça.** Sur ton processeur, sans carte graphique, il faut réduire les images à 64 × 64 pixels pour que ça tourne en un temps raisonnable. À cette taille, beaucoup de détails du plumage disparaissent.

Je m'attends à un score entre 30 et 50 %. C'est une estimation, on le mesurera. C'est déjà plus de 150 fois mieux que la baseline, mais très loin de 99 %. Cette étape sert à le **prouver avec un chiffre** dans ton rapport, plutôt qu'à l'affirmer.

## Le transfer learning : réutiliser un cerveau qui sait déjà voir

Plutôt que de partir de zéro, on prend un modèle déjà entraîné sur des millions d'images et on lui apprend seulement à nommer nos oiseaux : c'est ce qui permet de passer de 40 % à 99 %.

### L'idée

Un enfant qui sait déjà regarder le monde (formes, couleurs, textures, « ça c'est un bec ») apprend très vite à reconnaître des espèces d'oiseaux. Un bébé qui ouvre les yeux pour la première fois, non. Le **transfer learning**, c'est réutiliser un modèle qui « sait déjà regarder » (slides 33 et 34).

Un tel modèle a été entraîné par une grande entreprise ou un laboratoire, sur des millions d'images, avec des centaines de cartes graphiques pendant des semaines. Il a déjà appris à repérer les bords, les textures et les formes. On télécharge ses poids gratuitement : on dit qu'il est **préentraîné** (*pretrained*).

### Les deux morceaux d'un modèle

- Le **backbone** (la « colonne vertébrale ») : la grosse partie qui regarde l'image et la transforme en une liste de nombres qui la résume, par exemple 1 024 nombres. Cette liste, c'est l'**embedding** (slides 44 à 46). Deux photos d'oiseaux qui se ressemblent auront des embeddings proches.
- La **tête** (*head*) : une petite couche à la fin, qui lit l'embedding et dit quelle espèce c'est. La tête d'origine connaît d'autres catégories que nos oiseaux. On la jette et on en met une neuve, avec 525 sorties.

### Deux façons de s'en servir

|  | Feature extraction | Fine-tuning |
| --- | --- | --- |
| Ce qu'on entraîne | Seulement la tête neuve | La tête et le backbone, doucement |
| Le backbone | **Gelé** : ses boutons ne bougent plus | **Dégelé** : ses boutons bougent un tout petit peu |
| Analogie | Un expert qui garde ses yeux et apprend juste les noms | Un expert qui apprend les noms et affine aussi son regard pour les plumes |
| Coût | Quelques minutes | Quelques heures sur un GPU |
| Score attendu ici | Très bon | Le meilleur possible |
| Slide du cours | 37 et 39 | 38 et 39 |

On va utiliser **les deux**, chacune pour une raison différente :

1. **La feature extraction pour comparer les backbones.** C'est rapide et peu cher. Elle nous dit quel cerveau de départ comprend le mieux les oiseaux.
2. **Le fine-tuning pour le modèle final.** On prend le gagnant et on l'affine en entier. Les oiseaux sont une tâche très fine : pour distinguer deux moineaux, le modèle gagne à ajuster son regard, pas seulement sa tête. Le cours le dit : le fine-tuning convient quand on a « plus de données, ou une tâche un peu différente » (slide 39). Nos 84 635 images, c'est beaucoup.

## Choisir le cerveau de départ

On met plusieurs backbones célèbres en compétition sur la validation, et on garde celui qui reconnaît le mieux les oiseaux. Mon favori est BioCLIP 2, mais c'est la mesure qui décidera.

### Deux grandes familles

- Les **CNN**, comme **ResNet** (celui du cours, slide 35) : ils regardent l'image avec des loupes, comme expliqué à l'étape 3, mais avec des dizaines de couches.
- Les **Vision Transformers** (**ViT**) : ils découpent l'image en petits carrés de 14 × 14 pixels, appelés **patchs**, comme un puzzle. Chaque pièce « discute » avec toutes les autres grâce à l'**attention**, le mécanisme vu dans la partie LLM du cours (slides 52 à 55). Le patch du bec peut ainsi regarder directement le patch de la queue, même s'ils sont loin. C'est très utile pour les oiseaux, où une espèce se reconnaît souvent à une combinaison de détails dispersés.

Les meilleurs modèles récents pour reconnaître des images sont presque tous des ViT. Le mot **Large** (ViT-L) désigne la grande taille : environ 300 millions de boutons, contre 11 millions pour ResNet18.

### Les candidats

| Backbone | Appris sur | Pourquoi le tester |
| --- | --- | --- |
| ResNet18 | 1,3 million d'images ImageNet, 1 000 catégories | Le modèle du cours : la référence. ImageNet ne contient qu'une soixantaine d'oiseaux, donc il devrait être le moins bon. |
| DINOv2 ViT-L/14 (Meta) | 142 millions d'images, **sans labels** | Il a appris à voir tout seul, en comparant des images entre elles. Ses embeddings sont parmi les meilleurs pour distinguer des choses très proches. |
| EVA-02 Large | Des millions d'images, puis ImageNet | L'un des meilleurs scores connus en fine-tuning. |
| BioCLIP 2 | Des centaines de millions de photos d'êtres vivants, avec leur nom d'espèce | Il est spécialisé dans le vivant : il a déjà vu énormément d'oiseaux et appris à les nommer. C'est mon favori. |

### Comment on les compare, équitablement

Pour chaque backbone, on fait la même chose :

1. On passe toutes les images (train et validation) une fois dans le backbone gelé, et on garde les embeddings sur le disque.
2. On entraîne une tête neuve sur les embeddings du train. Ça prend quelques minutes.
3. On mesure l'accuracy sur la validation.

Mêmes images, même tête, même mesure : seul le backbone change. C'est le même principe que le « même budget » du TP7 pour comparer MLP et CNN. Le gagnant est celui qui a la meilleure accuracy de validation, et c'est lui qu'on affinera ensuite.

Un détail important : chaque backbone attend que les images soient préparées d'une façon précise (taille, moyenne et écart-type des couleurs). C'est le **preprocessing**. Il faut utiliser exactement celui avec lequel il a été entraîné, sinon ses résultats s'effondrent (slides 36 et 40).

## Le GPU et RunPod

On loue une carte graphique L40S sur RunPod pour environ 1,09 $ de l'heure, et le projet devrait coûter autour de 4 à 5 $ avec un seul modèle.

### Pourquoi ton ordinateur ne suffit pas

Ton ordinateur a un **CPU** (le processeur, un Ryzen 5 4650U) mais pas de carte graphique utilisable pour l'IA. Un CPU, c'est comme quelques professeurs de maths très forts : ils font n'importe quel calcul, mais un à la fois. Un **GPU** (carte graphique), c'est comme une salle de 10 000 élèves qui font chacun une petite multiplication en même temps.

Or un réseau de neurones, c'est surtout des milliards de petites multiplications indépendantes. Le GPU les fait toutes en même temps : il va des dizaines de fois plus vite. Le fine-tuning d'un ViT-Large sur 84 635 images prendrait plusieurs jours sur ton ordinateur, contre quelques heures sur un GPU.

La quantité de mémoire du GPU s'appelle la **VRAM**. Elle doit contenir le modèle, ses gradients et un batch d'images en même temps. Un ViT-Large tient confortablement dans 48 Go.

### Pourquoi la L40S

- **48 Go de VRAM** : assez pour un ViT-Large avec de bons batchs, sans astuce compliquée.
- **16 cœurs de CPU** : pendant que le GPU calcule, le CPU doit décompresser les images suivantes. Avec trop peu de cœurs, le GPU attend sans rien faire.
- **62 Go de RAM** : on peut garder les 84 635 images déjà décompressées en mémoire (environ 13 Go), ce qui évite de les relire à chaque epoch.
- **1,09 $ de l'heure** : les cartes plus puissantes (H100, B200) vont plus vite, mais coûtent 3 à 8 fois plus cher pour un modèle identique. Notre dataset est trop petit pour en profiter.

### Combien ça coûte

| Étape | Durée estimée | Coût estimé |
| --- | --- | --- |
| Installer le pod et vérifier que tout tourne | environ 1 h | environ 1 $ |
| Comparer les backbones | environ 1 h | environ 1 $ |
| Fine-tuning du meilleur | 2 à 3 h | 2 à 3 $ |
| **Total** | **4 à 5 h** | **environ 4 à 5 $** |

Ce sont des estimations : on mesurera la durée d'une epoch avant de lancer l'entraînement complet. Prévois une marge si un essai doit être relancé. Le stockage du disque persistant coûte en plus quelques dollars par mois, tant qu'on ne le supprime pas.

### Comment ça se passe, concrètement

1. **Tout le code est écrit et testé sur ton ordinateur**, sur un petit morceau des données. On ne paie pas le GPU pour chercher des bugs.
2. Tu crées un **pod** sur RunPod : c'est un ordinateur loué à distance, avec PyTorch déjà installé. Tu lui ajoutes un **volume persistant** d'environ 50 Go : un disque qui garde les fichiers même quand le pod est arrêté.
3. On y envoie les fichiers parquet et le code.
4. On lance l'entraînement dans `tmux`, un outil qui le laisse tourner même si tu fermes ta connexion. Le script enregistre régulièrement le modèle (des **checkpoints**) pour ne rien perdre en cas de coupure.
5. On récupère le modèle final sur ton ordinateur.
6. **Tu arrêtes le pod tout de suite.** Il est facturé à l'heure tant qu'il est allumé, même s'il ne fait rien.

C'est toi qui loues et paies le pod : je prépare tout, mais je ne le lancerai pas à ta place.

## Le fine-tuning en détail : chaque réglage

Le fine-tuning, c'est réapprendre doucement à un cerveau déjà très bon, sans lui faire oublier ce qu'il sait. Chaque réglage ci-dessous sert l'un de ces deux buts : bien apprendre les oiseaux, ou ne pas tout casser.

Le grand danger s'appelle l'**overfitting** (le surapprentissage) : le modèle apprend les photos du train par cœur au lieu de comprendre les espèces. Il a alors un score parfait sur le train et moins bon sur des photos nouvelles. Plusieurs réglages servent à l'en empêcher.

### L'optimiseur AdamW : celui qui tourne les boutons

L'**optimiseur** décide de combien tourner chaque bouton après chaque batch. **Adam**, celui du cours, adapte la taille du pas pour chaque bouton : un bouton qui bouge toujours dans le même sens prend de l'élan, un bouton qui hésite avance prudemment. **AdamW** ajoute le **weight decay** : à chaque pas, il ramène très légèrement tous les boutons vers zéro. Ça empêche le modèle de s'appuyer énormément sur quelques détails, et ça limite l'overfitting. C'est l'optimiseur standard pour les ViT.

### Le learning rate : la taille des pas

C'est le réglage le plus important. On en utilise deux, comme à la slide 38 :

- **Petit pour le backbone** (environ 0,00001, noté 1e-5). Il sait déjà très bien voir. De grands pas lui feraient oublier ce qu'il sait : c'est l'**oubli catastrophique**. On le retouche avec un pinceau fin.
- **Plus grand pour la tête** (environ 0,001, noté 1e-3). Elle est neuve et ne sait rien : elle doit apprendre vite.

### Le layer-wise learning rate decay : protéger les premières couches

On va plus loin que la slide 38. Les premières couches repèrent des bords et des couleurs : c'est utile pour toutes les images, il n'y a presque rien à changer. Les dernières couches combinent ces motifs en choses plus spécifiques : c'est là qu'il faut apprendre les plumes de nos 525 espèces.

Donc chaque couche reçoit un learning rate un peu plus petit que celle d'après, par exemple 0,75 fois. La dernière couche bouge le plus, la première presque plus. C'est le **layer-wise learning rate decay**. C'est comme la slide 38, mais en plus fin : au lieu de « gelé » ou « dégelé », chaque couche est « plus ou moins dégelée ».

### Le warmup et la courbe cosinus : accélérer puis ralentir

Le learning rate ne reste pas fixe pendant tout l'entraînement :

1. **Warmup (échauffement)** : pendant la première epoch, il part de presque zéro et monte doucement. Au début, la tête neuve est réglée au hasard et produit des gradients désordonnés. Sans échauffement, ces gradients pourraient abîmer le backbone dès les premiers pas.
2. **Décroissance en cosinus** (*cosine schedule*) : ensuite, il redescend doucement jusqu'à presque zéro, en suivant la forme d'une vague. Au début on fait de grands pas pour avancer vite, à la fin de tout petits pas pour se poser précisément au fond de la vallée.

C'est comme se garer : on roule normalement dans le parking, puis on ralentit de plus en plus pour s'arrêter pile au bon endroit.

### Le bf16 : calculer avec des nombres plus courts

Par défaut, chaque nombre est stocké avec beaucoup de chiffres après la virgule (32 bits). Le **bf16** utilise des nombres deux fois plus courts (16 bits). C'est comme calculer avec 3 chiffres après la virgule au lieu de 7 : un peu moins précis, mais le réseau n'en a pas besoin. En échange, le GPU calcule environ 2 fois plus vite et utilise moins de VRAM. Concrètement : moins d'heures facturées pour le même score.

### La data augmentation : montrer des variantes

Pendant l'entraînement, chaque image est légèrement modifiée au hasard, différemment à chaque epoch (slide 28) :

- **RandomResizedCrop** : on prend un morceau au hasard de l'image (entre la moitié et la totalité) et on le remet à la bonne taille. Le modèle apprend à reconnaître un oiseau zoomé ou un peu coupé.
- **Flip horizontal** : on retourne l'image comme dans un miroir. Un oiseau qui regarde à gauche est la même espèce que s'il regardait à droite. On ne fait **pas** de flip vertical : un oiseau à l'envers n'existe presque jamais dans la vraie vie.
- **TrivialAugment**, en léger : de petites variations de luminosité, de contraste ou de netteté. Le modèle apprend à reconnaître un oiseau au soleil comme à l'ombre.

Pourquoi c'est utile : le modèle ne voit jamais deux fois exactement la même image. Il ne peut pas apprendre les photos par cœur, il est obligé d'apprendre ce qui fait vraiment l'espèce. On ne modifie **que** le train : la validation et le test gardent leurs images d'origine, sinon la mesure ne serait plus fiable.

On ne change **pas** les couleurs de façon forte : la couleur des plumes est souvent ce qui distingue deux espèces. La transformer reviendrait à mentir au modèle.

### Le label smoothing : ne pas être sûr à 100 %

Normalement, on dit au modèle : « cette image, c'est 100 % un pygargue ». Avec le **label smoothing** à 0,1, on lui dit plutôt : « c'est 90 % un pygargue, et les 10 % restants sont répartis sur les autres espèces ».

Ça l'empêche de devenir arrogant. Un modèle qui veut être sûr à 100 % pousse ses boutons à l'extrême pour y arriver, et il apprend par cœur. Avec le label smoothing, il garde une petite place pour le doute, il généralise mieux, et ses pourcentages de confiance sont plus honnêtes.

### Le drop path : s'entraîner avec des morceaux en moins

Un ViT est une pile de blocs. Avec le **drop path** (ou *stochastic depth*), à chaque batch, on saute quelques blocs au hasard (par exemple 10 à 20 % de chances par bloc). Le modèle ne peut pas compter sur un seul bloc pour tout faire : chaque bloc doit être utile tout seul.

C'est comme une équipe de foot qui s'entraîne en retirant un joueur différent à chaque match : tout le monde apprend à jouer à tous les postes. Le jour du vrai match (la prédiction), toute l'équipe joue, et elle est plus solide.

### L'EMA : la moyenne des versions récentes

Pendant l'entraînement, les boutons bougent à chaque batch et « tremblent » un peu autour de la bonne valeur. L'**EMA** (*Exponential Moving Average*) garde une deuxième copie du modèle, qui suit lentement la première : à chaque pas, elle se rapproche juste un tout petit peu des nouveaux boutons.

C'est comme la moyenne de tes notes de l'année, plutôt que ta note du dernier contrôle : elle lisse les bons et les mauvais jours. À la fin, la copie EMA est souvent un peu meilleure et plus stable. On compare les deux sur la validation et on garde la meilleure.

### La résolution : voir plus de détails, en option

On entraîne en 224 × 224, la taille d'origine des images. En option, on pourrait finir par quelques epochs en 336 ou 448 pixels : en agrandissant l'image, le ViT a plus de patchs, donc plus de détails sur les plumes. Mais ça coûte 2 à 4 fois plus cher, et nos images ne font que 224 pixels à l'origine : agrandir n'ajoute pas de vrais détails. Je ne le ferai que si la validation montre un gain.

### Récapitulatif des réglages de départ

| Réglage | Valeur de départ | Pourquoi, en une ligne |
| --- | --- | --- |
| Optimiseur | AdamW, weight decay 0,05 | Le standard pour les ViT, avec un frein contre l'overfitting |
| Learning rate du backbone | environ 1e-5 | Retoucher sans casser |
| Learning rate de la tête | environ 1e-3 | Apprendre vite, elle part de zéro |
| Layer-wise decay | 0,75 | Les premières couches bougent moins |
| Warmup | 1 epoch | Éviter les secousses du début |
| Schedule | Cosinus | Grands pas puis petits pas |
| Epochs | 10 à 15 | Assez pour converger, sans trop coûter |
| Batch | 64 | Ce qui tient bien dans 48 Go de VRAM |
| Précision | bf16 | Deux fois plus rapide |
| Label smoothing | 0,1 | Garder une part de doute |
| Drop path | 0,1 à 0,2 | Rendre chaque bloc utile |
| EMA | Oui | Lisser les tremblements |

Ce sont des valeurs de départ classiques. On surveillera les courbes de loss de train et de validation à chaque epoch, et on ajustera si besoin.

## Mesurer : savoir si le modèle est bon

On juge le modèle avec plusieurs mesures complémentaires, parce qu'un seul chiffre peut cacher des espèces mal reconnues.

### L'accuracy : le pourcentage de bonnes réponses

L'**accuracy**, c'est le nombre de bonnes réponses divisé par le nombre d'images. 99 % sur 2 625 images, c'est 2 599 bonnes réponses et 26 erreurs.

On regardera aussi le **top-5** : le pourcentage d'images pour lesquelles la bonne espèce est parmi les 5 propositions les plus probables. C'est utile pour une application : si le modèle hésite entre deux moineaux, montrer les deux à l'utilisateur est déjà très utile.

### La matrice de confusion et le recall par espèce

La **matrice de confusion** est un grand tableau : en ligne la vraie espèce, en colonne la réponse du modèle (slide 25). La diagonale, ce sont les bonnes réponses. Tout ce qui est en dehors, ce sont les confusions. Avec 525 espèces, le tableau aurait 275 625 cases : illisible. On en tirera donc plutôt **la liste des paires d'espèces les plus souvent confondues**, avec des photos.

Le **recall** d'une espèce, c'est : parmi les vraies photos de cette espèce, combien le modèle en a reconnu. Il permet de trouver les espèces que le modèle reconnaît mal, même si l'accuracy globale est excellente (slide 27 : « un score global peut masquer une classe mal reconnue »). Attention : avec 5 images par espèce, un recall de 80 %, c'est juste une erreur. Il faut rester prudent.

### Repérer l'overfitting avec les courbes

À chaque epoch, on note la loss et l'accuracy sur le train et sur la validation, et on trace les courbes, comme au TP5 et au TP7 :

- Les deux courbes s'améliorent ensemble : tout va bien, le modèle apprend.
- La courbe du train continue de s'améliorer mais celle de la validation stagne ou se dégrade : c'est l'overfitting. Le modèle commence à apprendre par cœur. On garde alors le checkpoint de la meilleure epoch de validation.

### Pourquoi le test ne se regarde qu'une fois

Toutes les décisions (quel backbone, quels réglages, quelle epoch) sont prises en regardant **uniquement la validation**. Une fois tout figé, on calcule le score du test, une seule fois, et on le garde même s'il déçoit (slide 85).

Si on regardait le test pour faire des choix, on choisirait sans le vouloir ce qui marche sur ces 2 625 images-là, et pas forcément sur de nouvelles photos. Le score serait trop optimiste : un peu comme un élève qui aurait pu jeter un œil au sujet.

### La limite de précision de nos mesures

Avec 2 625 images de test, une seule erreur représente 0,04 point. Entre deux modèles à 99,5 % et 99,6 %, il y a 3 erreurs d'écart : c'est trop peu pour dire lequel est vraiment meilleur, le hasard suffit à l'expliquer. Au-delà d'environ 99,5 %, gratter des dixièmes ne veut presque plus rien dire. C'est pour ça que l'analyse des erreurs compte plus que le dernier dixième.

## Comprendre les erreurs

C'est la partie que le cours valorise le plus : « la qualité de la démarche compte davantage qu'un classement entre scores » (slide 86). On fera quatre analyses.

### 1. Les erreurs où le modèle était le plus sûr de lui

Comme au TP7, on affiche les erreurs classées par confiance. Une erreur à 40 % de confiance, c'est un modèle qui hésite : normal. Une erreur à 98 %, c'est un modèle convaincu d'autre chose : c'est là qu'on apprend le plus. Souvent, on y trouve :

- deux espèces presque identiques ;
- une photo ambiguë (oiseau de dos, coupé, dans l'ombre) ;
- parfois une **erreur de label** dans le dataset lui-même : le modèle a raison et l'étiquette a tort.

Pour chacune, on écrit une explication, comme dans le TP7.

### 2. Chercher des doublons entre le train et le test

Ce dataset Kaggle est connu pour contenir des photos très semblables (voire identiques) dans le train et dans le test. Si c'est le cas, le modèle peut « reconnaître » une photo de test simplement parce qu'il l'a déjà vue, et le score est gonflé.

Pour le vérifier, on utilise les embeddings (slide 46) : pour chaque image du test, on cherche l'image du train dont l'embedding est le plus proche. Si la **similarité cosinus** est très haute (par exemple plus de 0,97), c'est sans doute la même photo, ou une photo de la même série. On affiche les paires pour vérifier à l'œil, puis on recalcule l'accuracy sans ces images. Le cours insiste sur l'indépendance des tas (slide 79) : montrer cette vérification est un vrai point fort.

### 3. Les biais du dataset

Un **biais**, c'est quand les données ne représentent pas bien la réalité. Dans ce dataset, la grande majorité des photos montrent des mâles. Chez beaucoup d'espèces, le mâle est coloré et la femelle beaucoup plus terne. Le modèle risque donc de mal reconnaître les femelles, et le test ne le montrera pas, puisqu'il contient surtout des mâles lui aussi.

Autres biais à signaler : des photos très propres (oiseau centré et net), et seulement 525 espèces. Si on montre au modèle une espèce qu'il ne connaît pas, il répondra quand même l'une des 525 : il ne sait pas dire « je ne connais pas ». On regardera si sa confiance baisse dans ce cas.

### 4. Le test sur tes propres photos

C'est le vrai test du monde réel. Tu prends des photos d'oiseaux toi-même (au jardin, au parc, ou sur des sites dont tu as le droit d'utiliser les images), d'espèces qui font partie des 525, par exemple un moineau domestique (`HOUSE SPARROW`). On regarde ce que le modèle répond.

On s'attend à un score nettement plus bas que 99 %, car ces photos ne ressemblent pas à celles du dataset : oiseau plus petit, décor différent, autre appareil. C'est normal, et c'est exactement ce que le cours demande d'expliquer dans les limites (slide 87). Ça répond aussi en partie à la consigne de la slide 76, qui demande de constituer ses propres images.

## Les livrables

À la fin, tu auras quatre choses, celles que demande la slide 86.

| Livrable | Ce qu'il contient | Pourquoi |
| --- | --- | --- |
| **Notebook exécutable** | Toutes les étapes, de l'exploration au test final, avec les courbes, les tableaux et un journal des essais | Montrer la démarche, et que n'importe qui puisse la refaire |
| **Modèle rechargeable** | Les poids, le nom du backbone, le preprocessing et la liste des 525 espèces dans le bon ordre | Sans la liste des espèces, le modèle dit « 53 » au lieu de « BALD EAGLE » ; sans le bon preprocessing, il se trompe |
| **Fonction `predict(image)`** | Elle prend une photo et renvoie les 5 espèces les plus probables avec leur pourcentage | Pouvoir utiliser le modèle en démo, sur n'importe quelle photo |
| **README et dataset documenté** | Les sources, les effectifs, les tas, les choix faits et leurs raisons, les limites | À la slide 86, le cours demande un dataset documenté |

Le modèle final tournera aussi sur ton ordinateur, sans GPU. Un ViT-Large y prendra environ une seconde par photo : c'est lent pour des milliers d'images, mais largement suffisant pour une démo.

## Ce qu'on ne fera pas, et pourquoi

Certaines techniques existent mais ne valent pas leur coût ici ; les citer dans le rapport comme pistes montre qu'on les connaît.

- **Un ensemble de plusieurs modèles.** On entraînerait deux modèles différents et on ferait la moyenne de leurs réponses. Comme ils ne se trompent pas sur les mêmes images, ils se corrigent parfois. Mais le gain serait de quelques erreurs sur 2 625, invisible face au hasard, pour un coût double. C'est une bonne « piste d'amélioration » à mentionner.
- **Entraîner un gros modèle à partir de zéro.** Sans préentraînement, un ViT-Large aurait besoin de millions d'images et de jours de calcul pour faire moins bien. Le transfer learning existe justement pour éviter ça.
- **Une très grosse carte (H100, B200).** Elle irait plus vite, mais pour un coût total plus élevé et le même modèle à la fin.
- **Des augmentations de couleur fortes.** Elles détruiraient l'information qui distingue les espèces.
- **Regarder le test pour choisir.** Même une seule fois, ça rendrait le score final moins crédible.

Un dernier point à vérifier avec ton professeur : le cours propose de constituer son propre dataset, en commençant par 2 ou 3 classes (slides 73 et 76). Utiliser un dataset public de 525 espèces s'en écarte. Le test sur tes propres photos compense en partie, mais demande-lui si c'est accepté.

## Le petit dictionnaire

Tous les mots techniques du document, chacun en une ligne, par ordre alphabétique.

| Mot | Ce que ça veut dire |
| --- | --- |
| Accuracy | Le pourcentage de bonnes réponses |
| Adam / AdamW | L'optimiseur qui tourne les boutons ; AdamW ajoute un léger frein (weight decay) |
| Attention | Le mécanisme qui permet à chaque morceau d'image de regarder tous les autres |
| Backbone | La grosse partie d'un modèle qui transforme l'image en embedding |
| Baseline | La méthode la plus simple, qui sert de point de comparaison |
| Batch | Un paquet d'images traitées ensemble, par exemple 64 |
| bf16 | Des nombres plus courts, pour calculer deux fois plus vite |
| Biais | Quand les données ne représentent pas bien la réalité |
| Canal | Une des 3 couleurs d'une image : rouge, vert ou bleu |
| Checkpoint | Une sauvegarde du modèle pendant l'entraînement |
| CNN | Un réseau qui regarde l'image avec des loupes (convolutions) |
| Convolution | Faire glisser un petit filtre sur toute l'image pour y chercher un motif |
| CPU | Le processeur : peu de calculs à la fois, mais très polyvalent |
| CrossEntropyLoss | La loss utilisée pour classer, qui punit fort les erreurs sûres d'elles |
| Data augmentation | Modifier légèrement les images du train pour en créer des variantes |
| Dataset | L'ensemble des images et de leurs labels |
| Descente de gradient | Tourner chaque bouton un peu dans le sens qui fait baisser la loss |
| Drop path | Sauter des blocs au hasard pendant l'entraînement, pour rendre chacun utile |
| EMA | Une copie du modèle qui suit la moyenne de ses versions récentes |
| Embedding | La liste de nombres qui résume une image |
| Ensemble | Faire la moyenne des réponses de plusieurs modèles |
| Epoch | Un passage complet sur toutes les images du train |
| Feature extraction | Garder le backbone gelé et n'entraîner que la tête |
| Filtre (kernel) | La petite loupe qui cherche un motif |
| Fine-tuning | Réentraîner doucement tout le modèle préentraîné |
| GPU | La carte graphique : des milliers de petits calculs en même temps |
| Gradient | La direction dans laquelle tourner chaque bouton |
| Label | La bonne réponse associée à une image (son espèce) |
| Label smoothing | Dire au modèle « 90 % sûr » plutôt que « 100 % sûr » |
| Layer-wise lr decay | Donner un learning rate plus petit aux premières couches |
| Learning rate | La taille des pas quand on tourne les boutons |
| Logits | Les 525 scores bruts en sortie du modèle |
| Loss | Le chiffre qui mesure à quel point le modèle s'est trompé |
| Matrice de confusion | Le tableau vraie espèce × réponse du modèle |
| Modèle | La machine à calculer qui transforme une image en réponse |
| Optimiseur | Le programme qui décide de combien tourner chaque bouton |
| Overfitting | Apprendre les images du train par cœur au lieu de comprendre |
| Patch | Un petit carré d'image, la pièce de puzzle d'un ViT |
| Pixel | Le plus petit point d'une image |
| Pod | L'ordinateur loué à distance sur RunPod |
| Poids (weights) | Les boutons du modèle, réglés pendant l'entraînement |
| Pooling | Rétrécir l'image en gardant le plus important de chaque zone |
| Préentraîné | Déjà entraîné par quelqu'un d'autre sur beaucoup d'images |
| Preprocessing | La préparation des images attendue par un modèle |
| Recall | Pour une espèce, la part de ses photos que le modèle a reconnues |
| Similarité cosinus | Une mesure de ressemblance entre deux embeddings (1 = identiques) |
| Softmax | La fonction qui transforme les logits en pourcentages |
| Tête (head) | La petite couche finale qui donne l'espèce à partir de l'embedding |
| Test | Le tas d'images pour le score final, utilisé une seule fois |
| Top-5 | La bonne espèce est-elle dans les 5 premières propositions ? |
| Train | Le tas d'images sur lequel le modèle apprend |
| Transfer learning | Réutiliser un modèle préentraîné pour une nouvelle tâche |
| Validation | Le tas d'images pour comparer et choisir |
| ViT | Vision Transformer : un modèle qui découpe l'image en patchs |
| VRAM | La mémoire de la carte graphique |
| Warmup | Faire monter doucement le learning rate au début |
| Weight decay | Un léger rappel des boutons vers zéro, contre l'overfitting |
