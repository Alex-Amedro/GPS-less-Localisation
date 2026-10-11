PHILOSOPHIE (prioritaire sur le reste)
Je ne fais pas de recherche, je construis un système réaliste, comme en industrie.
- Je prends des briques éprouvées (modèle pré-entraîné, filtre de particules
  classique, méthodes standard d'incertitude). Je ne réinvente rien.
- Je ne cherche pas l'état de l'art et je ne revendique aucune nouveauté.
  Le README dit simplement : "voici un système, voici ce qu'il fait, voici
  où il échoue".
- D'abord une version 0 qui marche de bout en bout, même médiocre :
  image -> position estimée -> affichée sur la carte.
- Ensuite on ajoute UNE brique à la fois (entraînement, filtre, "je ne sais pas").
  Une brique n'est gardée que si elle améliore mesurablement la version
  précédente. Sinon on la retire et on le note dans experiments.md.
- Ce qui compte : fiabilité, vitesse par image, cas d'échec connus, démo claire.
  Pas la sophistication.

CONTEXTE
Projet : localisation visuelle sans GPS. Un drone à voilure fixe vole à ~300 m
d'altitude, caméra stabilisée vers le sol, altitude et cap connus. Le GPS est perdu
en plein vol. À bord : une carte satellite préchargée (descripteurs compacts).
Le système doit :
  1) retrouver où il est à partir de l'image,
  2) s'aider du contexte (dernière position connue, plusieurs images successives),
  3) dire "je ne sais pas" quand il n'est pas sûr.
Présentation : "images aériennes de n'importe où". Le drone n'est que la démo
d'application. Projet portfolio (ML / défense / Forward Deployed Engineer), qui doit
être HONNÊTE et DÉFENDABLE. Un projet 2 (déploiement embarqué) viendra plus tard :
ne pas s'en occuper ici.

MACHINE
Windows, PowerShell, MSI Crosshair 16, RTX 5070 8 Go, 16 Go RAM.
PyTorch avec CUDA 12.8+ (GPU Blackwell). Créer un dossier de projet séparé du
projet Scanner (aucune réutilisation de son code).

RÈGLES
- N'invente AUCUN chiffre, AUCUNE source, AUCUN résultat. Si tu ne sais pas :
  mesure, ou écris "non vérifié".
- Chaque étape finit par un petit rapport (données d'abord, interprétation ensuite).
- À chaque porte (GATE) tu t'arrêtes et tu me montres les résultats avant de continuer.
- Un script = une tâche, option --help, graines fixes, résultats sauvegardés.
- Tiens un journal experiments.md : ce qui a été tenté, ce qui a marché ou non.
- Ne pas fine-tuner tout un gros modèle : backbone pré-entraîné avec la plupart des
  blocs gelés, seule la tête (et éventuellement les derniers blocs) s'entraîne.
- Si quelque chose bloque vraiment : pose-moi 1 ou 2 questions courtes à choix.
  Sinon prends la décision la plus raisonnable et dis laquelle.

ÉTAPE 0 — Vérification des données (jour 1). GATE.
a) Extraire 50 tuiles de 512x512 avec coordonnées (Lambert 93 ou lat/lon) d'une dalle
   IGN BD ORTHO (.jp2), avec lecture par fenêtre (rasterio), sans charger toute la
   dalle. Afficher 6 tuiles avec leurs coordonnées. Si tu ne peux pas télécharger la
   dalle toi-même, dis-moi exactement où et quoi télécharger.
b) Vérifier pour CHAQUE dataset réel : existence, lien de téléchargement, taille,
   licence, contenu réel (altitude, pays, métadonnées). Écris "non vérifié" si tu ne
   peux pas le confirmer. Candidats :
   - OrthoLoC (deepscenario.github.io/OrthoLoC) : 16 425 images, Allemagne et USA
   - AnyVisLoc (github.com/UAV-AVL/Benchmark) : 18 000 images, Chine, 30-300 m
     (attention : plutôt basse altitude, vues obliques)
   - UAV-VisLoc (GitHub IntelliSensing) : cartes issues de Google Earth, licence floue
   - VPAIR (github.com/AerVisLoc/vpair) : disponibilité NON confirmée
   - ALTO : portée publique limitée
   Vérifier aussi si les images historiques IGN "Remonter le temps" sont
   téléchargeables et à quelles conditions.
c) Proposer UN seul jeu de test réel (le plus simple à obtenir et le plus proche du
   scénario), expliquer pourquoi, et comment l'utiliser sans fuite d'information
   (aucune image de test vue à l'entraînement).
Note : si le jeu de test est hors de France, c'est un test de généralisation à un
autre pays. À écrire clairement dans les limites.

ÉTAPE 1 — Baseline classique, sans deep learning. GATE.
SIFT/ORB + mise en correspondance sur un petit échantillon du jeu réel retenu.
Définir la métrique : une position est "correcte" si l'erreur est < X m (rapporter
plusieurs seuils, par ex. 50 / 100 / 500 m), plus l'erreur médiane et le temps par
image. Ce résultat sert de plancher de comparaison.

ÉTAPE 2 — Version 0 de bout en bout.
Modèle pré-entraîné (DINOv2 petit ou ConvNeXt), SANS aucun entraînement :
image -> descripteur -> recherche dans la carte IGN découpée en tuiles -> position
estimée. Mesurer sur le jeu réel avec la même métrique que la baseline.
Faire un petit affichage : image du drone, tuile retrouvée, position sur la carte.
GATE : si le résultat est proche du hasard, on s'arrête et on analyse pourquoi
(écart de domaine, saisons, relief, résolution) avant de continuer.

ÉTAPE 3 — Entraînement (seulement si l'étape 2 le justifie).
Fabriquer des paires (vue "drone", tuile de la carte) à partir de l'IGN, avec rotation,
échelle, flou, changements de luminosité, recadrage décalé, quelques occultations.
Séparer train/val/test PAR ZONE géographique (jamais deux tuiles voisines dans deux
ensembles). Entraînement contrastif (InfoNCE) avec négatifs difficiles (tuiles
voisines), tête de descripteur compacte, précision mixte.
Mesurer ce que l'entraînement apporte réellement par rapport à l'étape 2, sur le
jeu réel. Rapporter honnêtement l'écart "synthétique -> réel". Si l'apport est
faible ou nul, le dire et garder la version 0.

ÉTAPE 4 — Contexte et filtre de particules.
D'abord un petit script NumPy jouet (positions simulées) pour valider la logique :
modèle de déplacement à vitesse constante, poids = similarité cosinus,
ré-échantillonnage systématique. Puis le brancher sur de vraies séquences d'images.
Aucun réentraînement. Mesurer le gain par rapport à "une image à la fois".
Le rayon de recherche grandit avec le temps depuis la dernière position connue.

ÉTAPE 5 — "Je ne sais pas".
Mesurer la confiance (ratio entre la meilleure et la deuxième meilleure candidate,
et/ou prédiction conforme). Produire une courbe couverture / erreur : si le système
accepte de répondre moins souvent, de combien l'erreur baisse ? Vérifier que la
calibration tient sur des zones jamais vues.

ÉTAPE 6 — Ablation et rendu.
Tableau : baseline classique / pré-entraîné seul / + entraînement / + filtre /
+ abstention. Cas d'échec montrés (forêt, eau, zones sans texture, saisons
différentes). Vitesse par image sur ma machine. Démo courte (vidéo ou gif).
README : problème, scénario, méthode, résultats, LIMITES CLAIRES (domaine synthétique,
pays différent, pas de test sur matériel embarqué). Aucune revendication de nouveauté.