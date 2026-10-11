# experiments.md — journal du projet

Règle : tout ce qui est tenté est noté ici, que ça ait marché ou non. Chiffres = mesurés (script ou fichier cité), sinon « non vérifié ».
Une brique n'est gardée que si elle améliore mesurablement la version précédente.

## Remarques de l'utilisateur (à tenir à jour)
- « fait le venv aussi et fait un exeriement.md qui permet de noté tout ce qui est fait »
- **Mode de travail** : étudiant, veut comprendre. Une étape à la fois, attendre son accord ; français simple, termes définis ; aucun chiffre ni source inventés ;
  aucun téléchargement/entraînement/étape suivante sans accord ; pas de limite de taille de téléchargement mais prévenir et vérifier le disque.
- **Règles de fond** : projet réaliste, briques éprouvées, pas de nouveauté revendiquée ; le système connaît cap et altitude approximative, jamais la position ;
  la vérité terrain ne sert qu'à mesurer l'erreur ; le test réel ne sert jamais à entraîner ni à régler.
- **Tests** : un modèle qui reconnaît des lieux à des dates différentes est déjà un bon résultat. Ajouter des tests en France : même zone à une autre date,
  et un paysage différent « sur le papier » (Lille/Marseille = exemples, pas des choix), **de taille similaire à Belfort** (Marseille, 25 Go, refusée). L'utilisateur vérifiera visuellement.
- **Éditions IGN** : il y a un écart de qualité entre les éditions (2013, 2017 vs 2020) : à garder en tête.
- **Entraînement** : d'abord notre idée (IGN/France) puis test sur les vols UAV-VisLoc ; si ça échoue, entraîner sur des vols UAV-VisLoc (jamais ceux du test).
- **SIFT (étape 1)** : pas d'entraînement, donc évaluable partout. Taille au sol d'un pixel : seulement en cas de soupçon, l'utilisateur ne sait pas la mesurer (je le fais).
- Droits des cartes Google : secondaire pour lui ; on ne redistribue pas les images, on met seulement des liens de téléchargement (dans le README).
- Jeu complet UAV-VisLoc (11 vols) : **l'utilisateur le télécharge lui-même** avec son explorateur (le téléchargement automatique était bloqué) et donnera le chemin quand c'est fini.
- **Documents** : ne garder que `Project.md`, `experiments.md` et `README.md`. Pas de fichiers .md en plus : écrire dans le chat.
- **Dépôt propre** : supprimer les scripts à usage unique (téléchargement, journaux), les images et extraits intermédiaires dès qu'ils ne servent plus. Les résultats utiles sont écrits ici.

## Étape 0 — données (2026-10-09)

**Environnement** : RTX 5070 Laptop 8 Go, Python 3.12.10, `.venv` (torch 2.11.0+cu128, GPU détecté), deps dans `requirements.txt`.

**IGN BD ORTHO (carte)** — catalogue : https://data.geopf.fr/telechargement/resource/BDORTHO ; une archive `.7z` par département et par édition.
- Belfort (D090) RVB 20 cm Lambert-93 : éditions 2013, 2017, 2020. **2020 et 2017 téléchargées** (3 882 Mo et 4 075 532 894 octets, tailles exactes), 43 dalles chacune, même grille de noms
  (`90-2020-0995-6715-…` ↔ `90-2017-0995-6715-…`) → comparaison dalle par dalle possible. 2013 non téléchargée.
- Une dalle = 25 000 × 25 000 px = 5 km × 5 km à 0,20 m/px. Prise de vue de l'édition 2020 : 2020-05-07 (métadonnées XML, une seule date pour tout le département, indicative) ; 2013 et 2017 non lues.
- Test de lecture par fenêtre (rasterio, sans charger la dalle) : 50 tuiles 512×512 (≈ 102 m) de la dalle du milieu (`0995-6715`) ; coordonnées cohérentes (47,43° N, 6,95° E).
  Script et images supprimés après usage. **Constat** : 6/6 tuiles affichées = forêt/prairie → cette dalle n'est pas représentative ; il faudra plusieurs dalles, dont urbaines.
- Licence : Licence Ouverte Etalab (source secondaire, à reconfirmer) ; mention « BD ORTHO®, IGN ».
- La plupart des départements pèsent 25–40 Go (mesuré, voir « Nouvelle ville ») ; Belfort (3,9 Go) est une exception.

**UAV-VisLoc** (test réel principal) — https://github.com/IntelliSensing/UAV-VisLoc · https://arxiv.org/abs/2405.11936
- Échantillon téléchargé (2,09 Go) = **vol 03 seulement** : 768 images 3976×2652, 2018-10-23 08:32→09:49, 73,9 km, pas médian 7 s / 95 m, 7 trous > 20 s (images retirées ? non vérifié),
  hauteur 465–468 m, tangage/roulis ≤ 5° (vue quasi verticale). Carte `satellite03.tif` : 35 092 × 24 308 px, EPSG:4326, ≈ 0,25–0,30 m/px, ≈ 8,9 km de large, pas de date dans le fichier
  (l'article donne 2021-04, à recouper). Écart de saison visible (riz jaune sur la photo, champs sombres sur la carte). CSV : lat, lon, hauteur, angles, cap (`Phi1` plus fiable que `Phi2`).
- **Jeu complet téléchargé par l'utilisateur** (le téléchargement automatique était bloqué : « Quota exceeded »), rangé dans `data/raw/uavvisloc/` (≈ 22 Go dézippé). Échantillon supprimé (doublon ; vol 03 vérifié complet : 768 images, csv, carte 2 578 Mo).
  Mesuré : 11 vols, **6 774 images** (l'article annonce 6 742 ; écart de 32 non expliqué). Images par vol : 01 817 · 02 1 071 · 03 768 · 04 738 · 05 473 · 06 344 · 07 30 · 08 1 033 · 09 766 · 10 144 · 11 590.
  Vol 09 : carte découpée en 4 fichiers `satellite09_*.tif`. Vol 07 : seulement 30 images et carte de 2,3 Mo dont les coordonnées couvrent ≈ 50 m de haut → à vérifier avant tout usage. `satellite_ coordinates_range.csv` donne « Huzhou-3 » pour les vols 08 et 09 (l'article distingue Huzhou-3 et Huzhou-6) : étiquette à vérifier.
- Taille au sol d'un pixel : le README dit ~0,1–0,2 m/px (non mesuré par nous).

**VPAIR** (test secondaire) — https://zenodo.org/records/6473989 (CC BY 4.0 d'après Zenodo ; arXiv affiche CC BY-NC-ND : incohérence). Échantillon de 861 Mo, MD5 conforme : 200 requêtes, 200 références, 200 distracteurs (800×600).
Références = **rendus** à la pose de chaque requête (nord en haut) ; distracteurs à **54–78 km** des requêtes (négatifs triviaux) ; ≈ 9 km, hauteur 360–403 m (ECEF → lat/lon) ; aucune date dans les fichiers
(l'article : requêtes le 2020-10-13, orthophotos 2019–2021) ; 199 poses pour 200 images.

**Écartés** : OrthoLoC (23–154 m, obliquité moyenne 14,6°, poses décalées au hasard donc pas de géoréférence absolue, ≈ 256–287 Go) ; AnyVisLoc (20 077 images, Chine, 6–500 m, obliques) ; ALTO et « Remonter le temps » : non vérifiés.

**Décisions** : test réel principal = UAV-VisLoc ; VPAIR secondaire ; carte d'entraînement = IGN ; version 0 d'abord, une brique à la fois.

**Modèle (étape 2, à choisir)** — sources : dépôt DINOv2, arXiv 2308.00688 (AnyLoc), arXiv 2311.15937 (SALAD), doc torchvision. A DINOv2 ViT-S/14 (21 M paramètres, Apache 2.0) ;
B ConvNeXt-Tiny (28,6 M) ; C SALAD (DINOv2-B 86 M, entraîné sur vues de rue, pas d'évaluation aérienne). AnyLoc (DINOv2 géant + VLAD, sans entraînement) : Recall@1 66,7 % sur VP-Air contre 10,3 % pour MixVPR.
Mémoire GPU et vitesse : non mesurées. **Recommandation : A.** Choix de l'utilisateur : en attente.

**Incidents / leçons** : 429 si on interroge le catalogue trop vite (espacer) ; le serveur IGN répond 403 au User-Agent par défaut de Python (en mettre un) ; `python -I` ignore `PYTHONIOENCODING` (utiliser `-X utf8 -I`) ;
chemins d'extraction .7z trop longs sous Windows (extraire vers un chemin court) ; GDAL `/vsizip/` refuse l'extension `.bin`. Marseille (D013) : téléchargement arrêté à 4 morceaux sur 7 (14,1 Go) et supprimé.

## Étape 1 — SIFT/ORB
Protocole validé par l'utilisateur (« vas-y »). Tests : A UAV-VisLoc vol 03 puis autres vols · B Belfort 2017 (fenêtres) contre 2020 (carte) · C Val-d'Oise 2018 contre 2021 · E VPAIR (secondaire).
Mesures : Recall@1/@5 (tuile à moins de 50/100/500 m), erreur médiane (m), temps par image, niveau du hasard. Réglages SIFT fixés avant de voir les résultats ; zones de test France réservées pour toujours.

**Test rapide (diagnostic, pas une mesure)** — `scripts/step1_quicktest_sift.py`, vol 03, 5 photos régulièrement espacées (indices 0, 192, 384, 575, 767), fenêtre de carte de 2 km centrée sur la vraie position,
carte rééchantillonnée à 0,8 m/px (**la carte est en degrés : pixels de 0,252 m × 0,299 m, il faut les rendre carrés**), ratio de Lowe 0,75, RANSAC similitude 5 px, SIFT OpenCV par défaut.
- 3 photos sur 5 placées au bon endroit : décalage 10, 7 et 21 m par rapport à la vérité, mais **seulement 8, 8 et 4 correspondances cohérentes**.
- 2 photos échouées (03_0001, 03_0768) : transformation dégénérée (échelle ≈ 0), décalage 500 et 734 m, pourtant 6 et 19 « correspondances cohérentes » → **le nombre de correspondances seul ne distingue pas une bonne d'une mauvaise réponse**.
- **Taille au sol d'un pixel des photos du vol 03 : ≈ 0,113–0,116 m/px** (3 photos concordantes ; le README annonçait 0,1–0,2). Valable pour ce vol (hauteur 466 m) ; les autres vols ont d'autres hauteurs.
- **Cap** : la rotation trouvée par SIFT ≈ `Phi1` (écarts de 2 à 8°, même signe) → `Phi1` est fiable et peut servir à vérifier une réponse. Les deux échecs ont une rotation incompatible avec le cap (133° contre −39° ; 0° contre 123°).
- Temps : ≈ 2,2–2,5 s par photo (3 tailles de photo essayées, fenêtre de 2 km seulement) ; une recherche sur toute la carte sera bien plus lente : non mesuré.
- Attention : ce test est local (on connaît déjà l'endroit à 1 km près) ; une recherche sur toute la carte sera plus dure.
- Images : `data/vis/quicktest_vol03.png` (photo, fenêtre de carte, cadre vert = placement SIFT, croix rouge = vérité) et `data/vis/carte_vol01.png` (carte entière du vol 01 avec le trajet).

**Val-d'Oise 2018** (6,97 Go) : téléchargé, vérifié (`data/raw/ign_d095_2018/`).

**Protocole final validé (2026-10-09)** : ordre B (contrôle Belfort) → A (vol 03) → C (Val-d'Oise). Principe fixé par l'utilisateur : **photo récente contre vieille carte** →
Belfort : carte **2017**, photos = fenêtres de l'édition **2020** ; Val-d'Oise : carte 2018, photos 2021. (UAV-VisLoc vol 03 est dans l'autre sens d'après l'article : carte 2021, photos 2018.)
Méthode : échelle commune 0,8 m/px ; SIFT de la carte calculé une fois (blocs sans doublons) ; points de la photo appariés à toute la carte (FLANN, ratio 0,75) ; votes par tuiles de 512 m (pas 128 m, 20 candidates au plus) ;
RANSAC similitude 5 px par candidate ; réponse valide seulement si échelle dans un facteur 1,5 de l'attendue et rotation à ±20° du cap ; classement par correspondances cohérentes, sans seuil minimum. Tolérances fixées avant de voir les résultats.
Requêtes IGN : 200 fenêtres de 460 × 305 m, position et rotation tirées par graine fixe dans un bloc de 2×2 dalles (10 × 10 km) tiré par graine fixe, réservé pour toujours au test.

**Test B — Belfort, carte 2017, photos = fenêtres de 2020 (200 photos, 460 × 305 m, rotation aléatoire donnée comme cap) — `scripts/prep_ign_test.py` puis `scripts/sift_eval.py`**
- Préparation (mesuré) : bloc 2×2 dalles tiré par graine 0 = Lambert-93 x 1005–1015 km, y 6720–6730 km (bord est du territoire ; paysage rural : forêts, champs, petits villages ; 24 % de la carte vide hors territoire, requêtes tirées seulement en zone valide).
  Grilles 2017 et 2020 identiques (bornes, taille, CRS, résolution) ; décalage entre éditions par corrélation de phase : médian ≈ 0 m, maximum 8,2 m sur 6 fenêtres. Éditions très différentes visuellement (saison, longueur des ombres, champ fauché ou non, bandeau de couleur entre deux dalles de 2017).
- Calcul (mesuré) : carte 12 500 × 12 500 px à 0,8 m/px → 1 274 161 points SIFT en 12 s, 0,65 Go ; index FLANN 6 s ; **0,12 s par photo** (médiane 0,11 s) → toutes les photos sont évaluables.
- **Résultat officiel** : taux de réponse **1,5 %** (3 photos sur 200) ; Recall@1 = Recall@5 = **1,5 %** à 50, 100 et 500 m ; les 3 réponses sont exactes (erreur médiane des photos avec réponse : 5 m) ; niveau du hasard 7,9·10⁻⁵ (50 m) à 7,9·10⁻³ (500 m).
  Aucune fausse réponse : le filtre cap + échelle n'a laissé passer que des réponses justes (3 sur 3, échantillon minuscule).
- **Contrôles (mesurés)** : (1) photos tirées de la *même* carte 2017 : 8/8 réussies, erreur 0,3–0,6 m, 674–2 440 correspondances cohérentes → la chaîne (rotation, échelle, votes, validation) est correcte ;
  (2) sur les vraies requêtes, seulement 5 à 18 appariements survivent au test du ratio sur ≈ 700–5 300 points par photo, et presque aucun n'est à moins de 400 m de la vérité ;
  (3) même avec la base restreinte à 1 km autour de la vérité (non réaliste) : 1 réussite sur 20 → l'échec ne vient pas de la taille de la base mais du changement d'édition.
- **Interprétation** : SIFT, réglé comme prévu, ne résiste pas à un changement d'édition IGN de 3 ans (saison, ombres, cultures) même entre deux orthophotos verticales. Plancher très bas.
  **Non testé** : l'échelle de travail 0,8 m/px a pu écraser des détails utiles à SIFT ; aucune variante (échelle plus fine, ratio plus tolérant) n'a été évaluée. Toute variante devra être choisie sur un bloc de validation distinct, pas sur ce bloc de test.
- Images : `data/vis/belfort_B_reussite.png` (erreur 1 m, 5 correspondances), `data/vis/belfort_B_echec.png` (forêt et étang, aucune réponse valide) ; aperçus `belfort_B_apercu_carte.png` et `belfort_B_qualite.png`.

**Correction et suite du test B (2026-10-10)**
- **Biais de sélection corrigé** : mon masque de « zone vide » écartait toute fenêtre contenant un seul pixel blanc ou noir pur (toit saturé, ombre profonde) → ≈ 2 tirages sur 3 rejetés. Désormais seules les grandes zones (> 8 m) blanches/noires comptent comme vides ; tirages de candidats d'avance (graine fixe), indépendants de l'échelle.
  Le résultat « 1,5 % » ci-dessus est la version v0 (biaisée). **B v1 (masque corrigé, mêmes dalles, graine 0, 200 photos)** : taux de réponse 3,0 % ; Recall@1 = Recall@5 = **2,5 %** à 50/100/500 m ; erreur médiane 3 m sur les photos avec réponse ; 0,16 s par photo.
- **Bloc de validation** (dalle `985-6735` de Belfort, graine 1, 100 photos, 25 km², hors du bloc de test) = **la ville de Belfort** (immeubles, rues, voies ferrées, quartiers pavillonnaires, un peu de champs et forêt). Même SIFT, mêmes réglages :
  - 0,8 m/px : taux de réponse 64 %, Recall@1 = Recall@5 = **64 %** à 50/100/500 m, erreur médiane 2 m, 0,15 s par photo ;
  - 0,4 m/px : taux de réponse 66 %, Recall@1 = Recall@5 = **64 %**, erreur médiane 2 m, **0,57 s par photo**, carte 1,4 M de points (15 s).
  → **Réduire moins la carte n'améliore rien** (suggestion de l'utilisateur, testée) et coûte 4× plus de temps.
- **Conclusion** : l'écart 64 % contre 2,5 % vient du **paysage**, pas de l'échelle ni du masque : SIFT s'accroche aux bâtiments et aux routes (stables entre éditions) et échoue en forêt/champs (cultures, ombres, saison). Le bloc de test B (rural, bord du territoire) est donc peu représentatif du département :
  le plancher doit être rapporté **par type de paysage**, sur plusieurs dalles, pas sur un seul bloc tiré au sort.
- Fichiers : `data/work/belfort_B` (carte 447 Mo + 200 photos), `data/work/belfort_val_g08` ; image `data/vis/belfort_val_apercu.png`.
- **Décision de l'utilisateur (2026-10-10) : plancher SIFT figé, on ne teste pas davantage** (pas de campagne sur 43 dalles, pas de variantes). Plancher retenu : **2,5 % en rural (test B), 64 % en ville (validation)**.
  Raisonnement de l'utilisateur : les humains reconnaissent facilement les formes des champs, forêts et routes entre 2017 et 2020 ; un réseau profond devrait pouvoir le faire là où SIFT (points de repère locaux) échoue → hypothèse à tester à l'étape 2.
  SIFT sur le vol 03 (test A) et Val-d'Oise (test C) : non lancé ; sera mesuré avec le même script au moment de comparer le modèle profond (même métrique, coût quasi nul).

## Étape 2 — version 0 : DINOv2 petit (ViT-S/14) sans entraînement (2026-10-11) — `scripts/dino_eval.py`
Modèle : `dinov2_vits14` (torch.hub, poids 84 Mo, 22,1 M paramètres comptés, Apache 2.0). **Aucun entraînement.** GPU : 146 Mo pour une image, **498 Mo max** pendant la carte (RTX 5070, fp16).
Méthode : carte découpée en **disques de 305 m** (pas 128 m, nord en haut), disque central de 305 m de la photo remise nord en haut avec le cap (partie toujours visible quelle que soit la rotation), même masque / taille (224 px) / normalisation des deux côtés,
similarité cosinus ; réponses = meilleurs disques non chevauchants (> 305 m entre deux). Variantes d'empreinte évaluées : `cls` (jeton de classe), `mean` (moyenne des patchs), `concat`. Les trois sont indiscernables sur la validation (≤ 1 point d'écart sur 100 photos) → **`cls` fixé avant de regarder le test B** (choix a priori, le plus standard).
Précision limitée par la grille : le centre d'un disque est à ≤ ≈ 90 m de la vérité ; le seuil 50 m n'est donc pas comparable à SIFT (qui affine la position).

| Test | Méthode | Recall@1 (50 / 100 / 500 m) | Recall@5 (500 m) | Erreur médiane | Temps par photo |
|---|---|---|---|---|---|
| Validation (ville de Belfort, 100 photos) | SIFT | 64 % / 64 % / 64 % | 64 % | 2 m | 0,15 s |
| Validation (ville) | **DINOv2-S cls** | 28 % / 54 % / 72 % | 95 % | 88 m | **0,010 s** |
| **B (campagne, 200 photos, 2017 → 2020)** | SIFT | 2,5 % / 2,5 % / 2,5 % | 2,5 % | pas de réponse pour 97 % | 0,16 s |
| **B** | **DINOv2-S cls** | **6,5 % / 13,5 % / 22,5 %** | **45,5 %** | 3 446 m | **0,012 s** |

- Hasard (disque / carte) : B 7,9·10⁻⁵ (50 m), 3,1·10⁻⁴ (100 m), 7,9·10⁻³ (500 m) → DINOv2 est ≈ 400× au-dessus du hasard à 100 m sur B, mais **77,5 % des photos restent à plus de 500 m**.
- Les autres variantes sur B (information, non retenues) : `mean` 8,0 / 15,0 / 26,0 % ; `concat` 8,5 / 16,5 / 26,5 %.
- En ville, SIFT reste plus précis (position affinée à 2 m) ; DINOv2 trouve la zone mais pas la position fine, avec toutes les photos et sans seuil de validation.
- Constats visuels : réussite = village + lisière de forêt (repères nets, erreur 41 m) ; échec = forêt pure (aucun repère), réponse à 4,1 km. Images : `data/vis/belfort_B_dino_reussite.png`, `belfort_B_dino_echec.png`.
- Interprétation : l'intuition de l'utilisateur est **en partie confirmée** (réseau profond ≫ SIFT en campagne : 13,5 % contre 2,5 % à 100 m), mais le modèle brut ne suffit pas. Écart « fabriqué → réel » : non mesuré (test A non fait).
- **Pas encore fait** : photos réelles de drone (vol 03) et Val-d'Oise ; mémoire/vitesse sur carte de 100 Mpx ; position affinée.

**Taille du modèle (2026-10-11, demandée par l'utilisateur : « pour acquit de conscience, un plus gros, on pourra le rétrécir »)** — même script (`--model`), mêmes disques, mêmes photos, sans entraînement.
Règle de choix de l'empreinte fixée **avant** de regarder B : meilleure variante au Recall@1 à 100 m sur la **validation** (ville) ; à égalité (≤ 2 points) la plus simple (cls < mean < concat).
Validation (100 photos, à 100 m) : S cls 54 / mean 53 / concat 53 → **cls** ; B (base) 51 / 64 / 57 → **mean** ; L (large) 50 / 57 / 58 → **mean**.

| Modèle (paramètres) | Empreinte | **B (campagne, 200 photos)** : Recall@1 à 50 / 100 / 500 m | Recall@5 à 500 m | Erreur médiane | Validation ville : Recall@1 à 100 / 500 m | GPU max | Temps par photo |
|---|---|---|---|---|---|---|---|
| SIFT (plancher) | — | 2,5 / 2,5 / 2,5 % | 2,5 % | pas de réponse (97 %) | 64 / 64 % | — | 0,16 s |
| DINOv2 petit ViT-S/14 (21 M) | cls | 6,5 / 13,5 / 22,5 % | 45,5 % | 3 446 m | 54 / 72 % | 498 Mo | 12 ms |
| DINOv2 base ViT-B/14 (86 M) | mean | 8,0 / 20,0 / 33,0 % | 57,5 % | 2 367 m | 64 / 82 % | 1 239 Mo | 10 ms |
| DINOv2 large ViT-L/14 (304 M) | mean | 15,0 / 26,5 / 37,5 % | 62,0 % | 1 545 m | 57 / 86 % | 2 772 Mo | 17 ms |

(variante `cls` préenregistrée sur B : S 13,5 / 22,5 % ; B 13,0 / 25,0 % ; L 22,0 / 36,5 % à 100 / 500 m.)
- **La taille aide en campagne** : de petit à large, le Recall@1 à 100 m passe de 13,5 % à 26,5 % et l'erreur médiane de 3,4 km à 1,5 km, mais **62,5 % des photos restent à plus de 500 m** même avec le grand modèle.
- Mémoire GPU et temps mesurés ci-dessus (carte de B, 4 346 disques, batch 128) : le grand modèle tient largement dans 8 Go.
- Effectif : 100 et 200 photos → incertitude de l'ordre de ± 5 à 10 points ; les écarts S → B à 100 m sont à la limite, S → L nets.
- Rétrécir plus tard : possible (distillation d'un petit élève sur les empreintes du grand, quantification) mais c'est une étape d'entraînement supplémentaire, avec perte possible ; à décider après le test sur les vraies photos.

**DINOv2 géant ViT-g/14 (1 136 M paramètres, poids 4,5 Go, utilisé en demi-précision)** — empreinte `mean` (règle de validation : cls 51 / mean 68 / concat 63 à 100 m) :
B (campagne) : 16,0 / 29,5 / 46,0 % à 50 / 100 / 500 m ; Recall@5 (500 m) 74,5 % ; erreur médiane 967 m ; validation ville : 46 / 68 / 90 % ; GPU max 2 674 Mo ; 31 ms par photo.

**Test A — vraies photos de drone, vol 03 UAV-VisLoc (768 photos, 2018-10-23, carte Google ≈ 2021) — `scripts/prep_uav_test.py`**
Carte rééchantillonnée à 0,8 m/px carrés (11 079 × 9 038 px ; pixels d'origine 0,253 × 0,297 m) ; photos ramenées à 0,8 m/px avec 0,115 m/px (échelle mesurée au test rapide SIFT sur ce même vol = « étalonnage de la caméra » que le vrai système connaîtrait par altitude + optique) ;
cap = `Phi1`. Photos : 458 × 305 m. Aucune photo hors carte. Même méthode, mêmes réglages que B ; empreinte choisie par la règle de validation (S cls, B/L/g mean).

| Méthode | Recall@1 à 50 / 100 / 500 m | Recall@5 (500 m) | Erreur médiane | Taux de réponse | Temps par photo |
|---|---|---|---|---|---|
| **SIFT** | 36,2 / 36,7 / 36,7 % | 36,7 % | 19 m (sur les réponses) | **36,7 %**, 0 % de réponses > 100 m | 0,06 s |
| DINOv2 petit (cls) | 14,3 / 32,0 / 53,5 % | 76,3 % | 287 m | 100 % | 11 ms |
| DINOv2 base (mean) | 16,8 / 36,3 / 54,3 % | 76,4 % | 254 m | 100 % | 11 ms |
| DINOv2 large (mean) | 16,4 / 37,4 / 57,2 % | 80,7 % | 184 m | 100 % | 17 ms |
| **DINOv2 géant (mean)** | **23,0 / 49,3 / 69,0 %** | **86,6 %** | **101 m** | 100 % | 31 ms |

Hasard : 1,2·10⁻⁴ (50 m), 4,9·10⁻⁴ (100 m), 1,2·10⁻² (500 m).
- **Surprise** : sur ces vraies photos (zone industrielle et villages), SIFT trouve la bonne position dans 36,7 % des cas, très précisément (19 m) et sans aucune fausse réponse ; il est bien meilleur qu'en campagne (test B : 2,5 %). Le paysage compte plus que « vraie photo ou orthophoto ».
- Le géant (sans entraînement) fait mieux que SIFT à 100 m (49,3 % contre 36,7 %) et à 500 m (69 % contre 36,7 %) mais **répond toujours**, donc se trompe parfois sans le savoir ; SIFT répond moins souvent mais jamais à tort : complémentarité évidente (hybride : à tester plus tard).
- Même ordre de taille que sur B : petit → géant double presque le score à 100 m (32 → 49 %).
- Limites : un seul vol, zone surtout bâtie ; échelle de photo étalonnée sur ce vol ; test non utilisé pour régler quoi que ce soit (variante choisie sur la validation IGN).

**Test C — Val-d'Oise, carte 2018, photos = fenêtres de 2021 (200 photos, bloc 2×2 dalles x 630–640 km, y 6895–6905 km, graine 0, même méthode que B), sans entraînement**

| Méthode | Recall@1 à 50 / 100 / 500 m | Recall@5 (500 m) | Erreur médiane |
|---|---|---|---|
| SIFT | 12,5 / 12,5 / 12,5 % (taux de réponse 12,5 %) | 12,5 % | — |
| DINOv2 base (mean) | 9,5 / 20,5 / 31,5 % | 47,0 % | 2 803 m |
| DINOv2 large (mean) | 10,5 / 20,0 / 31,0 % | 50,5 % | 2 777 m |
| DINOv2 géant (mean) | 10,5 / 20,5 / 34,0 % | 61,5 % | 2 572 m |

Grilles 2018 et 2021 identiques ; décalage médian ≈ −0,2 / −0,4 m (max 6,2 m). Ici la taille du modèle n'aide presque pas à 100 m (20 %), seulement pour le top-5.

## Étape 3 — entraînement des dernières couches (en cours, 2026-10-11)
Décisions de l'utilisateur : on itère vite ; modèle **base** (86 M), 2 derniers blocs ; une 2ᵉ validation rurale ; Val-d'Oise = test C pour l'instant (peut servir à l'entraînement plus tard) ; Belfort 2013 à télécharger ; le GPU (RTX 5070, 8 Go) est sous-utilisé, on peut monter en taille (large) ensuite.
- **Séparation par zone** (`scripts/prep_train_data.py`) : 43 dalles de Belfort ; test B (4 dalles) + validation ville (`985,6735`) + validation rurale (`980,6745`, graine 2 : montagne entièrement boisée avec un village, très dure : 12 % de Recall@1 à 100 m sans entraînement) + leurs 8 voisines (tampon 5 km) exclues → **21 dalles d'entraînement**, éditions 2020, 2017 (puis 2013), mosaïques 0,8 m/px dans `data/work/train/`.
- Entraînement (`scripts/train_dino.py`) : paires (disque 305 m d'une édition récente, même endroit dans une édition plus ancienne décalé jusqu'à 64 m) ; perte InfoNCE (τ = 0,05) avec négatifs du lot + un négatif difficile à 320–800 m ; augmentations (rotation ± 10°, échelle ± 5 %, luminosité, contraste, couleur, flou, niveaux de gris, occultation) ; AdamW lr 2·10⁻⁵, cosinus, 100 pas d'échauffement ; bf16 ; choix du point d'arrêt sur la **validation** (moyenne ville + rural).

**Run 1 (2026-10-11) : DINOv2 base, 2 derniers blocs entraînés (14,2 M paramètres sur 87 M), 1 500 pas × 64 paires, éditions 2020 + 2017, 15,8 min, 3,8 Go de GPU, 0,63 s par pas**
(un bug de marge corrigé en cours de route : les négatifs difficiles sortaient de la dalle ; relance sur les mêmes données).
- Validation (le point d'arrêt est choisi dessus) : score 0,443 (pas 0) → 0,748 (pas 250) → 0,787 (pas 1500, meilleur). Ville : R@1 à 100 m 63 → 96 % ; montagne boisée : 12 → 54 %. Perte 2,4 → 0,10, validation stable de 1 000 à 1 500 pas : pas de sur-apprentissage visible.
- **Tests jamais vus (empreinte `mean`, mêmes photos et mêmes disques qu'avant)** :

| Test | Base sans entraînement : R@1 50/100/500 m — R@5(500) — médiane | **Base entraînée : R@1 50/100/500 m — R@5(500) — médiane** |
|---|---|---|
| B campagne (Belfort 2017 → 2020, 200 photos) | 8,0 / 20,0 / 33,0 % — 57,5 % — 2 367 m | **38,5 / 76,5 / 84,5 % — 95,5 % — 60 m** |
| **A vol 03, vraies photos de drone (768 photos)** | 16,8 / 36,3 / 54,3 % — 76,4 % — 254 m | **32,8 / 69,8 / 81,9 % — 95,4 % — 67 m** |
| C Val-d'Oise (2018 → 2021, 200 photos) | 9,5 / 20,5 / 31,5 % — 47,0 % — 2 803 m | **35,0 / 65,5 / 76,5 % — 91,5 % — 69 m** |

- **Le gain se transfère** : entraîné uniquement sur des orthophotos IGN de Belfort, le modèle double presque ses scores sur les vraies photos de drone (36,3 → 69,8 % à 100 m), sur une carte Google, dans un autre pays, avec un autre capteur ; et sur un autre département (C : 20,5 → 65,5 %).
  Il dépasse SIFT (36,7 % sur A) et le géant sans entraînement (49,3 % sur A). Écart « fabriqué → réel » : B 76,5 % contre A 69,8 % à 100 m, soit ≈ 7 points.
- Précautions : 200 photos par test (incertitude ≈ ± 6 points) ; test B proche des données d'entraînement par le couple d'éditions (2017/2020) et le département, séparé par zone avec un tampon de 5 km ; C (autre département et autres éditions) et A (autre pays, autre capteur) confirment que le gain n'est pas propre à B ; le modèle répond toujours (pas d'abstention) : 23–30 % des réponses sont fausses à 100 m ; l'échelle des photos de A est étalonnée sur ce vol.
- Pas encore fait : 2013 (téléchargé, non préparé), modèle large, plus de pas, hybride SIFT + DINOv2, abstention, filtre de particules.

**Run 2 (arrêté par l'utilisateur) : DINOv2 large (304 M), 4 derniers blocs (50,4 M entraînés), mêmes données IGN que le run 1**
- 1ʳᵉ tentative : 7,9 Go de GPU sur 8 → débordement vers la mémoire centrale, aucun pas affiché en > 30 min (4 blocs × 192 images par pas de 64 paires : activations > 8 Go) ; corrigé par recalcul des activations (`--ckpt-act`, 3,4 Go de GPU).
- 2ᵉ tentative : 1,8 s par pas (≈ 45 min pour 1 500 pas) ; validation au pas 250 : score **0,785** (ville R@1 à 100 m 93 %, montagne boisée 56 %), soit le niveau final du petit modèle (0,787 à 1 500 pas ; 0,748 à 250 pas). Arrêté au pas 300 : « trop long et coûteux ».
- Décision (2026-10-11) : on développe tout le reste avec le petit modèle (base) ; le large sera refait plus tard. Checkpoint partiel supprimé.

**Run 3 (en cours) : DINOv2 base, 2 derniers blocs, entraîné sur de VRAIES photos de drone** — `scripts/calib_uav_gsd.py` (échelle des photos), `scripts/prep_uav_test.py` (mêmes données au format commun), `UavStream` dans `scripts/train_dino.py`.
- Échelle (m/px) des photos : vol 03 0,115 (mesuré, 4 photos) ; vol 04 0,136 (mesuré sur 9/12 photos, 0,133–0,137, règle « même angle par pixel : 0,115 × hauteur/466 m » = 0,134 → écart 1,5 %) ; vol 01 0,095 mesuré (1 photo) contre 0,100 par la règle ;
  vol 11 0,195 mesuré (11/12 photos, étendue 0,157–0,212, « hauteur » CSV 2 573 m = au-dessus de la mer) ; vols 02, 06, 08 : SIFT ne trouve rien → règle de la hauteur (0,100 ; 0,207 ; 0,136) ; vols 05 et 10 : autre caméra (3000 × 2000) et hauteur inutilisable → exclus ; vol 07 (30 photos, carte de 50 m) et vol 09 (carte en 4 fichiers) exclus.
- Séparation : entraînement = vols 01, 02, 06, 08, 11 (≈ 3 900 photos) ; **validation = vol 04** (1 photo sur 4, 185 photos ; voisin du vol 03 mais jamais utilisé pour entraîner, seulement pour choisir le point d'arrêt) ; **test = vol 03**, jamais vu ni pour entraîner ni pour régler.
- Paires : disque de 305 m nord en haut de la photo (cap Phi1 ± 10°, échelle ± 10 %), disque de la carte Google à la position GPS décalée de ≤ 64 m, négatif difficile à 320–800 m ; mêmes augmentations que pour l'IGN.

**Résultat du run 3 (2026-10-11, 16,1 min, 3,8 Go de GPU, 0,65 s par pas)** : validation vol 04 : score 0,400 (pas 0) → 0,749 (pas 750, point retenu) → 0,732 (pas 1 500, léger recul : début de sur-apprentissage). Perte 2,7 → 0,22.
Comparaison propre sur les trois tests (empreinte `mean`, base, mêmes photos / disques ; fichiers de résultats distincts par checkpoint) :

| Test | Sans entraînement | Run 1 : IGN seul | **Run 3 : vrais vols seuls** |
|---|---|---|---|
| **A, vol 03 réel (768 photos)** | 16,8 / 36,3 / 54,3 % — R@5 76,4 % — 254 m | 32,8 / 69,8 / 81,9 % — 95,4 % — 67 m | **33,7 / 71,4 / 82,6 % — 94,8 % — 65 m** |
| B campagne Belfort (200) | 8,0 / 20,0 / 33,0 % — 57,5 % — 2 367 m | **38,5 / 76,5 / 84,5 % — 95,5 % — 60 m** | 25,0 / 48,0 / 59,5 % — 80,0 % — 109 m |
| C Val-d'Oise (200) | 9,5 / 20,5 / 31,5 % — 47,0 % — 2 803 m | **35,0 / 65,5 / 76,5 % — 91,5 % — 69 m** | 24,0 / 44,5 / 53,5 % — 71,0 % — 263 m |

(R@1 à 50 / 100 / 500 m — R@5 à 500 m — erreur médiane.) SIFT sur A : 36,2 / 36,7 / 36,7 %.
- **Sur le test réel (vol 03), les deux entraînements sont équivalents** (71,4 contre 69,8 % à 100 m : écart de 1,6 point, dans l'incertitude d'environ ± 3 points pour 768 photos).
- **L'entraînement sur l'IGN généralise mieux** : sur B et C (orthophotos, autres zones) il fait 76,5 et 65,5 % à 100 m contre 48,0 et 44,5 % pour le modèle entraîné sur les vols ; le modèle « vols » est spécialisé sur le style des cartes Google.
- Sur A, les deux modèles plafonnent à ≈ 70 % à 100 m (82 % à 500 m, 95 % parmi les 5 premières) : l'erreur médiane de 65 m est proche de la limite de la grille (centre d'un disque de 128 m de pas) ; le reste vient probablement d'autres causes (voir run 4, mélange).
- **Run 4 (en cours)** : mélange 50 % IGN / 50 % vrais vols, mêmes réglages, validation = ville + montagne + vol 04 ; checkpoint choisi sur la moyenne des trois.

**Run 4 (2026-10-11, 17,9 min) : mélange 50 % IGN / 50 % vrais vols, base, 2 derniers blocs** — validation (ville / montagne boisée / vol 04) : score 0,766 au pas 1 500 (stable de 1 000 à 1 500).

| Test | Run 1 (IGN seul) | Run 3 (vols seuls) | **Run 4 (mélange)** |
|---|---|---|---|
| **A vol 03 réel** | 32,8 / 69,8 / 81,9 % — R@5 95,4 % — 67 m | 33,7 / 71,4 / 82,6 % — 94,8 % — 65 m | **35,0 / 75,0 / 86,8 % — 97,7 % — 61 m** |
| B campagne Belfort | 38,5 / 76,5 / 84,5 % — 95,5 % — 60 m | 25,0 / 48,0 / 59,5 % — 80,0 % — 109 m | 34,5 / 70,0 / 82,5 % — 95,0 % — 66 m |
| C Val-d'Oise | 35,0 / 65,5 / 76,5 % — 91,5 % — 69 m | 24,0 / 44,5 / 53,5 % — 71,0 % — 263 m | 33,5 / 64,0 / 74,5 % — 88,5 % — 68 m |

Le mélange est le meilleur sur le test réel (75,0 contre 69,8 % à 100 m ; écart à la limite de l'incertitude de ± 3 points) et reste à moins de 6,5 points de l'IGN seul sur B et C (incertitude ± 6). **Modèle de travail retenu : `data/work/ckpt_b14_last2_mix_run4.pt`** (choisi sur la validation).

## Étape 4 — contexte et filtre de particules (2026-10-11) — `scripts/particle_filter.py`
Scénario : GPS coupé à la photo t0 ; le système connaît alors sa dernière position et sa vitesse (3 derniers fixes GPS), puis seulement les photos suivantes. Aucun réentraînement ; descripteurs du modèle de travail (run 4, base, `mean`).
- Observation : similarité cosinus photo / disques de la carte (grille de 128 m, interpolation bilinéaire), poids ∝ (1 − 0,05)·exp((s − s_max)/T) + 0,05 ; rééchantillonnage systématique si ESS < N/2 ; 5 % de particules réinjectées autour de la meilleure réponse « une image à la fois » ; rayon de recherche r0 + vmax·Δt (150 m + 35 m/s) ; estimation = moyenne autour du mode le plus lourd.
- **Premier modèle de mouvement (vitesse constante en vecteur) : mauvais** — sur la validation (vol 04) erreur médiane 214–500 m, ≤ 100 m 41–46 %, donc **pire** qu'une image seule (71 m, 70 %) : le drone fait des lignes avec des demi-tours, la vitesse constante en vecteur est fausse à chaque virage.
- **Modèle retenu : le drone avance dans la direction du cap connu (± 5° de dérive), vitesse scalaire inconnue qui varie lentement.** Vérifié sur le vol 04 (730 paires) : écart cap du déplacement − cap de la photo : médiane − 0,1°, 99 % à moins de 10°, vitesse 14–28 m/s (médiane 22 m/s, pas de temps 5 s).
  Hypothèse à garder en tête : le cap vient du compas (dans ces données `Phi1` égale la direction du déplacement) et le drone avance vers le haut de la photo.
- **Réglage sur la VALIDATION uniquement** (vol 04, 738 photos consécutives, 7 coupures × 150 photos), grille σ_s ∈ {0,1 ; 0,3 ; 0,6}, T ∈ {0,03 ; 0,05 ; 0,1} : tous les réglages sont bons (≤ 100 m : 85–95 %, médiane 35–44 m) ; retenu σ_s = 0,6 m/s·√s, T = 0,1 (95,2 % à ≤ 100 m, 97,6 % à ≤ 500 m, médiane 35 m).
- **Test réel, vol 03, réglage inchangé, une seule exécution** (7 coupures GPS à 3, 100, …, 600 ; 150 photos suivies chacune = 1 050 photos sans GPS) :

| Méthode | Erreur médiane | Moyenne | ≤ 100 m | ≤ 500 m | Dernière photo (médiane) |
|---|---|---|---|---|---|
| Une image à la fois (modèle seul) | 60 m | 335 m | 75,6 % | 87,2 % | 50 m |
| À l'aveugle (cap + vitesse, sans photo) | 1 518 m | 1 984 m | 4,0 % | 23,1 % | 3 663 m |
| **Filtre de particules** | **57 m** | **83 m** | **87,0 %** | **98,3 %** | 48 m |

- Longue durée (coupure à 3, 100, 300 puis tout le reste du vol, 1 898 photos) : filtre 86,8 % ≤ 100 m, 98,1 % ≤ 500 m, médiane 57 m, dernière photo 51 m ; à l'aveugle : 11 km de médiane, 21 km à la fin → **pas de dérive du filtre**.
- Coupures de vol (photos manquantes, le filtre ne fait que se déplacer pendant le trou) : 6 manquantes toutes les 20 → 67,2 % ≤ 100 m, 96,1 % ≤ 500 m ; 20 manquantes toutes les 60 → 59,6 % ≤ 100 m, 91,4 % ≤ 500 m (médiane 84 m). La colonne « une image à la fois » n'est pas dégradée dans ces tableaux (elle garde toutes les photos) : comparaison non équitable à ce niveau.
- Limites : un seul vol réel testé (et un de validation) ; cap supposé connu et fiable (erreur supposée de 5°, non testée au-delà) ; position et vitesse initiales issues du GPS juste avant la coupure ; pas de test de perte totale (position initiale inconnue).
- Animation : `data/vis/filtre_vol03.mp4` (coupure à la photo 100, jusqu'à la fin du vol) : vraie trajectoire (vert), position estimée (rouge), particules (cyan), réponse « une image » (jaune), estimation à l'aveugle (orange).

**Correction du filtre aux virages (2026-10-11, à la demande de l'utilisateur qui voyait des mouvements « systématiques » dans l'animation)**
- **Les données sont saines** : numéros source croissants sans trou (768 / 768), heures croissantes, vitesse médiane 13,8 m/s, `Phi1` ≈ direction du déplacement. La photo 119 (`03_0120.JPG`) a des valeurs normales (dt 4 s, pas 94 m, 23,6 m/s, cap stable).
- **Cause = mon modèle de mouvement, pas les photos** : à 7 endroits du vol 03 (paires 95→96, 191→192, 287→288, 383→384, 479→480, 575→576, 671→672) il y a **un trou de 43–47 s avec un demi-tour de ≈ 150–180° et 240 m de déplacement net (≈ 5 m/s apparent)** ; mon modèle faisait avancer le drone 46 s × 14 m/s en ligne droite (≈ 650 m) → toutes les hypothèses 400–1 000 m trop loin après chaque virage. Même structure sur le vol 04 : 7 trous de 41–45 s, Δcap −180°…+179°.
- Corrections (chacune validée sur le vol 04, pas sur le test) : cap **moyen** sur l'intervalle ; en virage (trou > 15 s ou cap qui tourne de plus de 25°) : corde de l'arc × facteur de 0,2 à 1 (vitesse utile inconnue), bruit de position 0,1·v·dt, 30 % de particules réinjectées **avant** la pondération autour des 3 meilleures réponses ; en régime normal 5 % après rééchantillonnage.
  Essai rejeté sur la validation : injection avant pondération partout (94,8 % ≤ 100 m, moyenne 143 m, contre 96,2 % et 114 m).
- Validation (vol 04) version finale : médiane 35 m, ≤ 100 m 96,2 %, ≤ 500 m 98,6 % (une image seule : 71 m, 70,2 %, 88,1 %).
- **Test vol 03 : trois évaluations** (le modèle a été modifié après avoir vu ses erreurs de virage sur le test ; chaque changement a été validé sur le vol 04) :

| Version | Médiane | Moyenne | ≤ 100 m | ≤ 500 m | Erreur 1ʳᵉ photo après un virage |
|---|---|---|---|---|---|
| v1 : vitesse constante en direction du cap | 57 m | 83 m | 87,0 % | 98,3 % | 280–650 m |
| v2 : + modèle de virage | 65 m | 83 m | 82,6 % | 99,0 % | 280–650 m |
| **v3 (finale) : + injection immédiate après virage** | **66 m** | **79 m** | **83,2 %** | **99,5 %** | **28–144 m** (sauf photo 192) |
| Une image seule (pour mémoire) | 60 m | 335 m | 75,6 % | 87,2 % | — |

  v3 : après coupure puis tout le reste du vol (1 898 photos) : 64 m, 83,5 % ≤ 100 m, 99,5 % ≤ 500 m. Coupures de vol : 6 photos manquantes / 20 → 83 m, 60,5 % ≤ 100 m, 95,6 % ≤ 500 m ; 20 manquantes / 60 → 93 m, 53,4 %, 90,7 %.
  Erreur médiane 4 photos après chaque trou : 100 m ; ailleurs : 66 m (95ᵉ centile 136 m). Le seul virage encore mauvais : photo 192 (5 737 m) — la réponse photo seule y est elle-même fausse de 5 748 m et le filtre la suit une photo, puis revient (83 m à la suivante).
  La précision à 100 m de la v1 (87,0 %) est un peu meilleure que celle de la v3 (83,2 %) sur ce test ; la v3 gagne sur la moyenne, le 500 m et surtout les virages.
- **Photos 117–119 (et 122)** : la ressemblance au vrai endroit y est faible (0,37 à la photo 118 contre 0,6–0,7 d'habitude) car les champs sont **jaunes (riz mûr, octobre 2018) sur la photo et sombres (inondés) sur la carte Google** ; l'IA seule se trompe de 318 à 1 354 m (elle choisit un autre village aux champs bruns). Le filtre les corrige : erreur 31, 27, 13 m aux photos 117, 118, 119 (version finale, coupure à 100). Ce n'est pas un défaut de données.
- Animation `data/vis/filtre_vol03.mp4` régénérée avec la version finale (coupure à la photo 100, jusqu'à la fin du vol).

## Nouvelle ville (taille similaire à Belfort) — en cours
Enquête sur 94 départements (édition RVB 20 cm la plus proche de 2020 ; relevé partiel du catalogue, script supprimé). Taille de téléchargement mesurée :
Paris D075 1,23 Go · Hauts-de-Seine D092 1,79 · Seine-Saint-Denis D093 1,89 · Val-de-Marne D094 2,17 · **Belfort D090 4,07** · **Val-d'Oise D095 7,42** · Essonne D091 9,39 · Yvelines D078 13,1 ;
tous les autres ≥ 16,6 Go (la plupart 25–40 Go). Candidats « paysage différent, taille voisine » : D095 (champs ouverts à plat, banlieue, forêts) ou D093 (urbain dense). **Choix de l'utilisateur : Val-d'Oise (D095), édition 2021** (2 parties, 7,42 Go, `data/raw/ign_d095_2021/`, téléchargement lancé). D095 a aussi des éditions 20 cm 2014 et 2018 (relevé partiel) :
une édition plus ancienne est nécessaire pour le test « autre date » (sinon photo et carte seraient la même image) ; à télécharger si accord.
