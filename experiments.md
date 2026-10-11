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

**Photos 465–476 du vol 03 : erreur du filtre de 1,9 km aux photos 473–474 (signalée par l'utilisateur, 2026-10-11)**
- Données de vol normales sur cette portion (pas de 94–96 m, 11,7–13,7 m/s, Δcap ≤ 1,8°). La photo 473 : 1 866 m d'erreur du filtre (coupure à 3 ou 100), la réponse photo seule se trompe de 1 251 à 2 189 m sur 8 des 12 photos 465–476.
- **Cause : désaccord d'apparence photo / carte** — photos jaune pâle (riz mûr, oct. 2018) contre carte très sombre (champs inondés) : l'IA ne reconnaît plus l'endroit, ses meilleures réponses sont **dispersées** (décalages (966, −795), (−1 647, −480), (1 500, −422), (1 138, 1 870), (1 417, 1 219), (1 184, 1 460)… m) et la vraie position n'apparaît souvent qu'en 2ᵉ ou 3ᵉ proposition (27 à 96 m). À 473 puis 474 deux réponses fausses voisines (≈ 1,8 km, 310 m d'écart) s'accordent : 100 particules injectées s'ajoutent, la vraie grappe a été amincie par des vraisemblances faibles (0,42–0,57 contre 0,6–0,7 ailleurs) et le mode bascule. Le filtre revient à 59–69 m dès la photo 475.
- Essai « ambiguïté » (écart de ressemblance entre la meilleure et la 2ᵉ réponse > 305 m ; α = clip(écart/0,05 ; 0,15 ; 1) réduit la vraisemblance et/ou le nombre de particules injectées). Résultats (médiane | moyenne | ≤ 100 m | ≤ 500 m | > 1 km), 7 coupures × 150 photos :

| Variante | Validation vol 04 | **Test vol 03** |
|---|---|---|
| Sans ambiguïté (retenue) | 35 m \| 114 m \| 96,2 % \| 98,6 % \| 1,4 % | **66 m \| 79 m \| 83,2 % \| 99,5 % \| 0,5 %** |
| Pondération + injection réduites | 36 m \| 95 m \| 95,4 % \| 98,9 % \| 1,1 % | 72 m \| 105 m \| 74,2 % \| 98,3 % \| 1,5 % |
| Injection seule réduite | 35 m \| 93 m \| 96,1 % \| 98,9 % \| 1,1 % | 68 m \| 86 m \| 80,2 % \| 99,0 % \| 1,0 % |

  Les variantes suppriment bien le saut des photos 473–474 mais **ne sont pas meilleures sur l'ensemble** (neutres à légèrement meilleures sur la validation, moins bonnes sur le test) : **non retenues** (règle du projet : une brique n'est gardée que si elle améliore mesurablement). Défaut par défaut : `ambig=False` (option `--ambig`).
  Ces portions sont précisément ce que le « je ne sais pas » doit signaler. Le test vol 03 a maintenant été regardé 5 fois (v1, v2, v3, deux variantes d'ambiguïté).
- Vidéos conservées pour comparaison dans le README : `data/vis/filtre_vol03_v3_virages_sans_ambiguite.mp4` (version retenue), `data/vis/filtre_vol03_v4_ambiguite_ajout.mp4` (variante à injection réduite). La toute première vidéo (vitesse constante en direction du cap, sans modèle de virage) a été écrasée avant cette règle de conservation et n'est pas reproductible.

## Étape 5 — « je ne sais pas » (2026-10-11) — `scripts/confidence.py`
Ordre de travail demandé par l'utilisateur : 1) « je ne sais pas », 3) hybride SIFT + filtre, 2) robustesse. Aucun commit git (consigne).
- **Signaux sans vérité terrain** enregistrés à chaque photo (`particle_filter.FEAT_NAMES` : étalement du mode, part du poids dans le mode, ESS, ressemblance à la position estimée, accord avec les réponses photo, écart entre les 2 meilleures réponses…).
  « Bonne » réponse = erreur du filtre ≤ 100 m. Données : validation vol 04 (2 384 photos suivies, 95,3 % de bonnes), test vol 03 (2 524 photos suivies, 80,5 % de bonnes ; vol plus difficile), coupures tous les 40 photos, 150 photos suivies.
- **AUROC test (validation)** : part du poids dans le mode **0,846 (0,928)** ; ressemblance à la position estimée 0,744 (0,639) ; écart estimé / 3 réponses 0,761 (0,677) ; étalement 0,756 (0,600) ; ESS 0,558 (0,553) ; virage 0,445 (0,598).
- **Confiance combinée** (régression logistique L2 apprise sur la validation) : AUROC validation 0,938 mais **test 0,814** — moins bonne que le meilleur signal seul, et **mal calibrée sur un autre vol** (confiance annoncée 0,99 contre 0,91 observée ; 0,79 contre 0,61 ; 0,91 contre 0,59) parce que le vol 03 est plus difficile que le 04. Non retenue.
- **Score retenu : « part du poids dans le mode »** (le seul que la validation classe en tête, sans regarder le test). Test, vol jamais vu :

| Couverture (part des photos où on répond) | ≤ 100 m | ≤ 500 m | Erreur médiane | Moyenne | Erreurs > 1 km gardées |
|---|---|---|---|---|---|
| 100 % (sans abstention) | 80,5 % | 99,2 % | 67 m | 88 m | 19 sur 19 |
| 95 % | 82,5 % | 100,0 % | 66 m | 73 m | **0** |
| 90 % | 85,4 % | 100,0 % | 64 m | 70 m | 0 |
| 80 % | 88,9 % | 100,0 % | 62 m | 66 m | 0 |
| 70 % | 91,8 % | 100,0 % | 59 m | 63 m | 0 |
| 50 % | 96,4 % | 100,0 % | 53 m | 57 m | 0 |

- **Seuil fixé sur la validation** (confiance ≥ 0,84 = on s'abstient sur les 10 % de photos les moins sûres de la validation) appliqué tel quel au test : validation → répond à 90,0 %, 98,6 % de bonnes (contre 95,3 %), 0 % de > 1 km (contre 1,1 %) ;
  **test → répond à 76,0 %** (davantage d'abstentions sur le vol plus difficile, ce qui est le bon comportement), **90,4 % de bonnes (contre 80,5 %), 0 % de > 1 km (contre 0,8 %)**, erreur médiane 61 m (contre 67 m).
- Photos 465–476 du vol 03 : toutes en « je ne sais pas » sauf la 465 ; la photo 473 (erreur 1 864 m) a une confiance de 0,26. Le système signale donc cette portion comme douteuse, y compris des photos où le filtre avait en fait raison (45–106 m) : abstention prudente.
- Pour comparaison : la réponse photo seule est bonne à 74,3 % (≤ 100 m) et son écart reponse1 − reponse2 a un AUROC de 0,827 (test).
- Limites : une seule paire validation / test ; la calibration absolue (probabilité annoncée) ne se transfère pas d'un vol à l'autre ; on ne l'utilise que comme classement + seuil ; l'abstention coûte de la couverture (en vol réel, ne pas répondre peut vouloir dire continuer à l'aveugle un moment).
- Vidéo : `data/vis/filtre_vol03_v5_je_ne_sais_pas.mp4` (même scénario que v3 ; la marque rouge devient un cercle gris « JE NE SAIS PAS » sous le seuil 0,84).

## Brique 3 — hybride SIFT + filtre (2026-10-11)
SIFT (`sift_eval.py`, recherche globale photo par photo, filtre cap ± 20° et échelle × 1,5) n'a jamais de fausse réponse quand il répond : validation vol 04 : répond à 64,0 % des photos, erreur médiane 32 m, 0,4 % à plus de 100 m, 0 % à plus de 500 m ;
test vol 03 : répond à 36,7 %, médiane 19 m, 0 % à plus de 100 m (0,06 s par photo).
- Intégration (paramètres fixés a priori, **aucun réglage**) : quand SIFT répond, 10 % des particules sont replacées autour de sa position (± 15 m) avant la pondération, puis pondération gaussienne σ = 30 m avec 2 % d'incertitude (cas rare où SIFT se tromperait).
- Résultats (médiane | moyenne | ≤ 100 m | ≤ 500 m | dernière photo) :

| Vol | Scénario | Photo + cap (filtre seul) | **Hybride photo + cap + SIFT** |
|---|---|---|---|
| Validation 04 | 7 coupures × 150 | 35 m \| 114 m \| 96,2 % \| 98,6 % \| 47 m | **29 m \| 105 m \| 96,8 % \| 98,6 % \| 34 m** |
| Validation 04 | reste du vol | 37 m \| 106 m \| 95,1 % \| 98,6 % \| 86 m | **30 m \| 85 m \| 96,6 % \| 98,8 % \| 80 m** |
| **Test 03** | 7 coupures × 150 | 66 m \| 79 m \| 83,2 % \| 99,5 % \| 57 m | **34 m \| 48 m \| 90,8 % \| 99,8 % \| 39 m** |
| **Test 03** | reste du vol | 64 m \| 84 m \| 83,5 % \| 99,5 % \| 59 m | **33 m \| 53 m \| 89,4 % \| 99,6 % \| 32 m** |

  → **retenu** (améliore mesurablement sur les deux vols) ; l'erreur médiane du test est divisée par 2.
- « Je ne sais pas » avec l'hybride (mêmes règles, seuil fixé sur la validation) : AUROC test du score « part du poids dans le mode » **0,915** (0,846 sans SIFT) ; à 95 % de couverture : 90,7 % ≤ 100 m, 0 erreur > 1 km ; à 80 % : 96,7 % ≤ 100 m, médiane 29 m ; à 50 % : 99,7 % ≤ 100 m.
  Seuil appliqué tel quel au test : répond à 78,8 % des photos, **97,1 % de bonnes (contre 88,0 %), 0 % > 1 km (contre 0,4 %), médiane 29 m (contre 35 m)**. Confiance combinée logistique : AUROC test 0,827 (inférieure au score simple, non retenue).
- Vidéo : `data/vis/filtre_vol03_v6_hybride_sift.mp4` (coupure à la photo 100) ; sur ce scénario : médiane 41 m, ≤ 100 m 90,0 %, ≤ 500 m 100 % (contre 59 m, 86,0 %, 100 % pour la v3).

## Brique 2 — robustesse (2026-10-11) — `scripts/robustness.py`
Modèle inchangé (run 4, aucun réentraînement). Une perturbation à la fois, 7 coupures GPS × 150 photos, même méthode que le reste. Colonnes : médiane / ≤ 100 m / ≤ 500 m / > 1 km ; entre parenthèses l'hybride avec SIFT quand il peut être évalué
(SIFT n'est pas recalculé pour le zoom et l'inclinaison ; pour un biais de cap ≥ 20° SIFT rejette tout par sa propre vérification ± 20°, donc hybride = filtre seul).
Cap erroné utilisé partout (remise nord en haut ET direction du mouvement) ; inclinaison = homographie d'une caméra de champ 60° (supposé) penchée dans une direction aléatoire à chaque photo ; zoom = la zone vue est × f plus large.

| Perturbation | **Test vol 03** | Validation vol 04 |
|---|---|---|
| Aucune (référence) | 66 m / 83,2 % / 99,5 % / 0,5 % (hybride 34 m / 90,8 %) | 35 m / 96,2 % / 98,6 % / 1,4 % (hybride 29 m / 96,8 %) |
| Biais de cap 5° | 61 m / 74,1 % / 98,7 % / 1,0 % (hybride 84,1 %) | 55 m / 88,9 % / 98,7 % (hybride 94,1 %) |
| Biais de cap 10° | 80 m / 64,9 % / 95,5 % / 2,9 % (hybride 78,8 %) | 81 m / 71,4 % / 95,7 % (hybride 90,5 %) |
| **Biais de cap 20°** | 125 m / 30,1 % / 85,6 % / 10,2 % | 118 m / 35,8 % / 90,5 % / 8,2 % |
| Biais de cap 30° | 183 m / 15,9 % / 70,1 % / 22,2 % | 163 m / 20,3 % / 82,1 % / 13,9 % |
| Biais de cap 45° | 709 m / 8,1 % / 44,4 % / 43,9 % | 329 m / 9,9 % / 56,7 % / 31,7 % |
| Bruit de cap 5° (écart-type) | 64 m / 81,4 % / 99,1 % (hybride 90,1 %) | 40 m / 95,3 % / 98,7 % (hybride 97,3 %) |
| Bruit de cap 10° | 61 m / 82,6 % / 98,0 % (hybride 91,2 %) | 42 m / 92,3 % / 97,4 % (hybride 94,9 %) |
| Bruit de cap 20° | 54 m / 80,5 % / 90,2 % / 8,8 % (hybride 83,7 %) | 55 m / 80,0 % / 91,2 % / 8,3 % (hybride 85,6 %) |
| Zone vue × 0,8 / 0,9 / 1,1 | 64 / 64 / 67 m ; 84,4 / 83,9 / 81,5 % | 34 / 34 / 36 m ; 95,7 / 96,3 / 95,7 % |
| Zone vue × 1,25 | 73 m / 77,0 % / 98,6 % | 40 m / 95,9 % / 99,8 % |
| Zone vue × 1,5 | 76 m / 68,7 % / 96,6 % / 3,2 % | 47 m / 93,0 % / 99,3 % |
| Inclinaison 5° / 10° | 70 m / 80,4 % ; 80 m / 67,5 % / 98,2 % | 40 m / 95,6 % ; 55 m / 90,2 % / 98,6 % |
| **Inclinaison 20°** | 112 m / 42,4 % / 93,7 % / 3,7 % | 93 m / 55,9 % / 98,8 % / 1,1 % |
| Inclinaison 30° | 305 m / 13,4 % / 57,7 % / 30,9 % | 177 m / 20,7 % / 88,8 % / 10,3 % |
| Vitesse initiale × 0,5 / 0,75 / 1,25 / 1,5 | 83,2 ± 4 % : 80,3 / 84,1 / 82,7 / 79,5 % ; ≤ 500 m ≥ 99,2 % | 94,0 / 96,4 / 95,6 / 92,0 % ; ≤ 500 m ≥ 98,6 % |
| Dernière position GPS fausse de 100 / 300 m | 82,9 / 80,0 % ; ≤ 500 m ≥ 99,1 % | 95,5 / 93,8 % ; ≤ 500 m 98,6–98,7 % |
| Dernière position GPS fausse de 1 000 m | 67 m / 79,6 % / 96,9 % / 2,0 % (hybride 86,8 %) | 37 m / 91,0 % / 93,7 % / 4,3 % (hybride 93,2 %) |

- **Les deux vraies faiblesses : l'erreur de cap (≥ 20°) et l'inclinaison de la caméra (≥ 20°).** À 10° de biais de cap la précision à 100 m tombe d'environ 18 points (filtre seul) mais reste exploitable avec SIFT ; à 20° elle s'effondre (30–36 %) et 8–10 % des réponses dépassent 1 km.
  Le modèle n'est pas invariant à la rotation (il a besoin du cap) et n'a jamais vu de perspective à l'entraînement.
- Le bruit de cap non biaisé (jusqu'à 10°) est bien toléré ; à 20° il apparaît une queue d'erreurs (8–9 % > 1 km).
- Le zoom (altitude mal estimée jusqu'à × 1,5) coûte peu : 68,7 % contre 83,2 % à × 1,5 sur le test, quasi rien sur la validation ; le reste est compensé par le filtre.
- La vitesse initiale (× 0,5 à × 1,5) et la position GPS initiale (jusqu'à 300 m) sont sans effet notable ; un décalage de 1 km est rattrapé (≥ 96 % ≤ 500 m sur le test).
- **Pistes d'amélioration (non faites)** : (a) augmentations de perspective et de rotation (± 15°) à l'entraînement ; (b) estimer le biais du cap en ligne à partir de la rotation trouvée par SIFT quand il répond (le cap estimé par SIFT est vérifié à 2–8° près sur le vol 03) ; (c) plus de pas de cap dans le filtre (σ_cap).
- Limites : une perturbation à la fois, constante sur tout le vol ; inclinaison simulée par homographie sur des photos déjà quasi verticales (le vol 03 a ≤ 5° de roulis et tangage) ; champ de vue supposé 60°.

## Nettoyage disque et « grosse faute » du filtre : sauts aléatoires aux virages (2026-10-11)
- **Disque** (demande de l'utilisateur : « le projet fait presque 100 Go ») : mesuré 69 Go (projet 59 Go dont archives 48,5 Go, données de travail 10,4 Go, .venv 4,6 Go ; cache des poids 5,8 Go). Supprimés : archives IGN Belfort 2013 et 2017 (8,2 Go), Val-d'Oise 2018 et 2021 (14,9 Go), poids DINOv2 petit, large et géant (5,9 Go) → **28,9 Go libérés** ; conservés : Belfort 2020, UAV-VisLoc complet (22,3 Go), VPAIR (0,9 Go), poids DINOv2 base, les 3 checkpoints (54 Mo chacun).
- **Les « sauts » vus dans la vidéo v4 étaient réels et ALÉATOIRES.** Mon test à une seule graine avait caché le problème : avec 20 graines (coupure à la photo 100, tout le reste du vol), la version retenue v3 fait un saut > 1 km aux photos 190–200 dans **9 tirages sur 20** (v4 : 7 sur 20, v6 hybride : 0 sur 20 mais 18 sur 20 avec une erreur > 1 km ailleurs), pire erreur 5 741 m ;
  la vidéo génère une trajectoire différente de mon test car elle enregistre 300 particules (suite de tirages différente). Cause : à la photo 192 (après un trou de 46 s) la réponse photo seule est fausse de 5 748 m, elle reçoit 30 % de particules réinjectées avant la pondération et gagne un tirage sur deux.
- **Correction (contrainte physique, aucun réentraînement)** : une réponse photo ou SIFT plus éloignée de la dernière position estimée que `vmax × dt + 3 × étalement + 100 m` (vmax = 35 m/s) est impossible et n'est pas réinjectée. Résultats moyennés sur **10 tirages**, 7 coupures × 150 photos (médiane | moyenne | ≤ 100 m | ≤ 500 m | > 1 km) :

| | Validation vol 04 | **Test vol 03** |
|---|---|---|
| Filtre sans contrainte (v3) | 35 m \| 106 m \| 96,2 % ± 0,2 \| 98,7 % \| 1,29 % | 65 m \| 79 m \| 83,4 % ± 0,6 \| 99,5 % \| 0,47 % |
| **Filtre avec contrainte** | 34 m \| 42 m \| 97,0 % ± 0,1 \| 100,0 % \| **0,00 %** | 55 m \| 61 m \| 89,8 % ± 0,3 \| 100,0 % \| **0,00 %** |
| Hybride SIFT sans contrainte | 29 m \| 91 m \| 97,1 % ± 0,2 \| 98,8 % \| 1,16 % | 34 m \| 51 m \| 90,3 % ± 0,3 \| 99,7 % \| 0,31 % |
| **Hybride SIFT avec contrainte (version finale)** | 29 m \| 37 m \| 97,6 % ± 0,1 \| 100,0 % \| **0,00 %** | **32 m \| 42 m \| 94,4 % ± 0,2 \| 100,0 % \| 0,00 %** |

- **« Retours en arrière »** (avancement estimé le long du cap < − 30 m entre deux photos, hors trous) : 2 sur 660 pour la v3 (photos 473 et 577), 5 pour l'hybride sans contrainte ; avancement médian 93–94 m contre 94 m vrai. Ils venaient des mêmes sauts ponctuels ; non retravaillés à part.
- Défaut de la contrainte : elle empêche de « rattraper » une position de départ très fausse (voir robustesse) ; un assouplissement quand la confiance reste basse longtemps est une piste.
- **« Je ne sais pas » avec la version finale** (`confidence.py --sift`, seuil 0,97 fixé sur la validation = on s'abstient sur les 10 % de photos les moins sûres) : AUROC test du score « part du poids dans le mode » 0,891 ; test : répond à 77,3 % des photos, **98,1 % de bonnes (contre 92,7 %)**, médiane 28 m (contre 33 m), 0 erreur > 500 m ; combinée logistique : AUROC test 0,723 (non retenue).
- Photos 465–476 avec la version finale : erreurs du filtre 32–105 m (plus aucun saut) ; le système dit « je ne sais pas » à 10 photos sur 12.
- Vidéos : `filtre_vol03_v7_contrainte_physique.mp4` (filtre + contrainte ; médiane 50 m, 92,0 % ≤ 100 m sur ce tirage), `filtre_vol03_v8_final_hybride_je_ne_sais_pas.mp4` (hybride + contrainte + abstention à 0,97 ; médiane 39 m, 94,0 % ≤ 100 m). GIF dans `docs/` pour le README : avant (v4, saut au virage), après (v7), final (v8).

**Robustesse refaite avec la version finale (test vol 03, 7 coupures × 150, un tirage) :** référence 56 m / 89,8 % ≤ 100 m (hybride 32 m / 94,0 %) ;
biais de cap 5° : 84,0 % (hybride 90,3 %) ; 10° : 72,6 % (85,4 %) ; **20° : 41,0 %, 12,4 % > 1 km** ; 30° : 16,3 % ; 45° : 4,0 %, 71 % > 1 km ;
bruit de cap 5° / 10° / 20° : 89,0 / 88,9 / 88,5 % (insensible) ; zone vue × 1,25 : 87,0 % ; × 1,5 : 78,5 % ; inclinaison 10° : 76,5 % ; **20° : 50,6 %** ; 30° : 15,5 % ;
vitesse initiale × 0,5…1,5 : 86,8–90,2 % ; dernière position GPS fausse de 100 / 300 m : 89,9 / 87,8 % ; **de 1 000 m : 67,9 %, 16,5 % > 1 km** (hybride 81,2 %, 6,0 %) — pire qu'avant la contrainte (le filtre ne peut plus sauter loin).

**Correction de la contrainte physique (même soirée) : trop confiante dans une estimation non confirmée.**
Le test « 20 photos manquantes sur 60 » a révélé un défaut : après un trou, l'estimation a dérivé à l'aveugle (≈ 1,2 km) puis la contrainte comparait les réponses photo à CETTE estimation fausse et rejetait les bonnes réponses (« impossibles ») ; l'erreur restait > 1 km pendant ≈ 40 photos.
Deux corrections de principe : (1) le temps de la contrainte est le temps depuis la dernière **observation utilisée**, pas depuis la dernière photo ; (2) une dérive à l'aveugle `0,15 × vmax × (temps depuis la dernière confirmation)` s'ajoute tant que l'estimation n'a pas été **confirmée** par la photo (accord à moins de 250 m) ou par SIFT (à moins de 100 m).
Résultats (photo + cap, moyenne de 6 tirages ; médiane | moyenne | ≤ 100 m | ≤ 500 m | > 1 km) :

| Scénario | Sans contrainte | Contrainte v1 (dérive nulle) | **Contrainte avec confirmation (finale)** |
|---|---|---|---|
| Test, aucun trou | 65 m \| 79 m \| 83,2 % \| 99,5 % \| 0,48 % | 55 m \| 61 m \| 89,8 % \| 100 % \| 0 % | **56 m \| 62 m \| 89,7 % \| 100 % \| 0 %** |
| Test, 6 manquantes / 20 | 82 m \| 139 m \| 61,6 % \| 96,2 % \| 1,87 % | 70 m \| 104 m \| 68,8 % \| 98,2 % \| 0,19 % | 70 m \| 106 m \| 68,4 % \| 98,3 % \| 0,49 % |
| Test, **20 manquantes / 60** | 89 m \| 172 m \| 54,9 % \| 92,5 % \| 2,19 % | 89 m \| 291 m \| 55,1 % \| **83,9 %** \| **7,27 %** | **80 m \| 151 m \| 60,5 % \| 93,8 % \| 1,11 %** |
| Test, GPS fausse de 1 000 m | 69 m \| 128 m \| 78,2 % \| 94,3 % \| 3,46 % | 290 m \| 675 m \| 55,5 % \| 63,6 % \| 27,3 % | **62 m \| 155 m \| 80,4 % \| 90,8 % \| 5,68 %** |
| Validation, aucun trou | 34 m \| 105 m \| 96,2 % \| 98,7 % \| 1,27 % | 34 m \| 42 m \| 97,0 % \| 100 % \| 0 % | 34 m \| 42 m \| 96,9 % \| 100 % \| 0 % |
| Validation, 20 manquantes / 60 | 44 m \| 119 m \| 84,1 % \| 96,5 % \| 1,00 % | 44 m \| 83 m \| 83,2 % \| 96,3 % \| 0 % | 44 m \| 74 m \| 84,6 % \| 97,4 % \| 0,02 % |
| Validation, GPS fausse de 1 000 m | 38 m \| 151 m \| 91,4 % \| 94,3 % \| 3,31 % | 47 m \| 319 m \| 71,3 % \| 75,5 % \| 15,2 % | 39 m \| 143 m \| 86,1 % \| 89,9 % \| 5,46 % |

→ la version finale est **meilleure ou égale à « sans contrainte » partout** sauf sur la précision à 100 m après une position de départ fausse de 1 km sur la validation (86,1 % contre 91,4 %).
Hybride SIFT final (6 tirages) : test aucun trou 32 m / 94,4 % ≤ 100 m / 100 % ≤ 500 m ; 6 manquantes / 20 : 76,4 % ; 20 / 60 : 61,0 % (≤ 500 m 85,7 % avant la correction de confirmation).
**Robustesse finale (test vol 03, 7 coupures × 150, un tirage)** : référence 55 m / 89,7 % (hybride 32 m / 94,4 %) ; biais de cap 5° : 83,0 % (hybride 89,1 %) ; 10° : 75,3 % (84,8 %) ; 20° : 45,1 %, 1,9 % > 1 km ; 30° : 22,5 %, 16,7 % > 1 km ; 45° : 8,8 %, 39,7 % > 1 km ;
bruit de cap 5° / 10° / 20° : 90,0 / 89,5 / 86,4 % (1,4 % > 1 km à 20°) ; zone vue × 0,8 … × 1,25 : 87,0–90,4 % ; × 1,5 : 77,6 % ; inclinaison 5° / 10° / 20° / 30° : 87,6 / 76,9 / 50,6 / 17,0 % ; vitesse × 0,5 … × 1,5 : 86,9–89,7 % ;
GPS fausse de 100 / 300 / 1 000 m : 89,5 / 88,2 / 80,2 % (hybride 94,5 / 93,4 / 86,7 %). « Je ne sais pas » avec la version finale : répond à 78,1 % des photos du test, **97,9 % de bonnes** (contre 92,6 %), médiane 28 m (contre 33 m), AUROC 0,891.
Vidéos `v7` et `v8` régénérées avec ce code (v7 : 51 m, 88,7 % ≤ 100 m ; v8 : 39 m, 94,0 % ≤ 100 m sur le tirage de la vidéo) ; GIF du README reconstruits à partir de v4 (avant), v7 (après), v8 (final).

## Nouvelle ville (taille similaire à Belfort) — en cours
Enquête sur 94 départements (édition RVB 20 cm la plus proche de 2020 ; relevé partiel du catalogue, script supprimé). Taille de téléchargement mesurée :
Paris D075 1,23 Go · Hauts-de-Seine D092 1,79 · Seine-Saint-Denis D093 1,89 · Val-de-Marne D094 2,17 · **Belfort D090 4,07** · **Val-d'Oise D095 7,42** · Essonne D091 9,39 · Yvelines D078 13,1 ;
tous les autres ≥ 16,6 Go (la plupart 25–40 Go). Candidats « paysage différent, taille voisine » : D095 (champs ouverts à plat, banlieue, forêts) ou D093 (urbain dense). **Choix de l'utilisateur : Val-d'Oise (D095), édition 2021** (2 parties, 7,42 Go, `data/raw/ign_d095_2021/`, téléchargement lancé). D095 a aussi des éditions 20 cm 2014 et 2018 (relevé partiel) :
une édition plus ancienne est nécessaire pour le test « autre date » (sinon photo et carte seraient la même image) ; à télécharger si accord.
