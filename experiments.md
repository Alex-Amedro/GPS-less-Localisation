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

## Nouvelle ville (taille similaire à Belfort) — en cours
Enquête sur 94 départements (édition RVB 20 cm la plus proche de 2020 ; relevé partiel du catalogue, script supprimé). Taille de téléchargement mesurée :
Paris D075 1,23 Go · Hauts-de-Seine D092 1,79 · Seine-Saint-Denis D093 1,89 · Val-de-Marne D094 2,17 · **Belfort D090 4,07** · **Val-d'Oise D095 7,42** · Essonne D091 9,39 · Yvelines D078 13,1 ;
tous les autres ≥ 16,6 Go (la plupart 25–40 Go). Candidats « paysage différent, taille voisine » : D095 (champs ouverts à plat, banlieue, forêts) ou D093 (urbain dense). **Choix de l'utilisateur : Val-d'Oise (D095), édition 2021** (2 parties, 7,42 Go, `data/raw/ign_d095_2021/`, téléchargement lancé). D095 a aussi des éditions 20 cm 2014 et 2018 (relevé partiel) :
une édition plus ancienne est nécessaire pour le test « autre date » (sinon photo et carte seraient la même image) ; à télécharger si accord.
