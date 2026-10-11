# GPS-less-Localisation

Localisation visuelle sans GPS : retrouver sa position à partir d'une image aérienne et d'une carte préchargée. Projet en cours (voir `experiments.md` pour l'état et les résultats).

## Données
Les images et cartes ne sont **pas** dans ce dépôt (`data/` est ignoré par git). Où les télécharger :

- **IGN BD ORTHO** (carte, entraînement et tests en France) : https://data.geopf.fr/telechargement/resource/BDORTHO — une archive par département et par édition ; Licence Ouverte Etalab (mention : « BD ORTHO®, IGN »).
- **UAV-VisLoc** (test réel) : https://github.com/IntelliSensing/UAV-VisLoc — article https://arxiv.org/abs/2405.11936.
  Jeu complet (16,4 Go) : https://drive.google.com/file/d/1xYODANyilEMM3CfWh85APwkTHQeLTcCT/view?usp=sharing — échantillon (2,04 Go, un vol) : https://drive.google.com/file/d/16tY7tPZiNIoyAhknvyXnp0jAfccIcHtL/view?usp=sharing.
  Google Drive limite les téléchargements par fichier : en cas de « Quota exceeded », réessayer plus tard. Les cartes viennent de Google : ne pas les redistribuer.
- **VPAIR** (test secondaire, échantillon de 861 Mo) : https://zenodo.org/records/6473989
