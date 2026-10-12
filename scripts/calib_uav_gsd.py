"""Mesure la taille au sol d'un pixel des photos de chaque vol UAV-VisLoc (utile pour fabriquer des paires d'ENTRAINEMENT, pas pour evaluer).

Pour N photos regulierement espacees : SIFT contre une fenetre de carte de 1,5 km centree sur la position GPS (le GPS sert ici d'etiquette),
plusieurs echelles d'essai ; la transformation de similitude donne l'echelle : gsd_photo = s * gsd_essai. Accepte si >= 8 correspondances coherentes
et placement a moins de 150 m de la verite. Resultat : mediane et nombre de photos retenues par vol.
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import rasterio
from pyproj import Geod
from rasterio.enums import Resampling
from rasterio.windows import Window

MAP_GSD, WIN_M, RATIO, RANSAC_PX = 0.8, 1500.0, 0.75, 5.0
MULTS = (0.15, 0.3, 0.6, 1.0, 2.0)


def window(ds, lat, lon, mx, my):
    row, col = ds.index(lon, lat)
    n = int(round(WIN_M / MAP_GSD))
    win = Window(col - WIN_M / mx / 2, row - WIN_M / my / 2, WIN_M / mx, WIN_M / my)
    img = ds.read([1, 2, 3], window=win, out_shape=(3, n, n), boundless=True, fill_value=0, resampling=Resampling.average)
    return cv2.cvtColor(np.moveaxis(img, 0, 2), cv2.COLOR_RGB2GRAY)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/raw/uavvisloc"))
    ap.add_argument("--vols", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--mults", type=float, nargs="+", default=list(MULTS), help="facteurs d'echelle essayes autour de l'essai de depart")
    a = ap.parse_args()

    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    geod = Geod(ellps="WGS84")
    for vol in a.vols:
        rows = list(csv.DictReader(open(a.root / vol / f"{vol}.csv", encoding="utf-8")))
        tifs = sorted((a.root / vol).glob("satellite*.tif"))
        if len(tifs) != 1:
            print(vol, "carte en plusieurs fichiers : ignore")
            continue
        ds = rasterio.open(tifs[0])
        lat_c, lon_c = (ds.bounds.top + ds.bounds.bottom) / 2, (ds.bounds.left + ds.bounds.right) / 2
        mx, my = geod.inv(lon_c, lat_c, lon_c + ds.res[0], lat_c)[2], geod.inv(lon_c, lat_c, lon_c, lat_c + ds.res[1])[2]
        try:
            h = float(np.median([float(r["height"]) for r in rows if r["height"].replace(".", "", 1).isdigit()]))
        except Exception:
            h = 466.0
        w0 = cv2.imread(str(a.root / vol / "drone" / rows[0]["filename"])).shape[1]
        guess = 0.115 * (3976 / w0) * min(h, 1000.0) / 466.0  # meme angle par pixel que le vol 03 (hauteur plafonnee : certaines « hauteurs » sont au-dessus de la mer)
        est = []
        for i in np.linspace(0, len(rows) - 1, a.n).round().astype(int):
            r = rows[i]
            lat, lon = float(r["lat"]), float(r["lon"])
            mg = window(ds, lat, lon, mx, my)
            if mg.min() == mg.max():
                continue
            photo = cv2.cvtColor(cv2.imread(str(a.root / vol / "drone" / r["filename"])), cv2.COLOR_BGR2GRAY)
            kp2, d2 = sift.detectAndCompute(mg, None)
            best = None
            for m in a.mults:
                g = guess * m
                f = g / MAP_GSD
                small = cv2.resize(photo, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
                if min(small.shape) < 100:
                    continue
                kp1, d1 = sift.detectAndCompute(small, None)
                if d1 is None or d2 is None or len(kp1) < 5 or len(kp2) < 5:
                    continue
                good = [p[0] for p in flann.knnMatch(d1, d2, k=2) if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
                if len(good) < 8:
                    continue
                M, inl = cv2.estimateAffinePartial2D(np.float32([kp1[x.queryIdx].pt for x in good]), np.float32([kp2[x.trainIdx].pt for x in good]),
                                                     method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX)
                if M is None or inl.sum() < 8:
                    continue
                c = M @ np.array([small.shape[1] / 2, small.shape[0] / 2, 1.0])
                off = np.hypot(c[0] - mg.shape[1] / 2, c[1] - mg.shape[0] / 2) * MAP_GSD
                if off > 150:
                    continue
                s = float(np.hypot(M[0, 0], M[1, 0]))
                if best is None or inl.sum() > best[0]:
                    best = (int(inl.sum()), s * g)
            if best:
                est.append(best[1])
        med = float(np.median(est)) if est else float("nan")
        print(f"vol {vol} : {len(est)}/{a.n} photos retenues ; taille au sol d'un pixel = {med:.3f} m/px "
              f"(etendue {min(est):.3f}-{max(est):.3f}) ; hauteur CSV {h:.0f} m ; essai de depart {guess:.3f}" if est else
              f"vol {vol} : 0/{a.n} photos retenues (SIFT n'a rien trouve) ; hauteur CSV {h:.0f} m", flush=True)


if __name__ == "__main__":
    main()
