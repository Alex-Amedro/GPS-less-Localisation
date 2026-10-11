"""Prepare un vol UAV-VisLoc comme test (meme format que prep_ign_test.py) : vraies photos de drone contre la carte satellite du vol.

La carte est en degres (pixels non carres) : elle est reechantillonnee a --gsd m/px CARRE. Chaque photo est ramenee a la meme echelle avec la taille au sol
d'un pixel --photo-gsd (connue du systeme : altitude + optique ; mesuree au test rapide SIFT du vol 03 : 0,113-0,116 m/px). Le cap donne a la methode est la colonne
Phi1 du CSV (verifie : la rotation photo -> carte trouvee par SIFT est egale a Phi1 a 2-8 degres pres). La position GPS ne sert que de verite terrain.

Sortie dans --out : map.npy, queries/NNN.png, truth.csv (x_m, y_m depuis le coin haut-gauche de la carte), meta.json.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import rasterio
from pyproj import Geod
from rasterio.enums import Resampling


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/raw/uavvisloc"))
    ap.add_argument("--vol", default="03")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gsd", type=float, default=0.8)
    ap.add_argument("--photo-gsd", type=float, default=0.115)
    ap.add_argument("--every", type=int, default=1, help="une photo sur N (reguliere, sans hasard)")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "queries").mkdir(exist_ok=True)
    rows = list(csv.DictReader(open(a.root / a.vol / f"{a.vol}.csv", encoding="utf-8")))[::a.every]
    ds = rasterio.open(a.root / a.vol / f"satellite{a.vol}.tif")
    geod = Geod(ellps="WGS84")
    lat_c, lon_c = (ds.bounds.top + ds.bounds.bottom) / 2, (ds.bounds.left + ds.bounds.right) / 2
    mx = geod.inv(lon_c, lat_c, lon_c + ds.res[0], lat_c)[2]  # m par pixel en x (est-ouest)
    my = geod.inv(lon_c, lat_c, lon_c, lat_c + ds.res[1])[2]  # m par pixel en y (nord-sud)
    W, H = round(ds.width * mx / a.gsd), round(ds.height * my / a.gsd)
    print(f"carte source {ds.width}x{ds.height} px, pixels de {mx:.3f} x {my:.3f} m -> carte de travail {W}x{H} px a {a.gsd} m/px")
    img = ds.read([1, 2, 3], out_shape=(3, H, W), resampling=Resampling.average)
    img = np.ascontiguousarray(np.moveaxis(img, 0, 2))
    np.save(a.out / "map.npy", img)

    out, skipped = [], 0
    for i, r in enumerate(rows):
        lat, lon = float(r["lat"]), float(r["lon"])
        row, col = ds.index(lon, lat)
        x_m, y_m = col * mx, row * my  # position dans la carte de travail, en metres depuis le coin haut-gauche
        if not (0 <= x_m <= W * a.gsd and 0 <= y_m <= H * a.gsd):
            skipped += 1
            continue
        photo = cv2.imread(str(a.root / a.vol / "drone" / r["filename"]))
        f = a.photo_gsd / a.gsd
        small = cv2.resize(photo, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        name = f"{len(out):03d}.png"
        cv2.imwrite(str(a.out / "queries" / name), small)
        out.append(dict(file=name, x_m=round(x_m, 2), y_m=round(y_m, 2), cap_deg=float(r["Phi1"]), expected_scale=1.0, source=r["filename"], date=r["date"], height_m=r["height"]))
    with open(a.out / "truth.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out[0].keys())
        w.writeheader(); w.writerows(out)
    json.dump(dict(gsd=a.gsd, photo_gsd=a.photo_gsd, vol=a.vol, map_shape=[H, W, 3], every=a.every, mx=mx, my=my, skipped_outside_map=skipped), open(a.out / "meta.json", "w"), indent=1)
    print(f"{len(out)} photos ({skipped} hors carte), taille de travail {small.shape[1]}x{small.shape[0]} px = {small.shape[1] * a.gsd:.0f}x{small.shape[0] * a.gsd:.0f} m ; sortie {a.out}")


if __name__ == "__main__":
    main()
