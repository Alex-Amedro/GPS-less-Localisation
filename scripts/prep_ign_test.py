"""Prepare un test IGN « photo recente contre vieille carte » (tests B et C, et blocs de validation).

Tire (graine fixe) un bloc de size x size dalles contigues present dans les deux editions, construit la carte de l'edition ANCIENNE a --gsd m/px,
puis decoupe N fenetres de 460 x 305 m (taille d'une photo du vol 03) dans l'edition RECENTE, avec position (en metres) et rotation tirees par la meme graine :
les memes requetes sont donc obtenues a n'importe quelle echelle. La rotation est donnee comme « cap ». Verite terrain exacte (meme systeme de coordonnees).
Les dalles extraites sont supprimees a la fin.

Sortie dans --out : map.npy (carte ancienne, RGB uint8), queries/NNN.png, truth.csv (x_m, y_m en metres depuis le coin haut-gauche), meta.json.
"""
import argparse
import csv
import json
import random
import re
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import py7zr
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

Q_W_M, Q_H_M = 460.0, 305.0  # empreinte au sol d'une photo du vol 03
TMP = Path(r"C:\t")  # chemin court : les noms internes des archives depassent sinon la limite Windows
NAME = re.compile(r"-(\d{4})-(\d{4})-LA93")


def list_dalles(archive):
    with py7zr.SevenZipFile(archive) as z:
        return {tuple(int(g) for g in NAME.search(f.filename).groups()): f.filename for f in z.list() if f.filename.lower().endswith(".jp2")}


def extract(archive, names, tag):
    out = TMP / tag
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    with py7zr.SevenZipFile(archive) as z:
        z.extract(path=out, targets=list(names.values()))
    return {k: next(out.rglob(Path(v).name)) for k, v in names.items()}


def mosaic(paths, keys, gsd):
    """Mosaique des dalles a gsd m/px. Retourne (image RGB, (xmin, ymax) en Lambert 93)."""
    bounds = {}
    for k in keys:
        with rasterio.open(paths[k]) as ds:
            bounds[k] = tuple(ds.bounds)
    xmin = min(b[0] for b in bounds.values())
    ymax = max(b[3] for b in bounds.values())
    xmax = max(b[2] for b in bounds.values())
    ymin = min(b[1] for b in bounds.values())
    img = np.zeros((round((ymax - ymin) / gsd), round((xmax - xmin) / gsd), 3), np.uint8)
    for k in keys:
        l, b, r, t = bounds[k]
        h, w = round((t - b) / gsd), round((r - l) / gsd)
        with rasterio.open(paths[k]) as ds:
            arr = ds.read(out_shape=(3, h, w), resampling=Resampling.average)  # lecture decimee : la dalle n'est jamais chargee en entier au-dela du necessaire
        oy, ox = round((ymax - t) / gsd), round((l - xmin) / gsd)
        img[oy:oy + h, ox:ox + w] = np.moveaxis(arr, 0, 2)
    return img, (xmin, ymax)


def valid_mask(img, gsd):
    """1 = pixel utilisable. Hors-territoire = GRANDES zones blanches ou noires pures (ouverture morphologique de 8 m) :
    un toit ssature ou une ombre profonde isoles ne sont pas des zones vides."""
    m = img.astype(np.int16).sum(axis=2)
    extreme = ((m <= 3) | (m >= 762)).astype(np.uint8)
    k = int(round(8.0 / gsd)) | 1
    nodata = cv2.morphologyEx(extreme, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    return (1 - nodata).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive-old", type=Path, required=True)
    ap.add_argument("--archive-new", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gsd", type=float, default=0.8, help="taille au sol d'un pixel de travail (m)")
    ap.add_argument("--size", type=int, default=2, choices=[1, 2], help="cote du bloc en dalles de 5 km")
    ap.add_argument("--exclude-meta", type=Path, default=None, help="meta.json d'un autre test : ses dalles sont exclues (bloc de validation)")
    ap.add_argument("--vis", type=Path, default=Path("data/vis"))
    a = ap.parse_args()

    gsd = a.gsd
    rng = random.Random(a.seed)
    nrng = np.random.default_rng(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "queries").mkdir(exist_ok=True)
    a.vis.mkdir(parents=True, exist_ok=True)
    tag = a.out.name

    t0 = time.time()
    old, new = list_dalles(a.archive_old), list_dalles(a.archive_new)
    common = set(old) & set(new)
    excluded = set()
    if a.exclude_meta:
        excluded = {tuple(k) for k in json.load(open(a.exclude_meta))["block"]}
    if a.size == 2:
        blocks = sorted((x, y) for (x, y) in common if all(k in common for k in [(x + 5, y), (x, y + 5), (x + 5, y + 5)]))
        blocks = [b for b in blocks if not ({b, (b[0] + 5, b[1]), (b[0], b[1] + 5), (b[0] + 5, b[1] + 5)} & excluded)]
    else:
        blocks = sorted(k for k in common if k not in excluded)
    print(f"dalles : ancienne {len(old)}, recente {len(new)}, communes {len(common)} ; blocs {a.size}x{a.size} possibles : {len(blocks)}")
    x0, y0 = rng.choice(blocks)
    keys = [(x0 + dx, y0 + dy) for dy in range(0, 5 * a.size, 5) for dx in range(0, 5 * a.size, 5)]
    print("bloc tire (graine %d) :" % a.seed, keys, f"; echelle {gsd} m/px")

    p_old = extract(a.archive_old, {k: old[k] for k in keys}, tag + "_old")
    p_new = extract(a.archive_new, {k: new[k] for k in keys}, tag + "_new")

    meta_ok = True
    for k in keys:
        with rasterio.open(p_old[k]) as d1, rasterio.open(p_new[k]) as d2:
            same = (d1.bounds == d2.bounds and d1.shape == d2.shape and d1.crs == d2.crs and d1.res == d2.res)
            meta_ok &= same
            print(f"  dalle {k} : meme grille (bornes, taille, CRS, resolution) : {same}")
    img_old, origin = mosaic(p_old, keys, gsd)
    img_new, _ = mosaic(p_new, keys, gsd)
    print(f"mosaiques {img_old.shape[1]}x{img_old.shape[0]} px a {gsd} m/px ({time.time() - t0:.0f} s)")

    m_old, m_new = valid_mask(img_old, gsd), valid_mask(img_new, gsd)
    mask = m_old & m_new
    print(f"pixels valides : ancienne {m_old.mean():.3f}, recente {m_new.mean():.3f}, les deux {mask.mean():.3f}")
    H, W = mask.shape

    # decalage entre editions par correlation de phase (fenetres de 800 m de cote)
    n = max(64, int(round(800 / gsd)) // 2 * 2)
    win = cv2.createHanningWindow((n, n), cv2.CV_32F)
    g1, g2 = (cv2.cvtColor(i, cv2.COLOR_RGB2GRAY).astype(np.float32) for i in (img_old, img_new))
    shifts = []
    for cy in np.linspace(n, H - 2 * n, 3).astype(int):
        for cx in np.linspace(n, W - 2 * n, 3).astype(int):
            if mask[cy:cy + n, cx:cx + n].mean() < 0.999:
                continue
            (dx, dy), resp = cv2.phaseCorrelate(g1[cy:cy + n, cx:cx + n], g2[cy:cy + n, cx:cx + n], win)
            shifts.append((dx * gsd, dy * gsd, resp))
    if shifts:
        s = np.array(shifts)
        print(f"decalage entre editions ({len(s)} fenetres) : dx median {np.median(s[:, 0]):.2f} m, dy median {np.median(s[:, 1]):.2f} m, "
              f"max |decalage| {np.abs(s[:, :2]).max():.2f} m, reponse mediane {np.median(s[:, 2]):.2f}")
    del g1, g2

    # requetes : fenetres tournees de l'edition recente ; position en METRES (memes requetes a toute echelle)
    qw, qh = round(Q_W_M / gsd), round(Q_H_M / gsd)
    W_m, H_m = W * gsd, H * gsd
    margin_m = float(np.hypot(Q_W_M, Q_H_M) / 2) + 4.0  # independant de l'echelle
    # candidats tires d'avance (graine fixe) : une echelle differente ne decale pas la suite des tirages
    K = 60 * a.n
    cand_x = nrng.uniform(margin_m, W_m - margin_m, K)
    cand_y = nrng.uniform(margin_m, H_m - margin_m, K)
    cand_t = nrng.uniform(0, 360, K)
    rows, tries = [], 0
    while len(rows) < a.n:
        if tries >= K:
            raise SystemExit("pas assez de fenetres valides dans ce bloc")
        cx_m, cy_m, theta = float(cand_x[tries]), float(cand_y[tries]), float(cand_t[tries])
        tries += 1
        cx, cy = cx_m / gsd, cy_m / gsd
        c, s_ = np.cos(np.radians(theta)), np.sin(np.radians(theta))
        M = np.array([[c, -s_, cx - (c * qw / 2 - s_ * qh / 2)], [s_, c, cy - (s_ * qw / 2 + c * qh / 2)]])  # pixel requete -> pixel carte
        if cv2.warpAffine(mask, M, (qw, qh), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderValue=0).min() == 0:
            continue
        q = cv2.warpAffine(img_new, M, (qw, qh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        i = len(rows)
        cv2.imwrite(str(a.out / "queries" / f"{i:03d}.png"), cv2.cvtColor(q, cv2.COLOR_RGB2BGR))
        rows.append(dict(file=f"{i:03d}.png", x_m=round(cx_m, 2), y_m=round(cy_m, 2), cap_deg=round(theta, 2), expected_scale=1.0))
    print(f"{len(rows)} requetes ({tries} tirages) de {qw}x{qh} px = {Q_W_M:.0f}x{Q_H_M:.0f} m")

    with open(a.out / "truth.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    np.save(a.out / "map.npy", img_old)
    json.dump(dict(gsd=gsd, block=keys, origin_lambert93_xmin_ymax=origin, map_shape=list(img_old.shape), seed=a.seed, grids_identical=meta_ok,
                   valid_fraction_both=float(mask.mean()), archive_old=str(a.archive_old), archive_new=str(a.archive_new)),
              open(a.out / "meta.json", "w"), indent=1)
    shutil.rmtree(TMP / (tag + "_old"), ignore_errors=True)
    shutil.rmtree(TMP / (tag + "_new"), ignore_errors=True)
    print(f"termine en {time.time() - t0:.0f} s ; dalles extraites supprimees ; sortie : {a.out}")


if __name__ == "__main__":
    main()
