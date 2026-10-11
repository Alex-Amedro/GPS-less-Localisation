"""Etape 3 : prepare les mosaiques d'entrainement (dalles de Belfort, plusieurs editions, 0,8 m/px, uint8) dans --out/<edition>/<x>_<y>.npy.

Separation par zone, sans fuite : sont exclues de l'entrainement les dalles de test (--exclude-meta : bloc de test B), les dalles de validation (--val) et
TOUTES les dalles voisines (8 voisines : zone tampon d'une dalle = 5 km) de celles-ci.
"""
import argparse
import json
import re
import shutil
import time
from pathlib import Path

import numpy as np
import py7zr
import rasterio
from rasterio.enums import Resampling

NAME = re.compile(r"-(\d{4})-(\d{4})-LA93")
TMP = Path(r"C:\t")
GSD = 0.8


def list_dalles(archive):
    with py7zr.SevenZipFile(archive) as z:
        return {tuple(int(g) for g in NAME.search(f.filename).groups()): f.filename for f in z.list() if f.filename.lower().endswith(".jp2")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archives", nargs="+", required=True, help="edition=chemin.7z (ex. 2020=data/raw/...7z)")
    ap.add_argument("--exclude-meta", type=Path, nargs="+", required=True, help="meta.json des blocs de test")
    ap.add_argument("--val", nargs="+", default=["985,6735", "980,6745"], help="dalles de validation x,y")
    ap.add_argument("--out", type=Path, default=Path("data/work/train"))
    ap.add_argument("--list-only", action="store_true")
    a = ap.parse_args()

    arch = {kv.split("=")[0]: Path(kv.split("=", 1)[1]) for kv in a.archives}
    listing = {ed: list_dalles(p) for ed, p in arch.items()}
    common = set.intersection(*[set(v) for v in listing.values()])
    held = {tuple(k) for m in a.exclude_meta for k in json.load(open(m))["block"]} | {tuple(int(v) for v in s.split(",")) for s in a.val}
    banned = {(x + dx, y + dy) for (x, y) in held for dx in (-5, 0, 5) for dy in (-5, 0, 5)}
    train = sorted(k for k in common if k not in banned)
    print(f"dalles communes {len(common)} ; reservees test/validation {len(held)} ; avec zone tampon {len(common & banned)} exclues ; "
          f"{len(train)} dalles d'entrainement : {train}")
    if a.list_only:
        return

    t0 = time.time()
    for ed, p in arch.items():
        d = a.out / ed
        d.mkdir(parents=True, exist_ok=True)
        for n, k in enumerate(train):
            f = d / f"{k[0]}_{k[1]}.npy"
            if f.exists():
                continue
            tmp = TMP / f"train_{ed}"
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
            with py7zr.SevenZipFile(p) as z:
                z.extract(path=tmp, targets=[listing[ed][k]])
            jp2 = next(tmp.rglob(Path(listing[ed][k]).name))
            with rasterio.open(jp2) as ds:
                arr = ds.read(out_shape=(3, round(ds.height * ds.res[1] / GSD), round(ds.width * ds.res[0] / GSD)), resampling=Resampling.average)
            np.save(f, np.moveaxis(arr, 0, 2))
            shutil.rmtree(tmp, ignore_errors=True)
            print(f"  {ed} {k} ({n + 1}/{len(train)}) {time.time() - t0:.0f} s", flush=True)
    json.dump(dict(train=[list(k) for k in train], editions=list(arch), gsd=GSD, held_out=sorted(map(list, held))), open(a.out / "split.json", "w"), indent=1)
    print("TERMINE", f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
