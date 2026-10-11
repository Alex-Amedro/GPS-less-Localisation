"""Etape 6 (robustesse) : on degrade ce que le systeme recoit et on mesure la tenue du filtre.

Perturbations (une a la fois) : biais de cap, bruit de cap, mauvaise altitude (zoom), inclinaison de la camera (deformation de perspective),
vitesse initiale fausse, derniere position GPS fausse. Le modele est inchange (aucun reentrainement). Les photos sont modifiees AVANT d'etre decrites ;
le cap errone est utilise partout (remise nord en haut ET direction du mouvement), comme avec un compas defectueux.

Hybride (SIFT) : les reponses SIFT precalculees avec le bon cap sont conservees seulement si le cap erone reste dans la tolerance de SIFT (+/- 20 deg) ;
pour le zoom et l'inclinaison SIFT n'est pas recalcule : seul le filtre photo + cap est evalue.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import dino_eval as DE  # noqa: E402
import particle_filter as PF  # noqa: E402

FOV_DEG = 60.0  # champ de vue horizontal suppose pour simuler l'inclinaison (camera inconnue)
GRAY = (124, 116, 104)


def disk(photo, cap_deg, d, scale=1.0):
    h, w = photo.shape[:2]
    c, sn = np.cos(np.radians(cap_deg)), np.sin(np.radians(cap_deg))
    R = np.array([[c, sn], [-sn, c]]) * scale  # scale > 1 : la zone vue est plus large (altitude sous-estimee)
    t = np.array([(w - 1) / 2, (h - 1) / 2]) - R @ np.array([(d - 1) / 2, (d - 1) / 2])
    return cv2.warpAffine(photo, np.hstack([R, t[:, None]]), (d, d), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=GRAY)


def tilt(photo, deg, direction_deg):
    """Camera inclinee de `deg` degres dans la direction `direction_deg` : homographie K R K^-1 autour du centre."""
    h, w = photo.shape[:2]
    f = 0.5 * w / np.tan(np.radians(FOV_DEG / 2))
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    a = np.radians(deg)
    ax = np.array([-np.sin(np.radians(direction_deg)), np.cos(np.radians(direction_deg)), 0.0])  # axe de rotation dans le plan image
    Rm, _ = cv2.Rodrigues(ax * a)
    H = K @ Rm @ np.linalg.inv(K)
    return cv2.warpPerspective(photo, H, (w, h), flags=cv2.INTER_LINEAR, borderValue=GRAY)


class Bench:
    def __init__(self, test, ckpt, dev="cuda"):
        self.test, self.dev = Path(test), dev
        base = PF.load_sims(self.test, "dinov2_vitb14", Path(ckpt))  # fournit aussi le cache des similarites de reference
        self.base = base
        gsd = float(__import__("json").load(open(self.test / "meta.json"))["gsd"])
        self.d = int(round(DE.DISK_M / gsd))
        self.mask = DE.disk_mask(self.d)
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False).eval()
        self.model.load_state_dict(torch.load(ckpt), strict=False)
        self.model = self.model.to(dev)
        # descripteurs de la carte (inchanges) : recalcules une fois
        img = np.load(self.test / "map.npy")
        H, W = img.shape[:2]
        st = int(round(DE.STRIDE_M / gsd))
        descs, batch, cent = [], [], []

        def flush():
            descs.append(DE.describe(self.model, torch.stack(batch), dev)["mean"]); batch.clear()
        for y in range(0, H - self.d + 1, st):
            for x in range(0, W - self.d + 1, st):
                crop = img[y:y + self.d, x:x + self.d]
                s = crop.astype(np.int16).sum(2)
                if (((s >= 762) | (s <= 3))[self.mask]).mean() > 0.2:
                    continue
                batch.append(DE.to_tensor(crop, self.mask)); cent.append(((x + self.d / 2) * gsd, (y + self.d / 2) * gsd))
                if len(batch) == DE.BATCH:
                    flush()
        if batch:
            flush()
        self.D = torch.cat(descs)
        assert len(cent) == len(base["centers"])
        rows = list(csv.DictReader(open(self.test / "truth.csv", encoding="utf-8")))
        self.photos = [cv2.cvtColor(cv2.imread(str(self.test / "queries" / r["file"])), cv2.COLOR_BGR2RGB) for r in rows]
        sp = self.test / "results.csv"
        self.sift = np.array([[float(r["pred_x"]), float(r["pred_y"])] for r in csv.DictReader(open(sp))]) if sp.exists() else None

    def sims(self, cap_used, scale=1.0, tilt_deg=0.0, seed=0):
        rng = np.random.default_rng(seed)
        qs = []
        for i, p in enumerate(self.photos):
            if tilt_deg > 0:
                p = tilt(p, tilt_deg, rng.uniform(0, 360))
            qs.append(DE.to_tensor(disk(p, cap_used[i], self.d, scale), self.mask))
        out = []
        for k in range(0, len(qs), 64):
            out.append(DE.describe(self.model, torch.stack(qs[k:k + 64]), self.dev)["mean"])
        Q = torch.cat(out)
        return (Q @ self.D.T).cpu().numpy().astype(np.float32)


def evaluate(sim, sift, P, cuts, horizon, seed=0):
    grid = PF.Grid(sim["centers"])
    rng = np.random.default_rng(seed)
    res = [PF.run_scenario(sim, grid, c, horizon, P, rng, None, sift=sift) for c in cuts if c + 5 < len(sim["sims"])]
    e = np.concatenate([r["e_est"] for r in res])
    return np.median(e), e.mean(), 100 * np.mean(e <= 100), 100 * np.mean(e <= 500), 100 * np.mean(e > 1000)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", default="data/work/visloc_03")
    ap.add_argument("--ckpt", default="data/work/ckpt_b14_last2_mix_run4.pt")
    ap.add_argument("--cuts", type=int, nargs="+", default=[3, 100, 200, 300, 400, 500, 600])
    ap.add_argument("--horizon", type=int, default=150)
    a = ap.parse_args()

    B = Bench(a.test, a.ckpt)
    base = B.base
    cap0 = base["cap"].copy()
    P = dict(PF.DEFAULTS)
    rows = []

    def run(name, val, cap_used=None, scale=1.0, tilt_deg=0.0, sift_mask=None, **over):
        cap_used = cap0 if cap_used is None else cap_used
        sim = dict(base)
        sim["cap"] = cap_used
        if cap_used is not cap0 or scale != 1.0 or tilt_deg > 0:
            sim["sims"] = B.sims(cap_used, scale, tilt_deg)
        Pq = dict(P); Pq.update(over)
        r1 = evaluate(sim, None, Pq, a.cuts, a.horizon)
        r2 = None
        if B.sift is not None and scale == 1.0 and tilt_deg == 0:
            s = B.sift.copy()
            if sift_mask is not None:
                s[~sift_mask] = np.nan
            r2 = evaluate(sim, s, Pq, a.cuts, a.horizon)
        rows.append((name, val, r1, r2))
        f = lambda r: "%4.0f m %4.0f m %5.1f%% %5.1f%% %4.1f%%" % r if r else "     (non evalue)"
        print(f"{name:34s} {str(val):>6s} | filtre photo+cap : {f(r1)} | hybride +SIFT : {f(r2)}", flush=True)

    n = len(cap0)
    rng = np.random.default_rng(0)
    print(f"{a.test} : {n} photos ; colonnes : mediane, moyenne, <=100 m, <=500 m, >1 km")
    run("reference (aucune perturbation)", "-")
    for b in (5, 10, 20, 30, 45):
        run("biais de cap (deg)", b, cap_used=cap0 + b, sift_mask=np.full(n, b < 20))
    for s_ in (5, 10, 20):
        noise = rng.normal(0, s_, n)
        run("bruit de cap, ecart-type (deg)", s_, cap_used=cap0 + noise, sift_mask=np.abs(noise) <= 20)
    for f_ in (0.8, 0.9, 1.1, 1.25, 1.5):
        run("zoom : zone vue x (altitude mal estimee)", f_, scale=f_)
    for t_ in (5, 10, 20, 30):
        run("inclinaison de la camera (deg)", t_, tilt_deg=t_)
    for v_ in (0.5, 0.75, 1.25, 1.5):
        run("vitesse initiale x", v_, v0_scale=v_)
    for e_ in (100, 300, 1000):
        run("derniere position GPS fausse de (m)", e_, p0_err=float(e_))


if __name__ == "__main__":
    main()
