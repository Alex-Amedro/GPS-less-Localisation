"""Etape 1 : evaluation SIFT « photo contre carte entiere ».

Entree (dossier --test) : map.npy (carte RGB a 0,8 m/px), queries/NNN.png (photos a 0,8 m/px), truth.csv (x_m, y_m, cap_deg, expected_scale).
Methode : points SIFT de la carte calcules une fois (blocs sans doublons) ; points de la photo apparies a TOUS ceux de la carte (FLANN, ratio de Lowe) ;
votes par tuiles de 512 m (pas 128 m) ; RANSAC (similitude) sur les meilleures tuiles ; une reponse n'est valide que si echelle et rotation sont
compatibles avec l'echelle attendue et le cap connu. Classement par nombre de correspondances coherentes, sans seuil minimum.
La verite terrain ne sert qu'a calculer l'erreur apres coup.

Reglages FIXES AVANT de voir les resultats (ne pas ajuster sur le test) : voir les constantes ci-dessous.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

GSD = 0.8
RATIO = 0.75  # test du ratio de Lowe
RANSAC_PX = 5.0
SCALE_TOL = 1.5  # echelle acceptee : [attendue / 1,5 ; attendue x 1,5]
ROT_TOL = 20.0  # rotation acceptee : cap +/- 20 degres
CORE, MARGIN = 2250, 125  # blocs de la carte : coeur sans doublon + marge de contexte (px)
CELL_M, TILE_CELLS = 128.0, 4  # cellules de 128 m, tuiles de 4x4 cellules = 512 m
CELL = 160  # px ; recalcule dans main() selon l'echelle de la carte
MAX_CAND, MIN_VOTES = 20, 3
THRESH_M = (50, 100, 500)


def map_features(img, sift):
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    pts, des = [], []
    for cy in range(0, H, CORE):
        for cx in range(0, W, CORE):
            x_lo, y_lo = max(cx - MARGIN, 0), max(cy - MARGIN, 0)
            blk = gray[y_lo:cy + CORE + MARGIN, x_lo:cx + CORE + MARGIN]
            if blk.size == 0 or blk.min() == blk.max():
                continue
            kp, d = sift.detectAndCompute(blk, None)
            if d is None:
                continue
            p = np.float32([k.pt for k in kp])
            keep = (p[:, 0] >= cx - x_lo) & (p[:, 0] < cx - x_lo + CORE) & (p[:, 1] >= cy - y_lo) & (p[:, 1] < cy - y_lo + CORE)
            pts.append(p[keep] + np.float32([x_lo, y_lo]))
            des.append(d[keep])
    return np.vstack(pts), np.vstack(des)


def angle_diff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def locate(q_gray, cap, exp_scale, sift, matcher, mpts, grid):
    """Retourne (candidats valides triees, nb d'appariements apres ratio). Candidat = (inliers, votes, x_m, y_m)."""
    kp, d = sift.detectAndCompute(q_gray, None)
    if d is None or len(kp) < 3:
        return [], 0
    qpts = np.float32([k.pt for k in kp])
    pairs = matcher.knnMatch(d, k=2)
    good = [(i, m[0].trainIdx) for i, m in enumerate(pairs) if len(m) == 2 and m[0].distance < RATIO * m[1].distance]
    if len(good) < MIN_VOTES:
        return [], len(good)
    qi = np.array([g[0] for g in good])
    P = mpts[np.array([g[1] for g in good])]
    ncy, ncx = grid
    hist = np.zeros((ncy, ncx))
    np.add.at(hist, (np.clip((P[:, 1] // CELL).astype(int), 0, ncy - 1), np.clip((P[:, 0] // CELL).astype(int), 0, ncx - 1)), 1)
    c = np.zeros((ncy + 1, ncx + 1))
    c[1:, 1:] = hist.cumsum(0).cumsum(1)
    t = TILE_CELLS
    V = c[t:, t:] - c[:-t, t:] - c[t:, :-t] + c[:-t, :-t]  # votes des tuiles (j, i)
    h, w = q_gray.shape
    out = []
    for _ in range(MAX_CAND):
        j, i = np.unravel_index(np.argmax(V), V.shape)
        votes = V[j, i]
        if votes < MIN_VOTES:
            break
        V[max(0, j - t + 1):j + t, max(0, i - t + 1):i + t] = 0  # tuiles suivantes : non chevauchantes
        x0, y0, x1, y1 = i * CELL, j * CELL, (i + t) * CELL, (j + t) * CELL
        sel = (P[:, 0] >= x0) & (P[:, 0] < x1) & (P[:, 1] >= y0) & (P[:, 1] < y1)
        if sel.sum() < 3:
            continue
        M, inl = cv2.estimateAffinePartial2D(qpts[qi[sel]], P[sel], method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX)
        if M is None:
            continue
        s = float(np.hypot(M[0, 0], M[1, 0]))
        rot = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
        if not (exp_scale / SCALE_TOL <= s <= exp_scale * SCALE_TOL) or angle_diff(rot, cap) > ROT_TOL:
            continue
        ctr = M @ np.array([w / 2, h / 2, 1.0])
        out.append((int(inl.sum()), int(votes), float(ctr[0] * GSD), float(ctr[1] * GSD), rot))
    out.sort(key=lambda r: (-r[0], -r[1]))
    return out, len(qi)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="n'evaluer que les N premieres requetes (mesure du temps)")
    ap.add_argument("--vis", type=Path, default=Path("data/vis"))
    ap.add_argument("--images", action="store_true", help="ecrire une image de reussite et une d'echec (par defaut : aucune image)")
    ap.add_argument("--rot-tol", type=float, default=20.0, help="tolerance de rotation autour du cap (deg) ; 180 = pas de verification de la rotation (sert a ESTIMER un biais de cap)")
    ap.add_argument("--out-name", default="results.csv")
    a = ap.parse_args()

    global GSD, CELL, ROT_TOL
    ROT_TOL = a.rot_tol
    GSD = float(json.load(open(a.test / "meta.json"))["gsd"])
    CELL = int(round(CELL_M / GSD))  # cellule de 128 m quelle que soit l'echelle
    img = np.load(a.test / "map.npy")
    truth = list(csv.DictReader(open(a.test / "truth.csv")))
    if a.limit:
        truth = truth[:a.limit]
    H, W = img.shape[:2]
    sift = cv2.SIFT_create()
    t0 = time.time()
    mpts, mdes = map_features(img, sift)
    t_feat = time.time() - t0
    print(f"carte {W}x{H} px : {len(mpts):,} points SIFT en {t_feat:.0f} s ; descripteurs {mdes.nbytes / 1e9:.2f} Go", flush=True)
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    t0 = time.time()
    matcher.add([mdes])
    matcher.train()
    print(f"index FLANN construit en {time.time() - t0:.0f} s", flush=True)
    grid = (int(np.ceil(H / CELL)), int(np.ceil(W / CELL)))

    res = []
    for k, r in enumerate(truth):
        q = cv2.imread(str(a.test / "queries" / r["file"]))
        qg = cv2.cvtColor(q, cv2.COLOR_BGR2GRAY)
        t0 = time.time()
        cand, ngood = locate(qg, float(r["cap_deg"]), float(r["expected_scale"]), sift, matcher, mpts, grid)
        dt = time.time() - t0
        tx, ty = float(r["x_m"]), float(r["y_m"])
        d = [float(np.hypot(c[2] - tx, c[3] - ty)) for c in cand[:5]]
        res.append(dict(file=r["file"], n_good=ngood, n_valid=len(cand), inliers=cand[0][0] if cand else 0, err_top1=d[0] if d else np.inf,
                        err_best5=min(d) if d else np.inf, time_s=dt, pred_x=cand[0][2] if cand else np.nan, pred_y=cand[0][3] if cand else np.nan,
                        rot_deg=cand[0][4] if cand else np.nan))
        if (k + 1) % 20 == 0:
            print(f"  {k + 1}/{len(truth)} photos, {np.mean([x['time_s'] for x in res]):.2f} s/photo", flush=True)

    with open(a.test / ("results_limit.csv" if a.limit else a.out_name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=res[0].keys())
        w.writeheader(); w.writerows(res)

    e1 = np.array([x["err_top1"] for x in res]); e5 = np.array([x["err_best5"] for x in res]); tm = np.array([x["time_s"] for x in res])
    area = W * H * GSD * GSD
    print(f"\n=== {a.test.name} : {len(res)} photos ===")
    print(f"taux de reponse (au moins une reponse valide) : {np.mean(e1 < np.inf):.1%}")
    print("seuil      | Recall@1 | Recall@5 | hasard (disque / carte)")
    for R in THRESH_M:
        print(f"{R:>4} m     | {np.mean(e1 <= R):7.1%}  | {np.mean(e5 <= R):7.1%}  | {np.pi * R * R / area:.2e}")
    ans = e1[e1 < np.inf]
    print(f"erreur mediane : toutes photos {np.median(e1):.0f} m (inf = sans reponse) ; photos avec reponse {np.median(ans) if len(ans) else float('nan'):.0f} m")
    print(f"temps par photo : median {np.median(tm):.2f} s, moyenne {tm.mean():.2f} s, max {tm.max():.2f} s ; carte : points {t_feat:.0f} s (une fois)")

    # une reussite (la plus forte en correspondances) et un echec (sans reponse ou > 500 m) en images
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = [i for i, x in enumerate(res) if x["err_top1"] <= 50] if a.images else []
    ko = [i for i, x in enumerate(res) if x["err_top1"] > 500] if a.images else []
    pick = [("reussite", max(ok, key=lambda i: res[i]["inliers"]))] if ok else []
    pick += [("echec", ko[0])] if ko else []
    for lab, i in pick:
        r, x = truth[i], res[i]
        tx, ty = float(r["x_m"]) / GSD, float(r["y_m"]) / GSD
        half = int(560 / GSD)
        x0, y0 = int(max(0, tx - half)), int(max(0, ty - half))
        crop = img[y0:y0 + 2 * half, x0:x0 + 2 * half]
        fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
        ax[0].imshow(cv2.cvtColor(cv2.imread(str(a.test / "queries" / r["file"])), cv2.COLOR_BGR2RGB)); ax[0].set_title(f"photo {r['file']} (cap {float(r['cap_deg']):.0f} deg)"); ax[0].axis("off")
        ax[1].imshow(crop); ax[1].plot(tx - x0, ty - y0, "r+", ms=18, mew=2, label="verite")
        if np.isfinite(x["pred_x"]):
            ax[1].plot(x["pred_x"] / GSD - x0, x["pred_y"] / GSD - y0, "x", color="lime", ms=14, mew=3, label=f"reponse SIFT ({x['inliers']} corresp.)")
        ax[1].set_title(f"{lab} : erreur {x['err_top1']:.0f} m" if np.isfinite(x["err_top1"]) else f"{lab} : aucune reponse valide"); ax[1].legend(); ax[1].axis("off")
        fig.tight_layout(); a.vis.mkdir(parents=True, exist_ok=True)
        fig.savefig(a.vis / f"{a.test.name}_{lab}.png", dpi=55); plt.close(fig)
    print("images :", [f"{a.test.name}_{lab}.png" for lab, _ in pick])


if __name__ == "__main__":
    main()
