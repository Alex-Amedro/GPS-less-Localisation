"""Correction du cap EN VOL (brique 1). Sans entrainement.

Quand SIFT reconnait une photo, il mesure aussi la rotation photo -> carte, donc le VRAI cap de l'appareil. L'ecart avec le cap annonce par le compas
est le biais du compas. On le moyenne (mediane des dernieres reponses de SIFT, causal : seulement le passe) et on le retire du cap utilise ensuite
pour (1) remettre les photos nord en haut, (2) la direction du mouvement, (3) valider les reponses SIFT.

Banc d'essai : on ajoute un biais connu au cap de chaque vol (0, 10, 20, 30, 45 deg), avec et sans correction, et on mesure le filtre
(photo + cap, et hybride + SIFT). SIFT est execute SANS verification de rotation (`sift_eval.py --rot-tol 180 --out-name results_wide.csv`).
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import particle_filter as PF  # noqa: E402
import robustness as RB  # noqa: E402

K_LAST = 7  # nombre de reponses SIFT recentes utilisees pour la mediane
MIN_ANS = 3
SIFT_ROT_TOL = 20.0


def adiff(a, b):
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def causal_bias(cap_used, rot):
    """Biais estime a chaque photo avec les reponses SIFT disponibles jusqu'a cette photo incluse."""
    n = len(cap_used)
    est = np.zeros(n)
    hist, cur = [], 0.0
    for i in range(n):
        if np.isfinite(rot[i]):
            d = float(adiff(cap_used[i], rot[i]))
            if abs(d) < 75.0:  # ecarte les rotations aberrantes (fausses reponses)
                hist.append(d)
        if len(hist) >= MIN_ANS:
            cur = float(np.median(hist[-K_LAST:]))
        est[i] = cur
    return est


def run(sim_base, grid, cap_cmd, sims, sift_pos, cuts, horizon, seeds):
    P = dict(PF.DEFAULTS)
    sim = dict(sim_base)
    sim["cap"], sim["sims"] = cap_cmd, sims
    out = []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        rs = [PF.run_scenario(sim, grid, c, horizon, P, rng, None, sift=sift_pos) for c in cuts if c + 5 < len(sims)]
        e = np.concatenate([r["e_est"] for r in rs])
        out.append((np.median(e), e.mean(), 100 * np.mean(e <= 100), 100 * np.mean(e <= 500), 100 * np.mean(e > 1000)))
    return np.mean(out, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", default="data/work/visloc_03")
    ap.add_argument("--ckpt", default="data/work/ckpt_b14_last2_mix_run4.pt")
    ap.add_argument("--biases", type=float, nargs="+", default=[0, 10, 20, 30, 45])
    ap.add_argument("--cuts", type=int, nargs="+", default=[3, 100, 200, 300, 400, 500, 600])
    ap.add_argument("--horizon", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()

    B = RB.Bench(a.test, a.ckpt)
    base = B.base
    grid = PF.Grid(base["centers"])
    cap0 = base["cap"].copy()
    rows = list(csv.DictReader(open(Path(a.test) / "results_wide.csv")))
    pos = np.array([[float(r["pred_x"]), float(r["pred_y"])] for r in rows])
    rot = np.array([float(r["rot_deg"]) for r in rows])
    print(f"{a.test} : {len(cap0)} photos, SIFT repond a {100 * np.isfinite(rot).mean():.1f} % ; 7 coupures x 150 photos, {a.seeds} tirages")
    print("biais |  estimation du biais (mediane de |erreur|) | SANS correction : photo+cap | hybride | AVEC correction : photo+cap | hybride")
    print("      |                                            | mediane moy <=100m <=500m >1km | idem                    | idem                   | idem")
    fm = lambda r: "%3.0fm %4.0fm %5.1f%% %5.1f%% %4.1f%%" % tuple(r)
    for b in a.biases:
        cap_used = cap0 + b
        # sans correction : SIFT valide avec le cap errone
        ok_u = np.isfinite(rot) & (np.abs(adiff(rot, cap_used)) <= SIFT_ROT_TOL)
        sift_u = np.where(ok_u[:, None], pos, np.nan)
        sims_u = B.sims(cap_used) if b != 0 else base["sims"]
        r_u1 = run(base, grid, cap_used, sims_u, None, a.cuts, a.horizon, a.seeds)
        r_u2 = run(base, grid, cap_used, sims_u, sift_u, a.cuts, a.horizon, a.seeds)
        # avec correction en vol (causale)
        bh = causal_bias(cap_used, rot)
        cap_c = cap_used - bh
        ok_c = np.isfinite(rot) & (np.abs(adiff(rot, cap_c)) <= SIFT_ROT_TOL)
        sift_c = np.where(ok_c[:, None], pos, np.nan)
        sims_c = B.sims(cap_c)
        r_c1 = run(base, grid, cap_c, sims_c, None, a.cuts, a.horizon, a.seeds)
        r_c2 = run(base, grid, cap_c, sims_c, sift_c, a.cuts, a.horizon, a.seeds)
        resid = np.abs(adiff(cap_c, cap0))
        print(f"{b:5.0f} | biais estime a la fin {bh[-1]:5.1f} ; erreur de cap residuelle mediane {np.median(resid):4.1f} deg | "
              f"{fm(r_u1)} | {fm(r_u2)} | {fm(r_c1)} | {fm(r_c2)}", flush=True)


if __name__ == "__main__":
    main()
