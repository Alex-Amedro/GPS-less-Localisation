"""Evaluations finales du filtre (briques 2, 3, 4) :
  gate    : contrainte physique assouplie quand l'estimation n'est plus confirmee depuis lost_s secondes (relocalisation)
  smooth  : lissage arriere (hors ligne) : erreur et « retours en arriere » avec / sans lissage
  ci      : intervalles de confiance a 95 % (bootstrap sur les coupures GPS) du tableau principal, avec et sans « je ne sais pas »
Les reglages se choisissent sur la validation (vol 04) ; le test (vol 03) ne sert qu'a confirmer.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import particle_filter as PF  # noqa: E402

CKPT = "data/work/ckpt_b14_last2_mix_run4.pt"
FLIGHTS = (("VALIDATION vol 04", "data/work/uav_val/visloc_04_full"), ("TEST vol 03", "data/work/visloc_03"))
CUTS = (3, 100, 200, 300, 400, 500, 600)


def load(test):
    sim = PF.load_sims(Path(test), "dinov2_vitb14", Path(CKPT))
    sift = np.array([[float(r["pred_x"]), float(r["pred_y"])] for r in csv.DictReader(open(Path(test) / "results.csv"))])
    return sim, PF.Grid(sim["centers"]), sift


def metrics(e):
    return np.median(e), e.mean(), 100 * np.mean(e <= 100), 100 * np.mean(e <= 500), 100 * np.mean(e > 1000)


FMT = "%3.0f m | moy %4.0f m | <=100 m %5.1f %% | <=500 m %5.1f %% | >1 km %4.2f %%"


def cmd_gate(a):
    scen = [("aucun trou", None, {}), ("20 manquantes / 60", lambda i: (i % 60) >= 20, {}), ("position GPS fausse de 1 000 m", None, dict(p0_err=1000.0)),
            ("tout le reste du vol (coupure 3 et 100)", None, dict(_long=True))]
    for lab, test in FLIGHTS:
        sim, grid, sift = load(test)
        print("\n" + lab + " (photo + cap, moyenne de %d tirages)" % a.seeds)
        for sname, obs, extra in scen:
            for ls in (1e9, 150.0, 100.0, 60.0):
                o = []
                for s in range(a.seeds):
                    rng = np.random.default_rng(s)
                    P = dict(PF.DEFAULTS); P["lost_s"] = ls; P.update({k: v for k, v in extra.items() if not k.startswith("_")})
                    cuts, hor = ((3, 100), 10000) if "_long" in extra else (CUTS, 150)
                    rs = [PF.run_scenario(sim, grid, c, hor, P, rng, obs) for c in cuts if c + 5 < len(sim["sims"])]
                    o.append(metrics(np.concatenate([r["e_est"] for r in rs])))
                print("  %-38s | perdu apres %6s s : %s" % (sname, "jamais" if ls > 1e8 else int(ls), FMT % tuple(np.mean(o, 0))))


def cmd_smooth(a):
    for lab, test in FLIGHTS:
        sim, grid, sift = load(test)
        cap, ts = sim["cap"], sim["ts"]
        print("\n" + lab + " (hybride SIFT, %d tirages, coupures %s, 150 photos)" % (a.seeds, CUTS))
        ef, es, bf, bs, n = [], [], 0, 0, 0
        for s in range(a.seeds):
            rng = np.random.default_rng(s)
            for c in CUTS:
                if c + 5 >= len(sim["sims"]):
                    continue
                r = PF.smooth_scenario(sim, grid, c, 150, dict(PF.DEFAULTS), rng, sift=sift)
                fr = r["frames"]
                u = PF.heading_vec(cap[fr]); dt = np.diff(ts[fr]); ok = dt < 15
                for est, acc in ((r["est"], "f"), (r["smooth"], "s")):
                    adv = np.array([(est[k] - est[k - 1]) @ u[k] for k in range(1, len(est))])
                    back = int(((adv < -30) & ok).sum())
                    if acc == "f":
                        bf += back
                    else:
                        bs += back
                n += int(ok.sum())
                ef.append(r["e_est"]); es.append(r["e_smooth"])
        ef, es = np.concatenate(ef), np.concatenate(es)
        print("  filtre en vol   : " + FMT % metrics(ef) + " | retours en arriere > 30 m : %d / %d photos" % (bf, n))
        print("  apres lissage   : " + FMT % metrics(es) + " | retours en arriere > 30 m : %d / %d photos" % (bs, n))


def cmd_ci(a):
    rng = np.random.default_rng(0)
    for lab, test in FLIGHTS:
        sim, grid, sift = load(test)
        n = len(sim["sims"])
        cuts = list(range(3, n - 100, 40))
        print("\n" + lab + " : %d coupures GPS (toutes les 40 photos), 150 photos chacune, %d tirages ; IC a 95 %% par bootstrap sur les coupures" % (len(cuts), a.seeds))
        meth = {"a l'aveugle": [], "une photo": [], "filtre photo+cap": [], "hybride SIFT": []}
        conf = []  # (erreur hybride, confiance) pour « je ne sais pas »
        for c in cuts:
            per = {k: [] for k in meth}
            ch, cf = [], []
            for s in range(a.seeds):
                r0 = PF.run_scenario(sim, grid, c, 150, dict(PF.DEFAULTS), np.random.default_rng(s), None)
                r1 = PF.run_scenario(sim, grid, c, 150, dict(PF.DEFAULTS), np.random.default_rng(s), None, sift=sift)
                per["a l'aveugle"].append(r0["e_dr"]); per["une photo"].append(r0["e_top1"]); per["filtre photo+cap"].append(r0["e_est"]); per["hybride SIFT"].append(r1["e_est"])
                ch.append(r1["e_est"]); cf.append(r1["feat"][:, 1])
            for k in meth:
                meth[k].append(np.concatenate(per[k]))
            conf.append((np.concatenate(ch), np.concatenate(cf)))

        def ci(fn, groups):
            vals = np.array([fn(g) for g in groups])
            bs = [np.mean(vals[rng.integers(len(vals), size=len(vals))]) for _ in range(2000)]
            return vals.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
        print("  %-18s | mediane des erreurs (m)         | <= 100 m (%%)                | <= 500 m (%%)" % "methode")
        for k, g in meth.items():
            m, l, h = ci(lambda x: np.median(x), g); p, pl, ph = ci(lambda x: 100 * np.mean(x <= 100), g); q, ql, qh = ci(lambda x: 100 * np.mean(x <= 500), g)
            print("  %-18s | %5.0f  [%5.0f ; %5.0f]        | %5.1f  [%5.1f ; %5.1f]     | %5.1f  [%5.1f ; %5.1f]" % (k, m, l, h, p, pl, ph, q, ql, qh))
        thr = a.thr
        ans = [(e[c >= thr], len(e)) for e, c in conf]
        cov = np.array([len(x) / m for x, m in ans])
        print("  hybride + « je ne sais pas » (seuil %.2f) : repond a %.1f %% des photos [%.1f ; %.1f]" % (thr, 100 * cov.mean(), *(100 * np.percentile(cov, [2.5, 97.5]))))
        keep = [x for x, _ in ans if len(x)]
        p, pl, ph = ci(lambda x: 100 * np.mean(x <= 100), keep); m, l, h = ci(lambda x: np.median(x), keep)
        print("      parmi les reponses : %.1f %% <= 100 m [%.1f ; %.1f], mediane %.0f m [%.0f ; %.0f]" % (p, pl, ph, m, l, h))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["gate", "smooth", "ci"])
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--thr", type=float, default=0.97, help="seuil « je ne sais pas » (part du poids dans le mode), fixe sur la validation")
    a = ap.parse_args()
    {"gate": cmd_gate, "smooth": cmd_smooth, "ci": cmd_ci}[a.what](a)


if __name__ == "__main__":
    main()
