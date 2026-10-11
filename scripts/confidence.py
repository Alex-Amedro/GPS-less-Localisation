"""Etape 5 : « je ne sais pas ». Une confiance par photo, calculee SANS verite terrain, pour decider de repondre ou de s'abstenir.

Signaux (voir particle_filter.FEAT_NAMES) : etalement du nuage de particules, part du poids dans le mode, ESS, ressemblance a la position estimee,
accord avec les reponses « une image a la fois », ecart entre les 2 meilleures reponses, etc. Confiance combinee = regression logistique apprise sur la
VALIDATION (vol 04), puis evaluee sur un vol JAMAIS VU (vol 03) : on mesure AUROC, calibration (fiabilite) et la courbe couverture / erreur.
Une reponse est « bonne » si l'erreur du filtre est <= 100 m.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import particle_filter as PF  # noqa: E402

GOOD_M = 100.0


def build(test, ckpt, cuts_step=40, horizon=150, seed=0, params=None, use_sift=False):
    sim = PF.load_sims(Path(test), "dinov2_vitb14", Path(ckpt))
    sift = None
    if use_sift:  # hybride : reponses SIFT de sift_eval.py (results.csv du dossier de test)
        import csv
        sift = np.array([[float(r["pred_x"]), float(r["pred_y"])] for r in csv.DictReader(open(Path(test) / "results.csv"))])
    grid = PF.Grid(sim["centers"])
    P = dict(PF.DEFAULTS)
    P.update(params or {})
    rng = np.random.default_rng(seed)
    X, E, E1, SC = [], [], [], []
    for t0 in range(3, len(sim["sims"]) - 100, cuts_step):
        r = PF.run_scenario(sim, grid, t0, horizon, P, rng, None, sift=sift)
        f = r["feat"]
        lik5 = np.array([f[max(0, k - 4):k + 1, 3].mean() for k in range(len(f))])  # ressemblance moyenne sur les 5 dernieres photos
        d5 = np.array([f[max(0, k - 4):k + 1, 4].mean() for k in range(len(f))])  # accord moyen avec la reponse photo
        X.append(np.column_stack([np.log1p(f[:, 0]), f[:, 1], f[:, 2], f[:, 3], np.log1p(f[:, 4]), f[:, 5], f[:, 6], f[:, 7], np.log1p(f[:, 8]), lik5, np.log1p(d5)]))
        E.append(r["e_est"]); E1.append(r["e_top1"]); SC.append(np.full(len(f), t0))
    return np.vstack(X), np.concatenate(E), np.concatenate(E1), np.concatenate(SC)


NAMES = ["log(etalement du mode)", "part du poids dans le mode", "ESS normalise", "ressemblance a la position estimee", "log(ecart estime / reponse photo)",
         "ecart reponse1 - reponse2", "ressemblance de la reponse 1", "virage", "log(ecart estime / 3 reponses)", "ressemblance moyenne (5 photos)", "log(ecart moyen 5 photos)"]


def auroc(score, y):
    o = np.argsort(score)
    ranks = np.empty(len(score)); ranks[o] = np.arange(1, len(score) + 1)
    pos = y.astype(bool)
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit_logistic(X, y, lam=1.0, it=60):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.column_stack([(X - mu) / sd, np.ones(len(X))])
    w = np.zeros(Z.shape[1])
    for _ in range(it):  # Newton / IRLS avec penalite L2
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (p - y) + lam * np.r_[w[:-1], 0]
        H = (Z * (p * (1 - p))[:, None]).T @ Z + lam * np.diag(np.r_[np.ones(Z.shape[1] - 1), 0]) + 1e-6 * np.eye(Z.shape[1])
        w -= np.linalg.solve(H, g)
    return mu, sd, w


def predict(model, X):
    mu, sd, w = model
    return 1 / (1 + np.exp(-(np.column_stack([(X - mu) / sd, np.ones(len(X))]) @ w)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val", default="data/work/uav_val/visloc_04_full")
    ap.add_argument("--test", default="data/work/visloc_03")
    ap.add_argument("--ckpt", default="data/work/ckpt_b14_last2_mix_run4.pt")
    ap.add_argument("--precision", type=float, default=0.95, help="precision visee sur la validation pour fixer le seuil de reponse")
    ap.add_argument("--sift", action="store_true", help="version hybride : SIFT est une mesure de position du filtre")
    a = ap.parse_args()

    Xv, Ev, E1v, _ = build(a.val, a.ckpt, use_sift=a.sift)
    Xt, Et, E1t, _ = build(a.test, a.ckpt, use_sift=a.sift)
    yv, yt = (Ev <= GOOD_M).astype(float), (Et <= GOOD_M).astype(float)
    print(f"validation : {len(yv)} photos suivies, {100 * yv.mean():.1f} % de bonnes reponses (<= {GOOD_M:.0f} m) ; test : {len(yt)} photos, {100 * yt.mean():.1f} %")

    print("\nAUROC (1 = la confiance classe parfaitement les bonnes avant les mauvaises ; 0,5 = hasard), signe choisi sur la validation :")
    for k, n in enumerate(NAMES):
        sv = auroc(Xv[:, k], yv)
        sg = 1 if sv >= 0.5 else -1
        print(f"  {n:38s} validation {max(sv, 1 - sv):.3f} | test {auroc(sg * Xt[:, k], yt) if sg > 0 else 1 - auroc(Xt[:, k], yt):.3f}")
    model = fit_logistic(Xv, yv)
    pv, pt = predict(model, Xv), predict(model, Xt)
    print(f"\nCONFIANCE COMBINEE (regression logistique apprise sur la validation) : AUROC validation {auroc(pv, yv):.3f} | TEST (vol jamais vu) {auroc(pt, yt):.3f}")
    print(f"  score de Brier test {np.mean((pt - yt) ** 2):.3f} (sans information : {np.mean((yt.mean() - yt) ** 2):.3f})")

    print("\nCALIBRATION sur le test : confiance annoncee / part reellement bonne par tranche")
    for lo, hi in ((0, .5), (.5, .7), (.7, .85), (.85, .95), (.95, 1.01)):
        m = (pt >= lo) & (pt < hi)
        if m.sum():
            print(f"  [{lo:.2f}, {min(hi, 1):.2f}) : {m.sum():5d} photos, annoncee {pt[m].mean():.2f}, observee {yt[m].mean():.2f}")

    print("\nCOURBE COUVERTURE / ERREUR sur le test (on ne repond qu'aux photos les plus sures) :")
    print("  couverture | photos | <= 100 m | <= 500 m | erreur mediane | erreur moyenne | erreurs > 1 km gardees")
    o = np.argsort(-pt)
    for c in (1.0, 0.95, 0.9, 0.8, 0.7, 0.5):
        k = o[:int(round(c * len(o)))]
        print(f"  {100 * c:8.0f} % | {len(k):6d} | {100 * np.mean(Et[k] <= 100):6.1f} % | {100 * np.mean(Et[k] <= 500):6.1f} % | {np.median(Et[k]):9.0f} m | {Et[k].mean():9.0f} m | {int((Et[k] > 1000).sum()):4d} sur {int((Et > 1000).sum())}")

    # seuil fixe sur la validation, applique tel quel au test (la calibration tient-elle sur un vol jamais vu ?)
    ov = np.argsort(-pv)
    cum = np.cumsum(yv[ov]) / np.arange(1, len(ov) + 1)
    ok = np.where(cum >= a.precision)[0]
    thr = pv[ov][ok.max()] if len(ok) else 1.0
    kv, kt = pv >= thr, pt >= thr
    print(f"\nSEUIL FIXE sur la validation pour {100 * a.precision:.0f} % de precision (confiance >= {thr:.2f}) :")
    print(f"  validation : repond a {100 * kv.mean():.1f} % des photos, {100 * yv[kv].mean():.1f} % de bonnes, > 1 km : {100 * np.mean(Ev[kv] > 1000):.1f} %")
    print(f"  TEST       : repond a {100 * kt.mean():.1f} % des photos, {100 * yt[kt].mean():.1f} % de bonnes, > 1 km : {100 * np.mean(Et[kt] > 1000):.1f} % (erreur mediane {np.median(Et[kt]):.0f} m)")
    print(f"  TEST, sans abstention : {100 * yt.mean():.1f} % de bonnes, > 1 km : {100 * np.mean(Et > 1000):.1f} %")
    # comparaison avec la reponse photo seule + sa propre confiance (ecart entre les 2 meilleures reponses)
    gv, gt = Xv[:, 5], Xt[:, 5]
    y1t = (E1t <= GOOD_M).astype(float)
    print(f"\nPOUR COMPARAISON, reponse photo seule (sans filtre) : bonnes {100 * y1t.mean():.1f} %, AUROC de son ecart reponse1-reponse2 : {auroc(gt, y1t):.3f} (test)")

    # --- score simple : le seul signal que la VALIDATION classe en tete (aucun reglage sur le test) ---
    aucs = [max(auroc(Xv[:, k], yv), 1 - auroc(Xv[:, k], yv)) for k in range(Xv.shape[1])]
    kb = int(np.argmax(aucs))
    sg = 1 if auroc(Xv[:, kb], yv) >= 0.5 else -1
    sv_, st_ = sg * Xv[:, kb], sg * Xt[:, kb]
    print(f"\nSCORE SIMPLE = « {NAMES[kb]} » (meilleur AUROC sur la validation : {aucs[kb]:.3f}) ; AUROC test {auroc(st_, yt):.3f}")
    ot = np.argsort(-st_)
    print("  couverture | <= 100 m | <= 500 m | erreur mediane | erreur moyenne | erreurs > 1 km gardees")
    for c in (1.0, 0.95, 0.9, 0.8, 0.7, 0.5):
        k = ot[:int(round(c * len(ot)))]
        print(f"  {100 * c:8.0f} % | {100 * np.mean(Et[k] <= 100):6.1f} % | {100 * np.mean(Et[k] <= 500):6.1f} % | {np.median(Et[k]):9.0f} m | {Et[k].mean():9.0f} m | {int((Et[k] > 1000).sum()):4d} sur {int((Et > 1000).sum())}")
    print("  SEUIL FIXE sur la validation (on s'abstient sur les 10 % de photos les moins sures de la validation) puis applique tel quel au test :")
    th = np.quantile(sv_, 0.10)
    kv2, kt2 = sv_ >= th, st_ >= th
    print(f"    validation : repond a {100 * kv2.mean():.1f} %, {100 * yv[kv2].mean():.1f} % de bonnes (contre {100 * yv.mean():.1f} %), > 1 km : {100 * np.mean(Ev[kv2] > 1000):.1f} % (contre {100 * np.mean(Ev > 1000):.1f} %)")
    print(f"    TEST       : repond a {100 * kt2.mean():.1f} %, {100 * yt[kt2].mean():.1f} % de bonnes (contre {100 * yt.mean():.1f} %), > 1 km : {100 * np.mean(Et[kt2] > 1000):.1f} % (contre {100 * np.mean(Et > 1000):.1f} %), "
          f"erreur mediane {np.median(Et[kt2]):.0f} m (contre {np.median(Et):.0f} m)")

    # les photos 465-476 du vol 03 (coupure a 3)
    sim = PF.load_sims(Path(a.test), "dinov2_vitb14", Path(a.ckpt))
    sift = None
    if a.sift:
        import csv
        sift = np.array([[float(x["pred_x"]), float(x["pred_y"])] for x in csv.DictReader(open(Path(a.test) / "results.csv"))])
    r = PF.run_scenario(sim, PF.Grid(sim["centers"]), 3, 10000, dict(PF.DEFAULTS), np.random.default_rng(0), None, sift=sift)
    fr = list(r["frames"])
    print(f"\nPhotos 465 a 476 du vol 03 (coupure a 3), seuil de reponse = {th:.2f} sur « {NAMES[kb]} » :")
    for i in range(465, 477):
        k = fr.index(i)
        print(f"  photo {i} : valeur {sg * r['feat'][k, 1]:.2f} -> {'repond' if sg * r['feat'][k, 1] >= th else 'JE NE SAIS PAS'} ; erreur du filtre {r['e_est'][k]:.0f} m")


if __name__ == "__main__":
    main()
