"""Etape 4 : contexte (derniere position connue + suite de photos) et FILTRE DE PARTICULES.

Scenario : le GPS est coupe a la photo t0. Le systeme connait alors sa derniere position et sa vitesse (estimee sur les derniers fixes GPS), puis plus rien :
seulement les photos suivantes. Chaque particule = une hypothese (x, y, vx, vy). A chaque photo : deplacement a vitesse quasi constante (bruit d'acceleration),
poids selon la ressemblance (cosinus DINOv2) entre la photo et la carte a la position de la particule, reechantillonnage systematique, et un peu de
particules reinjectees autour des meilleures reponses « une image a la fois » (pour se retrouver apres une perte). Le rayon de recherche grandit avec le
temps ecoule depuis la derniere position connue. Aucune verite terrain n'entre dans le filtre apres t0 : elle ne sert qu'a calculer l'erreur.

Compare : (a) « une image a la fois » (meilleur disque), (b) estime a l'aveugle (vitesse constante depuis la derniere position connue), (c) le filtre.
"""
import argparse
import datetime as dtm
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import dino_eval as DE  # noqa: E402

STRIDE = DE.STRIDE_M


# ---------------------------------------------------------------- similarites photo x disques de la carte
def load_sims(test, model_name, ckpt, dev="cuda"):
    cache = test / f"sims_{ckpt.stem if ckpt else model_name}.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    import csv
    gsd = float(json.load(open(test / "meta.json"))["gsd"])
    img = np.load(test / "map.npy")
    H, W = img.shape[:2]
    d, st = int(round(DE.DISK_M / gsd)), int(round(DE.STRIDE_M / gsd))
    mask = DE.disk_mask(d)
    model = torch.hub.load("facebookresearch/dinov2", model_name, verbose=False).eval()
    if ckpt:
        model.load_state_dict(torch.load(ckpt), strict=False)
    model = model.to(dev)
    centers, descs, batch = [], [], []

    def flush():
        descs.append(DE.describe(model, torch.stack(batch), dev)["mean"].cpu())
        batch.clear()
    for y in range(0, H - d + 1, st):
        for x in range(0, W - d + 1, st):
            crop = img[y:y + d, x:x + d]
            s = crop.astype(np.int16).sum(2)
            if (((s >= 762) | (s <= 3))[mask]).mean() > 0.2:
                continue
            batch.append(DE.to_tensor(crop, mask)); centers.append(((x + d / 2) * gsd, (y + d / 2) * gsd))
            if len(batch) == DE.BATCH:
                flush()
    if batch:
        flush()
    D = torch.cat(descs).to(dev)
    rows = list(csv.DictReader(open(test / "truth.csv")))
    qs = []
    for r in rows:
        rgb = cv2.cvtColor(cv2.imread(str(test / "queries" / r["file"])), cv2.COLOR_BGR2RGB)
        qs.append(DE.describe(model, DE.to_tensor(DE.northup_disk(rgb, float(r["cap_deg"]), d), mask)[None], dev)["mean"][0])
    sims = (torch.stack(qs) @ D.T).cpu().numpy().astype(np.float32)
    ts = np.array([dtm.datetime.fromisoformat(r["date"]).timestamp() for r in rows])
    out = dict(sims=sims, centers=np.array(centers, np.float32), truth=np.array([[float(r["x_m"]), float(r["y_m"])] for r in rows]), ts=ts,
               cap=np.array([float(r["cap_deg"]) for r in rows]))
    np.savez(cache, **out)
    return out


class Grid:
    """Les disques sont sur une grille reguliere de pas 128 m : interpolation bilineaire de la similarite a une position quelconque."""

    def __init__(self, centers):
        self.x0, self.y0 = float(centers[:, 0].min()), float(centers[:, 1].min())
        self.ix = np.round((centers[:, 0] - self.x0) / STRIDE).astype(int)
        self.iy = np.round((centers[:, 1] - self.y0) / STRIDE).astype(int)
        self.nx, self.ny = self.ix.max() + 1, self.iy.max() + 1

    def field(self, s):
        g = np.full((self.ny, self.nx), s.min(), np.float32)
        g[self.iy, self.ix] = s
        return g

    def sample(self, g, x, y):
        fx, fy = (x - self.x0) / STRIDE, (y - self.y0) / STRIDE
        out = (fx < -0.5) | (fx > self.nx - 0.5) | (fy < -0.5) | (fy > self.ny - 0.5)
        fx, fy = np.clip(fx, 0, self.nx - 1.001), np.clip(fy, 0, self.ny - 1.001)
        x0, y0 = fx.astype(int), fy.astype(int)
        wx, wy = fx - x0, fy - y0
        s = ((1 - wy) * ((1 - wx) * g[y0, x0] + wx * g[y0, x0 + 1]) + wy * ((1 - wx) * g[y0 + 1, x0] + wx * g[y0 + 1, x0 + 1]))
        s[out] = g.min()
        return s


# ---------------------------------------------------------------- filtre
def mode_estimate(pos, w, radius=150.0, ncand=64):
    """Position la plus plausible = moyenne des particules autour du MODE le plus lourd (la moyenne de tout le nuage est fausse s'il y a plusieurs modes).
    Retourne (position, etalement du mode, part du poids total contenue dans le mode)."""
    idx = np.argsort(-w)[:ncand]
    d = np.hypot(pos[idx, 0][:, None] - pos[None, :, 0], pos[idx, 1][:, None] - pos[None, :, 1])  # (ncand, N)
    mass = ((d < radius) * w[None, :]).sum(1)
    m = d[int(np.argmax(mass))] < radius
    ww = w[m] / w[m].sum()
    c = (ww[:, None] * pos[m]).sum(0)
    return c, float(np.sqrt((ww * ((pos[m] - c) ** 2).sum(1)).sum())), float(mass.max() / w.sum())


# sigma_s et temp ont ete choisis sur la validation (vol 04), jamais sur le test
DEFAULTS = dict(n=2000, sigma0=15.0, sigma_s0=2.0, sigma_s=0.6, sigma_cap=5.0, temp=0.1, p_out=0.05, vmax=35.0, drift_frac=0.15, r0=150.0, inject=0.05, inject_turn=0.3, jit_pos=4.0, jit_speed=0.2,
                gap_ref=0.05, alpha_min=0.15, sigma_sift=30.0, p_sift=0.02, inject_sift=0.1, v0_scale=1.0, p0_err=0.0, gate=True, use_filter=True, ambig=False, ambig_weight=False)


def heading_vec(cap_deg):
    """Direction d'avance dans le repere carte (x = est, y = sud) pour un cap en degres (0 = nord, sens horaire) : le drone avance vers le haut de la photo."""
    r = np.radians(cap_deg)
    return np.stack([np.sin(r), -np.cos(r)], -1)


def run_scenario(sim, grid, t0, horizon, P, rng, observe=None, record=0, sift=None):
    S, centers, truth, ts, cap = sim["sims"], sim["centers"], sim["truth"], sim["ts"], sim["cap"]
    n = len(S)
    last = min(t0 + horizon, n - 1)
    k = 3
    v0 = float(np.hypot(*(truth[t0] - truth[t0 - k])) / (ts[t0] - ts[t0 - k])) * P["v0_scale"]  # vitesse estimee sur les derniers fixes GPS (v0_scale : essai de robustesse)
    p0 = truth[t0].copy()
    if P["p0_err"] > 0:  # essai de robustesse : la derniere position connue est fausse (decalage dans une direction tiree au hasard)
        ang = rng.uniform(0, 2 * np.pi)
        p0 = p0 + P["p0_err"] * np.array([np.cos(ang), np.sin(ang)])
    N = P["n"]
    pos = p0 + rng.normal(0, P["sigma0"], (N, 2))
    spd = np.maximum(v0 + rng.normal(0, P["sigma_s0"], N), 1.0)
    w = np.full(N, 1.0 / N)
    est, spread, top1, dr, parts, frames, feats = [], [], [], [], [], [], []
    dr_pos = p0.copy()
    prev_m, prev_sp = p0.copy(), P["sigma0"]
    last_obs_t = last_valid_t = ts[t0]
    for i in range(t0 + 1, last + 1):
        dt = ts[i] - ts[i - 1]
        # le cap est connu (compas) : le drone avance dans cette direction (a un petit ecart de derive pres), la vitesse est inconnue
        spd = np.maximum(spd + rng.normal(0, P["sigma_s"] * np.sqrt(dt), N), 1.0)
        # cap moyen sur l'intervalle (et non le cap de la fin seule) ; virage = trou de plus de 15 s ou cap qui tourne de plus de 25 degres
        dcap = (cap[i] - cap[i - 1] + 180.0) % 360.0 - 180.0
        mid = cap[i - 1] + dcap / 2.0
        turn = dt > 15.0 or abs(dcap) > 25.0
        u = heading_vec(mid + rng.normal(0, P["sigma_cap"], N))
        if turn:
            # pendant un virage la vitesse utile est inconnue (le drone ralentit et tourne) : corde de l'arc, de 20 % a 100 % de la distance a vitesse nominale
            th = abs(np.radians(dcap))
            chord = np.sin(th / 2) / (th / 2) if th > 1e-3 else 1.0
            fac = rng.uniform(0.2, 1.0, N)
            pos += (spd * fac * chord)[:, None] * dt * u + rng.normal(0, 3.0 + 0.1 * float(spd.mean()) * dt, (N, 2))
        else:
            pos += spd[:, None] * dt * u + rng.normal(0, 3.0 + 0.02 * float(spd.mean()) * dt, (N, 2))
        dr_pos = dr_pos + v0 * dt * heading_vec(mid) * (0.6 if turn else 1.0)
        s = S[i]
        order = np.argsort(-s)[:400]
        top_idx = DE.nms_top(order, centers, 3)
        cands = [centers[q] for q in top_idx]
        cand = cands[0]
        # contrainte physique : le drone ne peut pas avoir parcouru plus de vmax x dt depuis la derniere position estimee (+ son incertitude) ;
        # une reponse photo plus lointaine est impossible (c'etait la cause des sauts de plusieurs km aux virages) et n'est pas reinjectee
        # temps depuis la derniere OBSERVATION utilisee (un trou de photos l'allonge) + une derive a l'aveugle qui grandit tant que l'estimation
        # n'a pas ete CONFIRMEE par une photo (accord estimation / reponse photo a moins de 250 m) : on ne se fie pas a une estimation non confirmee
        dmax = P["vmax"] * (ts[i] - last_obs_t) + 3.0 * prev_sp + 100.0 + P["drift_frac"] * P["vmax"] * (ts[i] - last_valid_t)
        inj = [c for c in cands if np.hypot(*(c - prev_m)) <= dmax] if P["gate"] else cands
        # ambiguite : ecart de ressemblance entre la meilleure reponse et la 2e (a plus d'un diametre) ; faible = on ne sait pas laquelle est la bonne
        gap = float(s[top_idx[0]] - s[top_idx[1]]) if len(top_idx) > 1 else 1.0
        alpha = float(np.clip(gap / P["gap_ref"], P["alpha_min"], 1.0)) if P["ambig"] else 1.0
        if P["use_filter"] and turn and (observe is None or observe(i)):
            # juste apres un virage ou un trou, le modele de mouvement n'est pas fiable : on reinjecte tout de suite (AVANT la ponderation)
            # des particules autour des 3 meilleures reponses « une image a la fois »
            ninj = int(P["inject_turn"] * alpha * N) if inj else 0
            if ninj:
                sel = rng.choice(N, ninj, replace=False)
                which = rng.integers(len(inj), size=ninj)
                pos[sel] = np.array(inj)[which] + rng.normal(0, 40.0, (ninj, 2))
                spd[sel] = float((w * spd).sum()) + rng.normal(0, 1.0, ninj)
        have_sift = sift is not None and np.isfinite(sift[i, 0]) and (observe is None or observe(i))
        if have_sift and P["gate"] and np.hypot(*(sift[i] - prev_m)) > dmax:
            have_sift = False
        if have_sift and P["use_filter"]:
            # SIFT a repondu (et son filtre cap + echelle l'a valide) : mesure de position precise, injectee avant la ponderation
            ninj = int(P["inject_sift"] * N)
            sel = rng.choice(N, ninj, replace=False)
            pos[sel] = sift[i] + rng.normal(0, 15.0, (ninj, 2))
            spd[sel] = float((w * spd).sum()) + rng.normal(0, 1.0, ninj)
        if observe is None or observe(i):
            g = grid.field(s)
            lik = grid.sample(g, pos[:, 0], pos[:, 1])
            a_w = alpha if P["ambig_weight"] else 1.0  # (essai rejete : reduire aussi la ponderation fait perdre en precision)
            w = w * ((1 - P["p_out"]) * np.exp(a_w * (lik - s.max()) / P["temp"]) + P["p_out"])
        if have_sift:
            d2 = ((pos - sift[i]) ** 2).sum(1)
            w = w * ((1 - P["p_sift"]) * np.exp(-d2 / (2 * P["sigma_sift"] ** 2)) + P["p_sift"])
        R = P["r0"] + P["vmax"] * (ts[i] - ts[t0])  # le rayon de recherche grandit avec le temps depuis la derniere position connue
        w[np.hypot(pos[:, 0] - p0[0], pos[:, 1] - p0[1]) > R] = 0.0
        if w.sum() <= 1e-300:  # tout est elimine : on repart autour de la meilleure reponse
            pos = cand + rng.normal(0, 60.0, (N, 2)); w = np.full(N, 1.0 / N)
        w = w / w.sum()
        m, sp, mf = mode_estimate(pos, w)
        est.append(m); spread.append(sp)
        prev_m, prev_sp = m, max(sp, 20.0)
        if observe is None or observe(i):
            last_obs_t = ts[i]
            if np.hypot(*(m - cand)) < 250.0 or (have_sift and np.hypot(*(m - sift[i])) < 100.0):
                last_valid_t = ts[i]  # estimation confirmee par la photo (ou par SIFT)
        # signaux de confiance (aucun n'utilise la verite terrain)
        lik_est = float(grid.sample(grid.field(s), np.array([m[0]]), np.array([m[1]]))[0] - s.max())
        feats.append([sp, mf, 1.0 / ((w ** 2).sum() * N), lik_est, float(np.hypot(*(m - cand))), gap, float(s[top_idx[0]]), float(turn),
                      float(min(np.hypot(*(m - c)) for c in cands))])
        top1.append(cand); dr.append(dr_pos.copy()); frames.append(i)
        if record:
            sel = rng.choice(N, record, p=w)
            parts.append(pos[sel].copy())
        if P["use_filter"]:
            ess = 1.0 / (w ** 2).sum()
            if ess < N / 2:  # reechantillonnage systematique
                u = (rng.random() + np.arange(N)) / N
                idx = np.minimum(np.searchsorted(np.cumsum(w), u), N - 1)
                pos, spd = pos[idx] + rng.normal(0, P["jit_pos"], (N, 2)), np.maximum(spd[idx] + rng.normal(0, P["jit_speed"], N), 1.0)
                w = np.full(N, 1.0 / N)
            ninj = int(P["inject"] * alpha * N) if inj else 0
            if ninj and not turn and (observe is None or observe(i)):  # en regime normal : quelques particules apres le reechantillonnage
                sel = rng.choice(N, ninj, replace=False)
                which = rng.integers(len(inj), size=ninj)
                pos[sel] = np.array(inj)[which] + rng.normal(0, 40.0, (ninj, 2))
                spd[sel] = float((w * spd).sum()) + rng.normal(0, 1.0, ninj)
        else:  # variante sans filtre : on ne garde que la derniere mesure
            pos = cand + rng.normal(0, 40.0, (N, 2)); w = np.full(N, 1.0 / N)
    est, top1, dr = np.array(est), np.array(top1), np.array(dr)
    tr = truth[frames]
    err = lambda a: np.hypot(*(a - tr).T)
    return dict(frames=np.array(frames), est=est, top1=top1, dr=dr, spread=np.array(spread), e_est=err(est), e_top1=err(top1), e_dr=err(dr),
                truth=tr, parts=parts, p0=p0, t0=t0, feat=np.array(feats))


FEAT_NAMES = ["etalement_du_mode_m", "part_du_poids_dans_le_mode", "ess_normalise", "ressemblance_a_la_position_estimee", "distance_estimee_vs_reponse_photo_m",
              "ecart_reponse1_reponse2", "ressemblance_reponse1", "virage", "distance_estimee_vs_3_reponses_m"]


def summarize(res, label):
    cat = lambda k: np.concatenate([r[k] for r in res])
    print(f"\n=== {label} : {len(res)} coupures GPS, {sum(len(r['frames']) for r in res)} photos sans GPS ===")
    print(f"{'methode':36s} | erreur mediane | moyenne | <= 100 m | <= 500 m | derniere photo (mediane)")
    for name, k in (("une image a la fois (disque le + proche)", "e_top1"), ("a l'aveugle (vitesse constante, sans photo)", "e_dr"), ("FILTRE DE PARTICULES", "e_est")):
        e = cat(k)
        last = np.median([r[k][-1] for r in res])
        print(f"{name:36s} | {np.median(e):9.0f} m | {e.mean():6.0f} m | {100 * np.mean(e <= 100):6.1f} % | {100 * np.mean(e <= 500):6.1f} % | {last:9.0f} m")


# ---------------------------------------------------------------- video
def make_video(test, sim, r, out_mp4, step=3, fps=10, thr=None):
    gsd = float(json.load(open(test / "meta.json"))["gsd"])
    img = np.load(test / "map.npy", mmap_mode="r")
    f = 1 / 8
    base = cv2.cvtColor(cv2.resize(np.ascontiguousarray(img[::2, ::2]), None, fx=f * 2, fy=f * 2, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
    base = (base * 0.75).astype(np.uint8)
    sc = gsd * 8.0  # m par pixel d'affichage
    P = lambda a: np.round(np.asarray(a) / sc).astype(int)
    h, w = base.shape[:2]
    vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    fr = None
    for n, i in enumerate(r["frames"]):
        fr = base.copy()
        tp, ep, dp = P(r["truth"][:n + 1]), P(r["est"][:n + 1]), P(r["dr"][:n + 1])
        cv2.polylines(fr, [P(sim["truth"][r["t0"]:r["frames"][-1] + 1]).reshape(-1, 1, 2)], False, (90, 160, 90), 1)
        cv2.polylines(fr, [dp.reshape(-1, 1, 2)], False, (0, 165, 255), 1)
        cv2.polylines(fr, [tp.reshape(-1, 1, 2)], False, (0, 255, 0), 2)
        cv2.polylines(fr, [ep.reshape(-1, 1, 2)], False, (0, 0, 255), 2)
        if r["parts"]:
            for q in P(r["parts"][n]):
                cv2.circle(fr, tuple(int(v) for v in q), 1, (255, 255, 0), -1)
        cv2.circle(fr, tuple(int(v) for v in tp[-1]), 6, (0, 255, 0), 2)
        conf = float(r["feat"][n, 1])  # part du poids du nuage dans le mode le plus lourd
        unsure = thr is not None and conf < thr
        cv2.circle(fr, tuple(int(v) for v in ep[-1]), 5 if not unsure else 9, (0, 0, 255) if not unsure else (180, 180, 180), -1 if not unsure else 2)
        cv2.circle(fr, tuple(int(v) for v in P(r["top1"][n])), 3, (0, 255, 255), -1)
        cv2.circle(fr, tuple(int(v) for v in P(r["p0"])), 7, (255, 255, 255), 2)
        txt = [f"photo {i}  ({n + 1}/{len(r['frames'])} apres la coupure GPS)", f"erreur du filtre : {r['e_est'][n]:.0f} m   (a l'aveugle : {r['e_dr'][n]:.0f} m, une image : {r['e_top1'][n]:.0f} m)"]
        if thr is not None:
            txt.append(f"confiance {conf:.2f} (seuil {thr:.2f}) : " + ("JE NE SAIS PAS" if unsure else "je repond"))
        for k, t in enumerate(txt):
            col = (255, 255, 255) if not (unsure and k == 2) else (120, 120, 255)
            cv2.putText(fr, t, (10, 22 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
        leg = [("vraie trajectoire", (0, 255, 0)), ("ou le drone pense etre (filtre)", (0, 0, 255)), ("a l'aveugle (sans photo)", (0, 165, 255)),
               ("particules (hypotheses)", (255, 255, 0)), ("une image a la fois", (0, 255, 255)), ("derniere position GPS", (255, 255, 255))]
        for k, (t, c) in enumerate(leg):
            cv2.putText(fr, t, (10, h - 12 - 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
        if n % step == 0 or n == len(r["frames"]) - 1:
            vw.write(fr)
    vw.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--model", default="dinov2_vitb14")
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--cuts", type=int, nargs="+", default=[3, 100, 200, 300, 400, 500, 600], help="photo ou le GPS est coupe")
    ap.add_argument("--horizon", type=int, default=150, help="nombre de photos suivies apres la coupure")
    ap.add_argument("--seed", type=int, default=0)
    for k, v in DEFAULTS.items():
        if not isinstance(v, bool):
            ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    ap.add_argument("--no-filter", action="store_true", help="variante : pas de filtre (mesure a chaque photo)")
    ap.add_argument("--no-gate", action="store_true", help="variante : sans la contrainte physique (reponse photo plus lointaine que vmax x dt = non reinjectee)")
    ap.add_argument("--sift", type=Path, default=None, help="results.csv de sift_eval.py : ajoute SIFT comme mesure de position (hybride)")
    ap.add_argument("--ambig", action="store_true", help="variante (rejetee par defaut) : moins de particules reinjectees autour des reponses photo ambigues")
    ap.add_argument("--gap-len", type=int, default=0, help="coupures de vol : nombre de photos manquantes d'affilee")
    ap.add_argument("--gap-every", type=int, default=40, help="...toutes les N photos")
    ap.add_argument("--abstain-thr", type=float, default=None, help="video : seuil de confiance (part du poids dans le mode) sous lequel le systeme dit « je ne sais pas »")
    ap.add_argument("--video-cut", type=int, default=None)
    ap.add_argument("--video-horizon", type=int, default=10000)
    ap.add_argument("--video", type=Path, default=Path("data/vis/filtre.mp4"))
    a = ap.parse_args()

    P = {k: getattr(a, k) for k in DEFAULTS if not isinstance(DEFAULTS[k], bool)}
    P["use_filter"] = not a.no_filter
    P["ambig"] = a.ambig
    P["ambig_weight"] = DEFAULTS["ambig_weight"]
    P["gate"] = not a.no_gate
    t = time.time()
    sim = load_sims(a.test, a.model, a.ckpt)
    grid = Grid(sim["centers"])
    print(f"{len(sim['sims'])} photos x {len(sim['centers'])} disques ({time.time() - t:.0f} s) ; parametres {P}", flush=True)
    rng = np.random.default_rng(a.seed)
    obs = (lambda i: (i % a.gap_every) >= a.gap_len) if a.gap_len else None
    sift = None
    if a.sift:  # reponses SIFT (results.csv de sift_eval.py, memes photos dans le meme ordre) : NaN = pas de reponse
        import csv
        sift = np.array([[float(r["pred_x"]), float(r["pred_y"])] for r in csv.DictReader(open(a.sift))])
        print(f"SIFT : {np.isfinite(sift[:, 0]).mean():.1%} des photos ont une reponse")
    res = [run_scenario(sim, grid, t0, a.horizon, P, rng, obs, sift=sift) for t0 in a.cuts if t0 + 5 < len(sim["sims"])]
    label = a.test.name + (f" ; coupures de vol : {a.gap_len} photos manquantes toutes les {a.gap_every}" if a.gap_len else "")
    summarize(res, label)
    if a.video_cut is not None:
        r = run_scenario(sim, grid, a.video_cut, a.video_horizon, P, np.random.default_rng(a.seed), obs, record=300, sift=sift)
        a.video.parent.mkdir(parents=True, exist_ok=True)
        make_video(a.test, sim, r, a.video, thr=a.abstain_thr)
        print(f"video : {a.video} ({len(r['frames'])} photos)")


if __name__ == "__main__":
    main()
