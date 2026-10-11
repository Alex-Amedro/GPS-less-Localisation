"""Etape 2 : version 0 sans AUCUN entrainement. Modele pre-entraine DINOv2 petit (ViT-S/14) : photo -> empreinte -> tuile la plus proche de la carte.

Meme entree que sift_eval.py (--test : map.npy, queries/, truth.csv, meta.json) donc meme carte, memes photos, meme metrique.
Methode : la carte est decoupee en DISQUES de 305 m (stride 128 m), nord en haut. La photo est remise nord en haut avec le cap connu, puis on ne garde que le
disque central de 305 m (la seule partie toujours visible quelle que soit la rotation). Disques de la photo et de la carte : meme masque, meme taille (224 px),
meme normalisation. Similarite cosinus ; reponses = meilleurs disques non chevauchants. La verite terrain ne sert qu'a l'erreur apres coup.

Variantes d'empreinte evaluees en meme temps (a choisir sur les blocs de VALIDATION, pas sur le test) : cls, mean, concat.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

DISK_M, STRIDE_M, INPUT, BATCH = 305.0, 128.0, 224, 128
NMS_M = DISK_M  # deux reponses doivent etre a plus d'un diametre l'une de l'autre
THRESH_M = (50, 100, 500)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
VARIANTS = ("cls", "mean", "concat")


def disk_mask(d):
    yy, xx = np.mgrid[:d, :d]
    c = (d - 1) / 2
    return ((xx - c) ** 2 + (yy - c) ** 2) <= (d / 2) ** 2


def to_tensor(rgb_disk, mask):
    """rgb uint8 (d, d, 3) -> tenseur normalise (3, INPUT, INPUT) ; hors disque = couleur moyenne."""
    x = rgb_disk.astype(np.float32) / 255.0
    x[~mask] = MEAN
    x = cv2.resize(x, (INPUT, INPUT), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1).copy())


@torch.no_grad()
def describe(model, batch, dev):
    with torch.autocast("cuda", dtype=torch.float16):
        o = model.forward_features(batch.to(dev, dtype=next(model.parameters()).dtype))
    cls = torch.nn.functional.normalize(o["x_norm_clstoken"].float(), dim=1)
    mean = torch.nn.functional.normalize(o["x_norm_patchtokens"].float().mean(1), dim=1)
    return dict(cls=cls, mean=mean, concat=torch.nn.functional.normalize(torch.cat([cls, mean], 1), dim=1))


def northup_disk(photo_rgb, cap_deg, d):
    """Disque central d x d de la photo, remis nord en haut (rotation -cap)."""
    h, w = photo_rgb.shape[:2]
    c, s = np.cos(np.radians(cap_deg)), np.sin(np.radians(cap_deg))
    R = np.array([[c, s], [-s, c]])  # rotation de -cap : vecteur carte -> vecteur photo
    ctr_out = np.array([(d - 1) / 2, (d - 1) / 2])
    t = np.array([(w - 1) / 2, (h - 1) / 2]) - R @ ctr_out
    A = np.hstack([R, t[:, None]])
    return cv2.warpAffine(photo_rgb, A, (d, d), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)


def nms_top(order, centers, k=5):
    keep = []
    for i in order:
        if all(np.hypot(*(centers[i] - centers[j])) > NMS_M for j in keep):
            keep.append(i)
            if len(keep) == k:
                break
    return keep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--vis", type=Path, default=Path("data/vis"))
    ap.add_argument("--variant", default="", help="variante pour les images (sinon la 1re)")
    ap.add_argument("--images", action="store_true", help="ecrire une image de reussite et une d'echec (par defaut : aucune image)")
    ap.add_argument("--model", default="dinov2_vits14", help="dinov2_vits14 | dinov2_vitb14 | dinov2_vitl14 | dinov2_vitg14")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--ckpt", type=Path, default=None, help="parametres entraines (train_dino.py) a charger par-dessus le modele pre-entraine")
    a = ap.parse_args()
    short = a.model.replace("dinov2_", "") + (("_" + a.ckpt.stem) if a.ckpt else "")

    gsd = float(json.load(open(a.test / "meta.json"))["gsd"])
    img = np.load(a.test / "map.npy")
    H, W = img.shape[:2]
    truth = list(csv.DictReader(open(a.test / "truth.csv")))
    d = int(round(DISK_M / gsd))
    stride = int(round(STRIDE_M / gsd))
    mask = disk_mask(d)
    dev = "cuda"
    model = torch.hub.load("facebookresearch/dinov2", a.model, verbose=False).eval()
    if a.ckpt:
        missing = model.load_state_dict(torch.load(a.ckpt), strict=False)
        print(f"checkpoint {a.ckpt.name} charge : {len(torch.load(a.ckpt))} tenseurs", flush=True)
    model = (model.half() if a.model.endswith("g14") else model).to(dev)  # le geant (1,1 G) ne tient en fp32 qu'avec peu de marge sur 8 Go
    print(f"modele {a.model} : {sum(p.numel() for p in model.parameters()) / 1e6:.0f} M parametres", flush=True)

    # --- carte : disques sur une grille, descripteurs ---
    t0 = time.time()
    centers, descs = [], {v: [] for v in VARIANTS}
    batch, bc = [], []
    ext = lambda c: ((c.astype(np.int16).sum(2) >= 762) | (c.astype(np.int16).sum(2) <= 3))
    flush = lambda: (descs.update({v: descs[v] + [o[v].cpu()] for o in [describe(model, torch.stack(batch), dev)] for v in VARIANTS}), batch.clear())
    for y in range(0, H - d + 1, stride):
        for x in range(0, W - d + 1, stride):
            crop = img[y:y + d, x:x + d]
            if ext(crop)[mask].mean() > 0.2:  # hors territoire (zone blanche) : on saute
                continue
            batch.append(to_tensor(crop, mask))
            centers.append(((x + d / 2) * gsd, (y + d / 2) * gsd))
            if len(batch) == a.batch:
                flush()
    if batch:
        flush()
    centers = np.array(centers)
    D = {v: torch.cat(descs[v]).to(dev) for v in VARIANTS}
    t_map = time.time() - t0
    print(f"carte {W}x{H} px : {len(centers)} disques de {DISK_M:.0f} m (pas {STRIDE_M:.0f} m) en {t_map:.0f} s ; "
          f"memoire GPU max {torch.cuda.max_memory_allocated() / 1e6:.0f} Mo ; descripteurs {D['cls'].shape[1]} (cls/mean), {D['concat'].shape[1]} (concat)", flush=True)

    # --- photos ---
    res = {v: [] for v in VARIANTS}
    times = []
    for r in truth:
        rgb = cv2.cvtColor(cv2.imread(str(a.test / "queries" / r["file"])), cv2.COLOR_BGR2RGB)
        t0 = time.time()
        q = describe(model, to_tensor(northup_disk(rgb, float(r["cap_deg"]), d), mask)[None], dev)
        sims = {v: (D[v] @ q[v][0]).cpu().numpy() for v in VARIANTS}
        times.append(time.time() - t0)
        tx, ty = float(r["x_m"]), float(r["y_m"])
        for v in VARIANTS:
            top = nms_top(np.argsort(-sims[v]), centers, 5)
            err = [float(np.hypot(centers[i, 0] - tx, centers[i, 1] - ty)) for i in top]
            res[v].append(dict(err1=err[0], err5=min(err), top=top[0], sim=float(sims[v][top[0]])))

    area = W * H * gsd * gsd
    print(f"\n=== {a.test.name} : {len(truth)} photos ; temps par photo (pretraitement + modele + recherche) : mediane {np.median(times) * 1000:.1f} ms ===")
    print("variante | seuil | Recall@1 | Recall@5 | hasard(@1)   | erreur mediane (top-1)")
    for v in VARIANTS:
        e1 = np.array([x["err1"] for x in res[v]]); e5 = np.array([x["err5"] for x in res[v]])
        for R in THRESH_M:
            print(f"{v:8s} | {R:4d} m | {np.mean(e1 <= R):7.1%}  | {np.mean(e5 <= R):7.1%}  | {np.pi * R * R / area:.1e}     | {np.median(e1):.0f} m" if R == THRESH_M[0] else
                  f"{'':8s} | {R:4d} m | {np.mean(e1 <= R):7.1%}  | {np.mean(e5 <= R):7.1%}  | {np.pi * R * R / area:.1e}")
    with open(a.test / f"dino_{short}_results.json", "w") as f:
        json.dump({v: dict(err1=[x["err1"] for x in res[v]], err5=[x["err5"] for x in res[v]]) for v in VARIANTS}, f)

    # --- images : une reussite et un echec (variante choisie) ---
    v = a.variant or VARIANTS[0]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    e1 = np.array([x["err1"] for x in res[v]])
    ok = np.where(e1 <= 100)[0] if a.images else np.array([], int)
    ko = np.where(e1 > 500)[0] if a.images else np.array([], int)
    picks = ([("reussite", int(max(ok, key=lambda i: res[v][i]["sim"])))] if len(ok) else []) + ([("echec", int(ko[0]))] if len(ko) else [])
    a.vis.mkdir(parents=True, exist_ok=True)
    for lab, i in picks:
        r = truth[i]
        rgb = cv2.cvtColor(cv2.imread(str(a.test / "queries" / r["file"])), cv2.COLOR_BGR2RGB)
        tx, ty = float(r["x_m"]) / gsd, float(r["y_m"]) / gsd
        c = centers[res[v][i]["top"]] / gsd
        half = int(560 / gsd)
        x0, y0 = int(max(0, tx - half)), int(max(0, ty - half))
        fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
        ax[0].imshow(rgb); ax[0].set_title(f"photo {r['file']} (cap {float(r['cap_deg']):.0f} deg)"); ax[0].axis("off")
        ax[1].imshow(img[y0:y0 + 2 * half, x0:x0 + 2 * half]); ax[1].plot(tx - x0, ty - y0, "r+", ms=18, mew=2, label="verite")
        ax[1].plot(c[0] - x0, c[1] - y0, "x", color="lime", ms=14, mew=3, label=f"reponse ({v}), erreur {res[v][i]['err1']:.0f} m"); ax[1].legend(); ax[1].axis("off")
        ax[1].set_title(lab)
        fig.tight_layout(); fig.savefig(a.vis / f"{a.test.name}_dino_{short}_{lab}.png", dpi=55); plt.close(fig)
    print("images :", [f"{a.test.name}_dino_{short}_{lab}.png" for lab, _ in picks])


if __name__ == "__main__":
    main()
