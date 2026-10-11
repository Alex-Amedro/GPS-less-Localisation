"""Etape 3 : affine les DERNIERES couches de DINOv2 (le reste est gele) par apprentissage contrastif (InfoNCE).

Paire = (disque de 305 m d'une edition RECENTE, disque du MEME endroit dans une edition plus ANCIENNE decale au hasard jusqu'a 64 m) ;
negatifs = les autres disques du lot + un disque voisin (320-800 m) de la meme carte ancienne (negatif difficile).
Augmentations de la photo : rotation +/-10 deg (erreur de cap), echelle +/-5 %, luminosite, contraste, couleur, flou, niveaux de gris, occultation.
Separation par zone : seules les dalles de split.json (hors test, validation et voisines) sont utilisees ; la validation sert a choisir le point d'arret.
La verite terrain de test n'est jamais vue.

Sortie : --out (checkpoint = seuls les parametres entraines, charges ensuite par dino_eval.py --ckpt).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint  # noqa: F401

sys.path.insert(0, str(Path(__file__).parent))
import dino_eval as DE  # noqa: E402

GSD = 0.8
D = int(round(DE.DISK_M / GSD))  # 381 px
TAU = 0.05


class PairStream:
    def __init__(self, root, seed):
        meta = json.load(open(root / "split.json"))
        self.root, self.seed = root, seed
        self.keys = [tuple(k) for k in meta["train"]]
        self.eds = sorted(meta["editions"], reverse=True)  # du plus recent au plus ancien
        self.pairs = [(q, m) for i, q in enumerate(self.eds) for m in self.eds[i + 1:]]
        self.mm = {}

    def arr(self, ed, k):
        if (ed, k) not in self.mm:
            self.mm[(ed, k)] = np.load(self.root / ed / f"{k[0]}_{k[1]}.npy", mmap_mode="r")
        return self.mm[(ed, k)]

    @staticmethod
    def crop(a, cx, cy, angle=0.0, scale=1.0):
        R = int(D * 0.8)
        x0, y0 = int(round(cx)) - R, int(round(cy)) - R
        sub = np.ascontiguousarray(a[y0:y0 + 2 * R, x0:x0 + 2 * R])
        M = cv2.getRotationMatrix2D((R, R), angle, scale)
        M[:, 2] += D / 2 - R
        return cv2.warpAffine(sub, M, (D, D), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    @staticmethod
    def photometric(img, rng, strong):
        x = img.astype(np.float32)
        s = 1.0 if strong else 0.5
        x = x * rng.uniform(1 - 0.3 * s, 1 + 0.3 * s) + rng.uniform(-25 * s, 25 * s)
        x = x * rng.uniform(1 - 0.1 * s, 1 + 0.1 * s, 3)
        x = 255 * (np.clip(x, 0, 255) / 255) ** rng.uniform(1 - 0.2 * s, 1 + 0.25 * s)
        if strong and rng.random() < 0.5:
            sg = rng.uniform(0.3, 1.5)
            x = cv2.GaussianBlur(x, (0, 0), sg)
        if strong and rng.random() < 0.1:
            x = np.repeat(x.mean(2, keepdims=True), 3, 2)
        if strong and rng.random() < 0.3:
            h, w = x.shape[:2]
            ch, cw = int(h * rng.uniform(0.1, 0.25)), int(w * rng.uniform(0.1, 0.25))
            y0, x0 = rng.integers(0, h - ch), rng.integers(0, w - cw)
            x[y0:y0 + ch, x0:x0 + cw] = x.mean((0, 1))
        return np.clip(x, 0, 255).astype(np.uint8)

    def gen(self, rng):
        mask = DE.disk_mask(D)
        margin = 1400  # 1120 m = 800 m (voisin) + 64 m (decalage) + rayon de fenetre : tout reste dans la dalle
        while True:
            k = self.keys[rng.integers(len(self.keys))]
            q_ed, m_ed = self.pairs[rng.integers(len(self.pairs))]
            qa, ma = self.arr(q_ed, k), self.arr(m_ed, k)
            H, W = qa.shape[:2]
            cx, cy = rng.uniform(margin, W - margin), rng.uniform(margin, H - margin)
            q = self.crop(qa, cx, cy, rng.uniform(-10, 10), rng.uniform(0.95, 1.05))
            ext = (q.astype(np.int16).sum(2) >= 762) | (q.astype(np.int16).sum(2) <= 3)
            if ext.mean() > 0.1:
                continue
            dx, dy = rng.uniform(-80, 80, 2)  # 64 m
            p = self.crop(ma, cx + dx, cy + dy)
            r, ph = rng.uniform(400, 1000), rng.uniform(0, 2 * math.pi)  # 320-800 m
            hn = self.crop(ma, cx + r * math.cos(ph), cy + r * math.sin(ph))
            yield (DE.to_tensor(self.photometric(q, rng, True), mask), DE.to_tensor(self.photometric(p, rng, False), mask),
                   DE.to_tensor(self.photometric(hn, rng, False), mask))


def northup_disk_scaled(photo, cap_deg, d, s):
    """Comme dino_eval.northup_disk, avec un facteur d'echelle s (s > 1 : on regarde une zone plus large) ; hors photo = gris neutre."""
    h, w = photo.shape[:2]
    c, sn = np.cos(np.radians(cap_deg)), np.sin(np.radians(cap_deg))
    R = np.array([[c, sn], [-sn, c]]) * s
    t = np.array([(w - 1) / 2, (h - 1) / 2]) - R @ np.array([(d - 1) / 2, (d - 1) / 2])
    return cv2.warpAffine(photo, np.hstack([R, t[:, None]]), (d, d), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=(124, 116, 104))


class UavStream:
    """Paires REELLES : (disque nord en haut d'une vraie photo de drone, disque de la carte satellite du vol a la position GPS decalee de <= 64 m).
    Lit les dossiers prepares par prep_uav_test.py (map.npy, queries/, truth.csv) ; photos et carte a 0,8 m/px ; le cap est la colonne Phi1."""

    def __init__(self, dirs):
        import csv
        self.vols, self.margin = [], int(D * 0.8) + 85
        for d in dirs:
            meta = json.load(open(d / "meta.json"))
            g = float(meta["gsd"])
            H, W = meta["map_shape"][:2]
            truth = list(csv.DictReader(open(d / "truth.csv")))
            ok = [i for i, r in enumerate(truth) if self.margin <= float(r["x_m"]) / g <= W - self.margin and self.margin <= float(r["y_m"]) / g <= H - self.margin]
            self.vols.append(dict(dir=d, truth=truth, ok=ok, gsd=g, HW=(H, W), map=None))
        n = np.array([len(v["ok"]) for v in self.vols], float)
        self.w = n / n.sum()
        print("vols reels d'entrainement :", {v["dir"].name: len(v["ok"]) for v in self.vols}, flush=True)

    def gen(self, rng):
        mask = DE.disk_mask(D)
        while True:
            v = self.vols[rng.choice(len(self.vols), p=self.w)]
            if v["map"] is None:
                v["map"] = np.load(v["dir"] / "map.npy", mmap_mode="r")
            r = v["truth"][v["ok"][rng.integers(len(v["ok"]))]]
            photo = cv2.cvtColor(cv2.imread(str(v["dir"] / "queries" / r["file"])), cv2.COLOR_BGR2RGB)
            q = northup_disk_scaled(photo, float(r["cap_deg"]) + rng.uniform(-10, 10), D, rng.uniform(0.9, 1.1))
            g, (H, W) = v["gsd"], v["HW"]
            cx, cy = float(r["x_m"]) / g, float(r["y_m"]) / g
            dx, dy = rng.uniform(-80, 80, 2)
            p = PairStream.crop(v["map"], cx + dx, cy + dy)
            hn = None
            for _ in range(20):
                rr, ph = rng.uniform(400, 1000), rng.uniform(0, 2 * math.pi)
                nx, ny = cx + rr * math.cos(ph), cy + rr * math.sin(ph)
                if self.margin - 80 <= nx <= W - self.margin + 80 and self.margin - 80 <= ny <= H - self.margin + 80:
                    hn = PairStream.crop(v["map"], nx, ny)
                    break
            if hn is None:
                continue
            yield (DE.to_tensor(PairStream.photometric(q, rng, True), mask), DE.to_tensor(PairStream.photometric(p, rng, False), mask),
                   DE.to_tensor(PairStream.photometric(hn, rng, False), mask))


class MixStream(torch.utils.data.IterableDataset):
    def __init__(self, ign, uav, uav_frac, seed):
        self.ign, self.uav, self.frac, self.seed = ign, uav, uav_frac, seed

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (wi.id if wi else 0))
        gi = self.ign.gen(rng) if self.ign else None
        gu = self.uav.gen(rng) if self.uav else None
        while True:
            use_u = gu is not None and (gi is None or rng.random() < self.frac)
            yield next(gu if use_u else gi)


def embed(model, x, n_last, grad, ckpt_act=False):
    """Empreinte `mean` (moyenne des jetons de patchs, normalisee). Les blocs avant les n_last derniers sont toujours sans gradient.
    ckpt_act : recalcule les activations des blocs entraines au lieu de les garder (moins de memoire GPU, ~30 % de temps en plus)."""
    split = len(model.blocks) - n_last
    with torch.no_grad():
        t = model.prepare_tokens_with_masks(x)
        for blk in model.blocks[:split]:
            t = blk(t)
    with torch.set_grad_enabled(grad):
        for blk in model.blocks[split:]:
            t = torch.utils.checkpoint.checkpoint(blk, t, use_reentrant=False) if (grad and ckpt_act) else blk(t)
        t = model.norm(t)
    return F.normalize(t[:, 1:].float().mean(1), dim=1)


class ValSet:
    """Disques de la carte et photos nord en haut d'un bloc de validation, prepares une fois (uint8 224 px, masque applique)."""

    def __init__(self, test_dir):
        gsd = float(json.load(open(test_dir / "meta.json"))["gsd"])
        img = np.load(test_dir / "map.npy")
        H, W = img.shape[:2]
        d, st = int(round(DE.DISK_M / gsd)), int(round(DE.STRIDE_M / gsd))
        mask = DE.disk_mask(d)
        self.name, self.centers, crops = test_dir.name, [], []
        for y in range(0, H - d + 1, st):
            for x in range(0, W - d + 1, st):
                c = img[y:y + d, x:x + d]
                if (((c.astype(np.int16).sum(2) >= 762) | (c.astype(np.int16).sum(2) <= 3))[mask]).mean() > 0.2:
                    continue
                crops.append(self.prep(c, mask)); self.centers.append(((x + d / 2) * gsd, (y + d / 2) * gsd))
        self.centers = np.array(self.centers)
        self.map = torch.from_numpy(np.stack(crops))
        truth = list(__import__("csv").DictReader(open(test_dir / "truth.csv")))
        qs = []
        for r in truth:
            rgb = cv2.cvtColor(cv2.imread(str(test_dir / "queries" / r["file"])), cv2.COLOR_BGR2RGB)
            qs.append(self.prep(DE.northup_disk(rgb, float(r["cap_deg"]), d), mask))
        self.q = torch.from_numpy(np.stack(qs))
        self.truth = np.array([[float(r["x_m"]), float(r["y_m"])] for r in truth])

    @staticmethod
    def prep(rgb, mask):
        x = rgb.astype(np.float32) / 255.0
        x[~mask] = DE.MEAN
        return (cv2.resize(x, (DE.INPUT, DE.INPUT), interpolation=cv2.INTER_AREA) * 255).round().astype(np.uint8)

    @torch.no_grad()
    def evaluate(self, model, dev):
        mean, std = torch.tensor(DE.MEAN, device=dev).view(1, 3, 1, 1), torch.tensor(DE.STD, device=dev).view(1, 3, 1, 1)

        def run(u8):
            out = []
            for i in range(0, len(u8), 128):
                x = u8[i:i + 128].to(dev).permute(0, 3, 1, 2).float() / 255
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out.append(embed(model, (x - mean) / std, 0, False))
            return torch.cat(out)
        dm, dq = run(self.map), run(self.q)
        sims = (dq @ dm.T).cpu().numpy()
        e1, e5 = [], []
        for i in range(len(sims)):
            top = DE.nms_top(np.argsort(-sims[i]), self.centers, 5)
            err = [float(np.hypot(*(self.centers[j] - self.truth[i]))) for j in top]
            e1.append(err[0]); e5.append(min(err))
        e1, e5 = np.array(e1), np.array(e5)
        return dict(r1_100=float((e1 <= 100).mean()), r1_500=float((e1 <= 500).mean()), r5_500=float((e5 <= 500).mean()), med=float(np.median(e1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="dinov2_vitb14")
    ap.add_argument("--data", type=Path, default=Path("data/work/train"))
    ap.add_argument("--val", type=Path, nargs="+", default=[Path("data/work/belfort_val_g08"), Path("data/work/belfort_val_rural")])
    ap.add_argument("--uav-dirs", type=Path, nargs="+", default=[], help="dossiers de vols UAV-VisLoc prepares (prep_uav_test.py), vols d'ENTRAINEMENT")
    ap.add_argument("--uav-frac", type=float, default=0.0, help="part des paires venant des vraies photos : 0 = IGN seul, 1 = vols seuls")
    ap.add_argument("--ckpt-act", action="store_true", help="recalculer les activations (necessaire pour le modele large sur 8 Go)")
    ap.add_argument("--n-last", type=int, default=2)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = "cuda"
    model = torch.hub.load("facebookresearch/dinov2", a.model, verbose=False).eval().to(dev)
    for p in model.parameters():
        p.requires_grad = False
    train_params = []
    for blk in model.blocks[-a.n_last:]:
        train_params += list(blk.parameters())
    train_params += list(model.norm.parameters())
    for p in train_params:
        p.requires_grad = True
    n_tr = sum(p.numel() for p in train_params)
    print(f"{a.model} : {sum(p.numel() for p in model.parameters()) / 1e6:.0f} M parametres dont {n_tr / 1e6:.1f} M entraines (derniers {a.n_last} blocs + norme)", flush=True)

    vals = [ValSet(v) for v in a.val]
    print("validation :", {v.name: (len(v.centers), len(v.q)) for v in vals}, flush=True)

    def val_all():
        model.eval()
        return {v.name: v.evaluate(model, dev) for v in vals}

    def score(r):
        return float(np.mean([(m["r1_100"] + m["r1_500"]) / 2 for m in r.values()]))

    try:  # courbes en direct : tensorboard --logdir data/work/tb
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(str(a.out.parent / "tb" / a.out.stem))
    except Exception:
        tb = None

    def log_val(r, step):
        if tb:
            for name, m in r.items():
                for key, val in m.items():
                    tb.add_scalar(f"validation_{name}/{key}", val, step)
            tb.add_scalar("validation/score", score(r), step)
            tb.flush()

    r0 = val_all()
    print("pas 0 (sans entrainement) :", {k: {a_: round(b, 3) for a_, b in v.items()} for k, v in r0.items()}, f"score {score(r0):.3f}", flush=True)
    log_val(r0, 0)
    best, best_step = score(r0), 0

    opt = torch.optim.AdamW(train_params, lr=a.lr, weight_decay=0.05)
    warm = 100
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    ign = PairStream(a.data, a.seed) if a.uav_frac < 1.0 else None
    uav = UavStream(a.uav_dirs) if a.uav_frac > 0.0 else None
    dl = torch.utils.data.DataLoader(MixStream(ign, uav, a.uav_frac, a.seed), batch_size=a.batch, num_workers=a.workers, persistent_workers=True, prefetch_factor=4)
    it = iter(dl)
    t0, run = time.time(), 0.0
    for step in range(1, a.steps + 1):
        q, p, h = next(it)
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            eq = embed(model, q.to(dev, non_blocking=True), a.n_last, True, a.ckpt_act)
            ep = embed(model, p.to(dev, non_blocking=True), a.n_last, True, a.ckpt_act)
            eh = embed(model, h.to(dev, non_blocking=True), a.n_last, True, a.ckpt_act)
        logits = eq @ torch.cat([ep, eh]).T / TAU
        loss = F.cross_entropy(logits, torch.arange(len(eq), device=dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step(); sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if tb and step % 10 == 0:
            tb.add_scalar("entrainement/perte", run, step)
            tb.add_scalar("entrainement/lr", sched.get_last_lr()[0], step)
        if step % 50 == 0:
            print(f"  pas {step}/{a.steps} perte {run:.3f} ({(time.time() - t0) / step:.2f} s/pas, GPU max {torch.cuda.max_memory_allocated() / 1e9:.1f} Go)", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            r = val_all()
            s = score(r)
            log_val(r, step)
            print(f"VALIDATION pas {step} :", {k: {a_: round(b, 3) for a_, b in v.items()} for k, v in r.items()}, f"score {s:.3f}", flush=True)
            if s > best:
                best, best_step = s, step
                sd = {k: v.detach().cpu() for k, v in model.state_dict().items() if any(k.startswith(f"blocks.{len(model.blocks) - a.n_last + i}.") for i in range(a.n_last)) or k.startswith("norm.")}
                a.out.parent.mkdir(parents=True, exist_ok=True)
                torch.save(sd, a.out)
    print(f"termine en {(time.time() - t0) / 60:.1f} min ; meilleur score validation {best:.3f} au pas {best_step} (0 = aucune amelioration) ; checkpoint : {a.out if best_step else 'aucun'}")


if __name__ == "__main__":
    main()
