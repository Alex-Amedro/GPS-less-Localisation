"""Videos d'un vol : ce que voit le drone, ou il est vraiment, ou le systeme pense qu'il est.

DEUX vues separees, a regarder cote a cote (chacune avec de gros textes) :
  camera : la photo du drone, la carte sous l'estimation et sous la reponse d'une photo seule, « je reponds / je ne sais pas », les erreurs ;
  carte  : la carte du vol entier (trajectoires) et un zoom de 1,6 km avec le cadre de la camera et les hypotheses.
Un episode continu (--start, --count photos consecutives), mouvements interpoles entre deux photos (--sub), fondu enchaine des photos.
Un GIF plus court (--gif-start, --gif-count, --gif-sub) peut etre produit pour le README.
Second GIF (--analysis-gif) : le graphique d'erreur sur TOUT le vol. Textes en anglais (README).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import particle_filter as PF  # noqa: E402

COL = dict(truth=(0, 255, 0), hyb=(255, 0, 255), one=(0, 255, 255), dr=(0, 165, 255), part=(255, 255, 0), white=(255, 255, 255), grey=(170, 170, 170))
BG = 18


def crop_pad(img, cx, cy, half):
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    out = np.zeros((2 * half, 2 * half, 3), np.uint8)
    xs, ys = max(x0, 0), max(y0, 0)
    xe, ye = min(x0 + 2 * half, img.shape[1]), min(y0 + 2 * half, img.shape[0])
    if xe > xs and ye > ys:
        out[ys - y0:ye - y0, xs - x0:xe - x0] = img[ys:ye, xs:xe]
    return out


def disk_view(img, c_px, cap_deg, size, d_px=381):
    """Disque de 305 m de la carte autour de c_px, tourne dans le repere de la photo (cap vers le haut)."""
    cr = crop_pad(img, c_px[0], c_px[1], d_px)
    rot = cv2.warpAffine(cr, cv2.getRotationMatrix2D((d_px, d_px), cap_deg, 1.0), (2 * d_px, 2 * d_px), flags=cv2.INTER_LINEAR)
    out = cv2.resize(rot[d_px - d_px // 2:d_px + d_px // 2 + 1, d_px - d_px // 2:d_px + d_px // 2 + 1], (size, size), interpolation=cv2.INTER_AREA)
    yy, xx = np.mgrid[:size, :size]
    out[((xx - size / 2) ** 2 + (yy - size / 2) ** 2) > (size / 2) ** 2] = 25
    return out


def txt(fr, s, xy, scale=0.6, col=COL["white"], th=1):
    for dx, dy in ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)):  # contour : meme epaisseur que le texte
        cv2.putText(fr, s, (xy[0] + dx, xy[1] + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th, cv2.LINE_AA)
    cv2.putText(fr, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)


def lerp(a, b, t):
    return a + (b - a) * t


def alerp(a, b, t):
    return a + ((b - a + 180.0) % 360.0 - 180.0) * t


def save_gif(frames_rgb, path, ms, colors=96, hold=10):
    """GIF avec une palette commune (images voisines identiques -> fichier plus leger, pas de scintillement des couleurs)."""
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    pick = frames_rgb[:: max(1, len(frames_rgb) // 8)][:8]
    fixed = [c[::-1] for c in COL.values()] + [(0, 0, 0), (BG, BG, BG), (150, 70, 70), (60, 120, 60), (80, 80, 80), (70, 70, 70), (45, 45, 45), (70, 120, 70)]
    base = Image.fromarray(np.concatenate(pick, axis=0)).quantize(colors=colors - len(fixed), method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    flat = base.getpalette()[:3 * (colors - len(fixed))] + [v for c in fixed for v in c]   # couleurs de la legende imposees (le cyan ne doit pas devenir vert)
    pal = Image.new("P", (1, 1))
    pal.putpalette(flat + [0] * (768 - len(flat)))
    idx = [np.array(Image.fromarray(f).quantize(palette=pal, dither=Image.Dither.NONE)) for f in frames_rgb]
    idx += [idx[-1]] * hold
    # seuls les pixels qui changent sont codes : les autres sont « transparents » (index 255, libre) et gardent l'image precedente
    TR = 255
    ims = []
    for i, a_ in enumerate(idx):
        b_ = a_.copy()
        if i > 0:
            b_[a_ == idx[i - 1]] = TR
        im = Image.fromarray(b_, mode="P")
        im.putpalette(flat + [0] * (768 - len(flat)))
        ims.append(im)
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=ms, loop=0, optimize=False, transparency=TR, disposal=1)
    print(f"gif : {path} ({len(ims)} images, {path.stat().st_size / 1e6:.1f} Mo, {len(ims) * ms / 1000:.0f} s)")


def analysis_gif(e, unsure, ts_min, out, n_frames=70, ms=90):
    """Graphique d'erreur sur tout le vol, qui se dessine doucement."""
    T = len(e["hyb"])
    cw, ch, x0, y0, w, h = 900, 540, 70, 80, 800, 330
    lo, hi = np.log10(10.0), np.log10(10000.0)
    yf = lambda v: int(y0 + h - (np.clip(np.log10(max(v, 1.0)), lo, hi) - lo) / (hi - lo) * h)
    xf = lambda k: int(x0 + k / max(T - 1, 1) * w)
    frs = []
    for f in range(n_frames):
        k = max(2, int(T * (f + 1) / n_frames) - 1)
        cvs = np.full((ch, cw, 3), BG, np.uint8)
        cv2.rectangle(cvs, (x0, y0), (x0 + w, y0 + h), (70, 70, 70), 1)
        for v in (10, 100, 1000, 10000):
            cv2.line(cvs, (x0, yf(v)), (x0 + w, yf(v)), (45, 45, 45), 1)
            txt(cvs, "%d m" % v, (6, yf(v) + 5), 0.55)
        for kk in range(k + 1):
            if unsure[kk]:
                cv2.line(cvs, (xf(kk), y0), (xf(kk), y0 + h), (70, 70, 120), 1)
        for name, th in (("dr", 2), ("one", 2), ("hyb", 3)):
            sm = np.convolve(np.pad(e[name], (2, 2), mode="edge"), np.ones(5) / 5, mode="valid")  # lissage leger : lecture reposante
            pts = np.array([(xf(kk), yf(sm[kk])) for kk in range(k + 1)], np.int32).reshape(-1, 1, 2)
            cv2.polylines(cvs, [pts], False, COL[name], th, cv2.LINE_AA)
        txt(cvs, "Error over the whole flight (log scale)", (x0, 32), 0.75, COL["white"], 2)
        txt(cvs, "5-photo average; shaded = \"I don't know\"; %d min after the GPS cut" % int(ts_min[k]), (x0, 58), 0.55)
        for j, (name, lab) in enumerate((("hyb", "system"), ("one", "one photo"), ("dr", "blind"))):
            txt(cvs, "%s: %d%% within 100 m, median %d m" % (lab, 100 * np.mean(e[name][:k + 1] <= 100), np.median(e[name][:k + 1])), (x0, y0 + h + 34 + 30 * j), 0.65, COL[name], 1)
        frs.append(cv2.cvtColor(cvs, cv2.COLOR_BGR2RGB))
    save_gif(frs, out, ms, colors=48, hold=12)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", type=Path, default=Path("data/work/uav_val/visloc_04_full"))
    ap.add_argument("--ckpt", type=Path, default=Path("data/work/ckpt_b14_last2_mix_run4.pt"))
    ap.add_argument("--cut", type=int, default=50, help="photo ou le GPS est coupe")
    ap.add_argument("--start", type=int, default=340, help="premiere photo de l'episode (videos)")
    ap.add_argument("--count", type=int, default=100, help="nombre de photos consecutives (videos)")
    ap.add_argument("--sub", type=int, default=10, help="images de video entre deux photos (interpolation)")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--gif-start", type=int, default=340)
    ap.add_argument("--gif-count", type=int, default=36)
    ap.add_argument("--gif-sub", type=int, default=4)
    ap.add_argument("--gif-ms", type=int, default=120)
    ap.add_argument("--gif-colors", type=int, default=64)
    ap.add_argument("--thr", type=float, default=0.97)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", type=Path, default=Path("data/vis/comparatif_vol04"))
    ap.add_argument("--gif-prefix", type=Path, default=None, help="ex. docs/comparatif_vol04 -> _camera.gif et _carte.gif")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--analysis-gif", type=Path, default=None)
    a = ap.parse_args()

    gsd = float(json.load(open(a.test / "meta.json"))["gsd"])
    sim = PF.load_sims(a.test, "dinov2_vitb14", a.ckpt)
    grid = PF.Grid(sim["centers"])
    sift = np.array([[float(r["pred_x"]), float(r["pred_y"])] for r in csv.DictReader(open(a.test / "results.csv"))])
    rh = PF.run_scenario(sim, grid, a.cut, 100000, dict(PF.DEFAULTS), np.random.default_rng(a.seed), None, record=60, sift=sift)
    fr = list(rh["frames"])
    T = len(fr)
    truth, est_h, top1, dr = rh["truth"], rh["est"], rh["top1"], rh["dr"]
    e = dict(hyb=rh["e_est"], one=rh["e_top1"], dr=rh["e_dr"])
    unsure = rh["feat"][:, 1] < a.thr
    cap, ts = sim["cap"], sim["ts"]
    capk = cap[fr]
    print(f"vol : {T} photos apres la coupure ; systeme {np.median(e['hyb']):.0f} m, {100 * np.mean(e['hyb'] <= 100):.1f} % <= 100 m ; une photo {np.median(e['one']):.0f} m, "
          f"{100 * np.mean(e['one'] <= 100):.1f} % ; a l'aveugle {np.median(e['dr']):.0f} m ; repond a {100 * np.mean(~unsure):.1f} % ({100 * np.mean(e['hyb'][~unsure] <= 100):.1f} % <= 100 m)")
    if a.analysis_gif is not None:
        analysis_gif(e, unsure, (ts[fr] - ts[a.cut]) / 60.0, a.analysis_gif)
    if a.no_video and a.gif_prefix is None:
        return

    img = np.load(a.test / "map.npy", mmap_mode="r")
    Hpx, Wpx = img.shape[:2]
    rows = list(csv.DictReader(open(a.test / "truth.csv", encoding="utf-8")))
    qdir = a.test / "queries"

    # ---- tailles des deux vues ----
    AW = 540
    q0 = cv2.imread(str(qdir / rows[fr[0]]["file"]))
    PH = int(round(q0.shape[0] * AW / q0.shape[1]))
    DS = 262                       # diametre des disques
    AH = 44 + PH + 8 + 72 + 24 + DS + 6
    GH = 730                       # hauteur de la carte du vol entier
    gscale = GH / Hpx
    gw = int(round(Wpx * gscale))
    MZ = 380
    zs = MZ / 1600.0
    half_z = int(800 / gsd)
    BW = gw + 14 + MZ
    BH = AH
    gmap = (cv2.resize(np.ascontiguousarray(img[::4, ::4]), (gw, GH), interpolation=cv2.INTER_AREA) * 0.8).astype(np.uint8)
    G = lambda p: (int(p[0] / gsd * gscale), int(p[1] / gsd * gscale))
    gbase = gmap.copy()
    cv2.polylines(gbase, [np.array([G(p) for p in truth], np.int32).reshape(-1, 1, 2)], False, (70, 120, 70), 1, cv2.LINE_AA)
    print(f"vues : camera {AW}x{AH}, carte {BW}x{BH}")

    def photo(i):
        q = cv2.imread(str(qdir / rows[i]["file"]))
        return cv2.resize(q, (AW, PH)).astype(np.float32)

    def episode(k0, k1, sub):
        """Genere les paires (image camera, image carte) pour les photos fr[k0] .. fr[k1], avec interpolation."""
        fade = max(1, sub // 3)
        for k in range(k0, k1):
            i, i2 = int(fr[k]), int(fr[k + 1])
            p0, p1 = photo(i), photo(i2)
            dk0 = {n: disk_view(img, (est_h[k] if n == "est" else top1[k]) / gsd, capk[k], DS).astype(np.float32) for n in ("est", "one")}
            dk1 = {n: disk_view(img, (est_h[k + 1] if n == "est" else top1[k + 1]) / gsd, capk[k + 1], DS).astype(np.float32) for n in ("est", "one")}
            for j in range(sub):
                t = j / sub
                tr, es, d_ = lerp(truth[k], truth[k + 1], t), lerp(est_h[k], est_h[k + 1], t), lerp(dr[k], dr[k + 1], t)
                cp = alerp(capk[k], capk[k + 1], t)
                al = 0.0 if j < sub - fade else (j - (sub - fade) + 1) / (fade + 1)  # fondu enchaine sur la fin de l'intervalle
                kk = k if t < 0.5 else k + 1
                uns = unsure[kk]
                # traces COMPLETES depuis la coupure du GPS (une valeur par photo, puis le point interpole)
                trail_t = list(truth[:k + 1]) + [tr]
                trail_h = list(est_h[:k + 1]) + [es]
                trail_d = list(dr[:k + 1]) + [d_]
                # ================= vue CAMERA =================
                A = np.full((AH, AW, 3), BG, np.uint8)
                txt(A, "WHAT THE DRONE SEES", (10, 30), 0.85, COL["white"], 2)
                A[44:44 + PH] = (p0 * (1 - al) + p1 * al).astype(np.uint8)
                tmin = lerp(ts[i], ts[i2], t) - ts[a.cut]
                txt(A, "%d min %02d s after the GPS cut" % (int(tmin // 60), int(tmin % 60)), (10, 44 + 30), 0.7, COL["white"], 2)
                ys = 44 + PH + 8
                if uns:
                    cv2.rectangle(A, (0, ys), (256, ys + 62), (70, 70, 150), -1)
                    txt(A, "I DON'T KNOW", (10, ys + 41), 0.85, COL["white"], 2)
                else:
                    cv2.rectangle(A, (0, ys), (256, ys + 62), (60, 120, 60), -1)
                    txt(A, "I ANSWER", (48, ys + 41), 0.85, COL["white"], 2)
                for jj, (n2, lab) in enumerate((("hyb", "system"), ("one", "one photo"), ("dr", "blind"))):
                    txt(A, lab, (272, ys + 20 + 22 * jj), 0.65, COL[n2], 1)
                    txt(A, "%d m" % e[n2][kk], (420, ys + 20 + 22 * jj), 0.65, COL[n2], 1)
                yd = ys + 72 + 24
                for n_, xo, col_, lab in (("est", 0, COL["hyb"], "Map under the estimate"), ("one", DS + 16, COL["one"], "Map under 1-photo answer")):
                    A[yd:yd + DS, xo:xo + DS] = (dk0[n_] * (1 - al) + dk1[n_] * al).astype(np.uint8)
                    txt(A, lab, (xo + 2, yd - 8), 0.55, col_ if not (n_ == "est" and uns) else COL["grey"], 1)
                # ================= vue CARTE =================
                B = np.full((BH, BW, 3), BG, np.uint8)
                txt(B, "WHERE IT IS", (10, 30), 0.85, COL["white"], 2)
                g = gbase.copy()
                for arr, colr, wd in ((trail_d, COL["dr"], 2), (trail_t, COL["truth"], 3), (trail_h, COL["hyb"], 3)):
                    pts = np.array([G(p) for p in arr], np.int32).reshape(-1, 1, 2)
                    if len(pts) > 1:
                        cv2.polylines(g, [pts], False, colr, wd, cv2.LINE_AA)
                if e["one"][kk] > 150:
                    cv2.line(g, G(tr), G(top1[kk]), COL["one"], 1, cv2.LINE_AA)
                cv2.circle(g, G(top1[kk]), 7, COL["one"], 3, cv2.LINE_AA)
                cv2.circle(g, G(d_), 7, COL["dr"], 3, cv2.LINE_AA)
                cv2.circle(g, G(tr), 9, COL["truth"], 3, cv2.LINE_AA)
                cv2.circle(g, G(es), 6, COL["grey"] if uns else COL["hyb"], -1, cv2.LINE_AA)
                B[44:44 + GH, 0:gw] = g
                txt(B, "Whole flight", (6, 44 + 24), 0.6, COL["white"], 1)
                # zoom 1,6 km
                c_px = tr / gsd
                zm = (cv2.resize(crop_pad(img, c_px[0], c_px[1], half_z), (MZ, MZ), interpolation=cv2.INTER_AREA) * 0.85).astype(np.uint8)
                Z = lambda p: (int(MZ / 2 + (p[0] - tr[0]) * zs), int(MZ / 2 + (p[1] - tr[1]) * zs))
                ang = np.radians(cp)
                R = np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]])
                corners = np.array([[-229.0, -152.0], [229.0, -152.0], [229.0, 152.0], [-229.0, 152.0]]) @ R.T
                cv2.polylines(zm, [np.array([Z(tr + c) for c in corners], np.int32)], True, COL["white"], 2, cv2.LINE_AA)
                for q2 in rh["parts"][kk][:60]:
                    pz = Z(q2)
                    if 0 <= pz[0] < MZ and 0 <= pz[1] < MZ:
                        cv2.circle(zm, pz, 2, COL["part"], -1)
                for arr, colr, wd in ((trail_h[-30:], COL["hyb"], 4), (trail_t[-30:], COL["truth"], 2)):   # la fenetre ne montre que le bout recent
                    cv2.polylines(zm, [np.array([Z(p) for p in arr], np.int32).reshape(-1, 1, 2)], False, colr, wd, cv2.LINE_AA)
                for p_, colr in ((top1[kk], COL["one"]), (d_, COL["dr"])):
                    pz = Z(p_)
                    if 0 <= pz[0] < MZ and 0 <= pz[1] < MZ:
                        cv2.circle(zm, pz, 7, colr, 3, cv2.LINE_AA)
                cv2.circle(zm, Z(tr), 9, COL["truth"], 3, cv2.LINE_AA)
                cv2.circle(zm, Z(es), 6, COL["grey"] if uns else COL["hyb"], -1, cv2.LINE_AA)
                cv2.rectangle(zm, (0, 0), (MZ - 1, MZ - 1), (80, 80, 80), 1)
                zx = gw + 14
                B[44:44 + MZ, zx:zx + MZ] = zm
                txt(B, "Zoom 1.6 km (box = camera)", (zx + 6, 44 + 24), 0.55, COL["white"], 1)
                ly = 44 + MZ + 38
                for jj, (colr, lab) in enumerate(((COL["truth"], "green: true position"), (COL["hyb"], "magenta: system's estimate"),
                                                  (COL["part"], "cyan: its hypotheses"), (COL["one"], "yellow: answer of ONE photo"),
                                                  (COL["dr"], "orange: blind (heading+speed)"), (COL["grey"], "grey: \"I don't know\""))):
                    cv2.circle(B, (zx + 10, ly + 34 * jj - 6), 8, colr, -1, cv2.LINE_AA)
                    txt(B, lab, (zx + 30, ly + 34 * jj), 0.55, COL["white"], 1)
                yield A, B

    # ---- videos ----
    if not a.no_video:
        k0 = fr.index(a.start)
        k1 = min(k0 + a.count, T - 1)
        a.out_prefix.parent.mkdir(parents=True, exist_ok=True)
        vwa = cv2.VideoWriter(str(a.out_prefix) + "_camera.mp4", cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (AW, AH))
        vwb = cv2.VideoWriter(str(a.out_prefix) + "_carte.mp4", cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (BW, BH))
        n = 0
        for A, B in episode(k0, k1, a.sub):
            vwa.write(A); vwb.write(B); n += 1
        vwa.release(); vwb.release()
        print(f"videos : {a.out_prefix}_camera.mp4 / _carte.mp4 ({k1 - k0} photos consecutives, {n} images a {a.fps:.0f} i/s = {n / a.fps:.0f} s)")
    # ---- GIF ----
    if a.gif_prefix is not None:
        k0 = fr.index(a.gif_start)
        k1 = min(k0 + a.gif_count, T - 1)
        fa, fb = [], []
        for A, B in episode(k0, k1, a.gif_sub):
            fa.append(cv2.cvtColor(A, cv2.COLOR_BGR2RGB)); fb.append(cv2.cvtColor(B, cv2.COLOR_BGR2RGB))
        save_gif(fa, Path(str(a.gif_prefix) + "_camera.gif"), a.gif_ms, a.gif_colors)
        save_gif(fb, Path(str(a.gif_prefix) + "_carte.gif"), a.gif_ms, a.gif_colors)


if __name__ == "__main__":
    main()
