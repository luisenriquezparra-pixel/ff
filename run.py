#!/usr/bin/env python3
"""Pipeline seg01: GVHMR + piso MoGe-2. Uso:
    !python3 ff/run.py --instalar   (solo modelos)
    VID = 'nombre'                   (celda aparte)
    !python3 ff/run.py $VID          (todo -> zip en Resultados/)
Receta: -s, --f-mm 24, --no-render, PySceneDetect -t 27, MoGe-2 vitb,
tobillos 7x7, std>6cm = descartado. Sin report OK no va a BVH.
SMPLX via API (Secrets); resto publico. Sin secretos ni pesos adentro.
"""
import csv
import datetime
import glob
import json
import os
import shutil
import subprocess
import sys

W = "/kaggle/working"
IN = f"{W}/inputs_demo"
OUT = f"{W}/outputs_demo"
BM = f"{W}/body_models"
MOGE = f"{W}/moge/moge-2-vitb-normal/model.pt"
FMM_DEFAULT = 24
T_DEFAULT = 27
FPS = 30


def sh(cmd, tail=3, check=False):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.stdout:
        for line in r.stdout.strip().split("\n")[-tail:]:
            print("  |", line)
    if check and r.returncode != 0:
        raise RuntimeError(f"fallo [{r.returncode}]: {cmd}\n{r.stderr[-800:]}")
    return r


def phase(msg):
    print(f"\n===== {msg} =====")


def need(path, what):
    ok = os.path.exists(path)
    print(f"  [{'OK' if ok else '--'}] {what}: {path}")
    return ok


# ------------------------------------------------------------------ FASE 0
def fase_modelos():
    phase("[1/5] MODELOS (todo auto)")
    os.makedirs(f"{BM}/smplx", exist_ok=True)
    os.makedirs(f"{BM}/smpl", exist_ok=True)
    os.makedirs(IN, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)

    kp = os.path.join(os.environ.get("HOME", "/root"), ".kaggle", "kaggle.json")
    if not os.path.exists(kp):
        try:
            from kaggle_secrets import UserSecretsClient
            u = UserSecretsClient()
            os.makedirs(os.path.dirname(kp), exist_ok=True)
            with open(kp, "w") as f:
                json.dump({"username": u.get_secret("KAGGLE_USERNAME"),
                           "key": u.get_secret("KAGGLE_KEY")}, f)
            os.chmod(kp, 0o600)
            print("  kaggle.json creado desde Secrets")
        except Exception as e:
            print(f"  *** SIN Secrets ({e}): SMPLX solo via Add Input manual ***")

    npz = f"{BM}/smplx/SMPLX_NEUTRAL.npz"
    pkl = f"{BM}/smpl/SMPL_NEUTRAL.pkl"
    if not (os.path.exists(npz) and os.path.exists(pkl)):
        r = sh("kaggle datasets download aimersito/smplx-neutral-npz "
               f"-p {BM}/smplx_raw --unzip 2>&1 | tail -2")
        raw_n = f"{BM}/smplx_raw/SMPLX_NEUTRAL.npz"
        raw_p = f"{BM}/smplx_raw/SMPLX_NEUTRAL.pkl"
        if os.path.exists(raw_n):
            shutil.copy(raw_n, npz)
        if os.path.exists(raw_p):
            shutil.copy(raw_p, pkl)
    if not (os.path.exists(npz) and os.path.exists(pkl)):
        raise SystemExit("*** FALTA SMPLX: adjunta tu dataset smplx-neutral-npz "
                         "(Add Input) y re-corre. Nada mas que hacer a mano. ***")

    sh("pip install \"gvhmr[preproc]\" \"scenedetect[opencv]\" huggingface_hub scipy 2>&1 | tail -1")
    if not os.path.exists(f"{W}/MoGe"):
        sh(f"git clone https://github.com/microsoft/MoGe.git {W}/MoGe 2>&1 | tail -1")
    os.environ["GVHMR_BODY_MODELS"] = BM
    sh(f"export GVHMR_BODY_MODELS={BM}; gvhmr download 2>&1 | tail -2")
    if not os.path.exists(MOGE):
        sh(f"cd {W} && python3 -c \"from huggingface_hub import snapshot_download; "
           "snapshot_download('Ruicheng/moge-2-vitb-normal', "
           "local_dir='./moge/moge-2-vitb-normal')\" 2>&1 | tail -1")
    sh(f"export GVHMR_BODY_MODELS={BM}; gvhmr info 2>&1 | grep -iE 'smplx|smpl |yolo|hmr2'")
    assert os.path.exists(MOGE), "*** MoGe no se descargo ***"
    print("  MODELOS-OK")


# --------------------------------------------------------------- VIDEO-IN
def video_in():
    phase("[2/5] VIDEO-IN (descubre .mp4, auto)")
    found = subprocess.run("find /kaggle/input -type f -iname '*.mp4' 2>/dev/null",
                           shell=True, capture_output=True, text=True).stdout.split()
    for s in found:
        shutil.copy(s, IN)
    vids = sorted(s[:-4] for s in os.listdir(IN) if s.endswith(".mp4"))
    if not vids:
        raise SystemExit("*** SIN VIDEO: adjunta tu dataset (Add Input) o sube el "
                         ".mp4 a inputs_demo/ y re-corre ***")
    print("  VIDEOS (escribe uno tal cual en VID?):")
    for v in vids:
        print(f"   - {v}")
    return vids


def pedir_vid(vids):
    raise SystemExit("*** Falta VID: corre como !python3 ff/run.py $VID ***")


# ------------------------------------------------------------------ SPLIT
def fase_split(vid, t):
    phase("[3/5] SPLIT (cortes, >=2s)")
    segdir = f"{W}/segments_{vid}"
    os.makedirs(segdir, exist_ok=True)
    sh(f"cd {W} && scenedetect -i inputs_demo/{vid}.mp4 detect-content -t {t} "
       f"list-scenes -o segments_{vid}/", check=False)
    csvp = f"{segdir}/{vid}-Scenes.csv"
    if not os.path.exists(csvp):
        raise SystemExit("*** SIN CSV: scenedetect no detecto nada "
                         "(video corrupto o -t muy alto) ***")
    kept = []

    def _col(row, *aliases):
        norm = {k.strip().lower(): k for k in row.keys()}
        for a in aliases:
            if a in norm:
                return row[norm[a]]
        return None

    with open(csvp) as f:
        rows = list(csv.DictReader(f))
    if rows:
        hdr = list(rows[0].keys())
        t0 = _col(rows[0], "start time (seconds)", "start time", "start_time",
                  "start (seconds)", "start")
        t1 = _col(rows[0], "end time (seconds)", "end time", "end_time",
                  "end (seconds)", "end")
        s0 = _col(rows[0], "start timecode", "start_timecode", "start_tc", "start tc")
        s1 = _col(rows[0], "end timecode", "end_timecode", "end_tc", "end tc")
        if t0 is None or t1 is None or s0 is None or s1 is None:
            raise SystemExit("*** CSV con columnas desconocidas. Vistas: "
                             f"{hdr}. Pega esto para ajustar alias ***")
    for r in rows:
        t0 = _col(r, "start time (seconds)", "start time", "start_time",
                  "start (seconds)", "start")
        t1 = _col(r, "end time (seconds)", "end time", "end_time",
                  "end (seconds)", "end")
        s0 = _col(r, "start timecode", "start_timecode", "start_tc", "start tc")
        s1 = _col(r, "end timecode", "end_timecode", "end_tc", "end tc")
        if float(t1) - float(t0) < 2.0:
            continue
        nm = f"seg{len(kept) + 1:02d}"
        sh(f"ffmpeg -y -v error -ss {s0} -to {s1} "
           f"-i {IN}/{vid}.mp4 -c copy {segdir}/{nm}.mp4", check=True)
        kept.append((nm, s0, s1))
    if not kept:
        raise SystemExit("*** 0 SEGMENTOS >=2s: baja el umbral o revisa el video ***")
    print("  SEGMENTOS:", [k[0] for k in kept])
    try:
        import cv2
        tiles = []
        for nm, _, _ in kept:
            cap = cv2.VideoCapture(f"{segdir}/{nm}.mp4")
            cap.set(1, int(cap.get(7)) // 2)
            ok, img = cap.read()
            cap.release()
            img = cv2.resize(img, (320, 568))
            cv2.imwrite(f"{segdir}/{nm}_mid.jpg", img)
            tiles.append(img)
        rows = [cv2.hconcat(tiles[i:i + 5]) for i in range(0, len(tiles), 5)]
        wmax = max(r.shape[1] for r in rows)
        rows = [cv2.copyMakeBorder(r, 0, 0, 0, wmax - r.shape[1],
                                   cv2.BORDER_CONSTANT) for r in rows]
        cv2.imwrite(f"{segdir}/contact_sheet.jpg", cv2.vconcat(rows))
        print("  SHEET-OK")
    except Exception as e:
        print(f"  (sheet opcional fallo: {e})")
    return segdir, [k[0] for k in kept]


# ------------------------------------------------------------------- LOTE
NAMES22 = ["Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
           "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck",
           "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
           "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist"]
_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        import smplx
        _MODEL = smplx.create(BM, model_type="smplx", gender="neutral", batch_size=1)
    return _MODEL


def escribir_bvh(d, bvh_path):
    import torch
    import numpy as np
    from scipy.spatial.transform import Rotation
    g = d["smpl_params_global"]
    n = g["body_pose"].shape[0]
    body_pose = g["body_pose"].reshape(n, 21, 3).float()
    global_orient = g["global_orient"].float()
    betas_mean = g["betas"].float().mean(dim=0, keepdim=True)
    transl_all = g["transl"].float().numpy()
    model = _model()
    with torch.no_grad():
        rest = model(betas=betas_mean).joints[0, :22].detach().numpy()
    parents = model.parents[:22].tolist()
    offsets = rest - rest[[0 if p < 0 else p for p in parents]]
    offsets[0] = np.zeros(3)
    pelvis = np.zeros((n, 3))
    with torch.no_grad():
        for st in range(0, n, 400):
            e = min(st + 400, n)
            B = e - st
            z3 = torch.zeros(B, 3)
            out = model(betas=betas_mean.expand(B, -1).contiguous(),
                        body_pose=body_pose[st:e], global_orient=global_orient[st:e],
                        jaw_pose=z3, leye_pose=z3, reye_pose=z3,
                        left_hand_pose=torch.zeros(B, 6),
                        right_hand_pose=torch.zeros(B, 6),
                        expression=torch.zeros(B, 10), transl=torch.zeros(B, 3))
            pelvis[st:e] = out.joints[:, 0].detach().numpy() + transl_all[st:e]
    aa = np.concatenate([global_orient.numpy()[:, None, :], body_pose.numpy()], axis=1)
    eul = Rotation.from_rotvec(aa.reshape(-1, 3)).as_euler("ZXY", degrees=True).reshape(n, 22, 3)
    children = [[] for _ in range(22)]
    for j, p in enumerate(parents):
        if p >= 0:
            children[p].append(j)
    order = []
    lines = ["HIERARCHY", "ROOT Pelvis", "{",
             "OFFSET 0.000000 0.000000 0.000000",
             "CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation"]

    def block(j, depth):
        order.append(j)
        pad = "\t" * depth
        lines.append(f"{pad}{'ROOT' if depth == 1 else 'JOINT'} {NAMES22[j]}")
        lines.append(f"{pad}" + "{")
        ox, oy, oz = offsets[j]
        lines.append(f"{pad}\tOFFSET {ox:.6f} {oy:.6f} {oz:.6f}")
        ch = "6 Xposition Yposition Zposition " if j == 0 else "3 "
        lines.append(f"{pad}\tCHANNELS {ch}Zrotation Xrotation Yrotation")
        if not children[j]:
            dv = offsets[j]
            nv = np.linalg.norm(dv)
            dv = dv / nv if nv > 1e-8 else np.array([0.0, -1.0, 0.0])
            ex, ey, ez = dv * 0.12
            lines.append(f"{pad}\tEnd Site")
            lines.append(f"{pad}\t" + "{")
            lines.append(f"{pad}\t\tOFFSET {ex:.6f} {ey:.6f} {ez:.6f}")
            lines.append(f"{pad}\t" + "}")
        else:
            for c in children[j]:
                block(c, depth + 1)
        lines.append(f"{pad}" + "}")

    for c in children[0]:
        block(c, 2)
    order.insert(0, 0)
    lines += ["}", "MOTION", f"Frames: {n}", f"Frame Time: {1.0 / FPS:.6f}"]
    for i in range(n):
        vals = [f"{pelvis[i, 0]:.6f}", f"{pelvis[i, 1]:.6f}", f"{pelvis[i, 2]:.6f}"]
        for j in order:
            z, x, y = eul[i, j]
            vals += [f"{z:.6f}", f"{x:.6f}", f"{y:.6f}"]
        lines.append(" ".join(vals))
    with open(bvh_path, "w") as f:
        f.write("\n".join(lines) + "\n")
def fase_lote(vid, fmm, segdir, segs):
    phase("[4/5] LOTE (receta + piso + delta + depths + report)")
    import torch
    import numpy as np
    if not torch.cuda.is_available():
        raise SystemExit("*** SIN GPU: activa acelerador (Settings -> Accelerator -> GPU) ***")
    import cv2
    from sklearn.linear_model import RANSACRegressor
    from moge.model.v2 import MoGeModel
    res = f"{W}/Resultados/{vid}"
    os.makedirs(res, exist_ok=True)
    os.environ["GVHMR_BODY_MODELS"] = BM
    moge = MoGeModel.from_pretrained(MOGE).cuda().eval()
    index = []
    for nm in segs:
        R = f"{res}/{nm}"
        os.makedirs(R, exist_ok=True)
        rep = [f"VID={vid} SEG={nm}"]
        try:
            if os.path.exists(f"{R}/{nm}_corrected.pt") and os.path.exists(f"{R}/report.txt"):
                print(f"  {nm}: YA HECHO (skip)")
                index.append((nm, "YA-HECHO"))
                continue
            seg = f"{segdir}/{nm}.mp4"
            probe = subprocess.run(
                f"ffprobe -v error -select_streams v:0 -show_entries stream=width,height "
                f"-of csv=p=0 {seg}", shell=True, capture_output=True, text=True).stdout.strip()
            ww, hh = [int(x) for x in probe.split(",")]
            vf = "scale=1280:720" if ww > hh else "scale=720:1280"
            norm = f"{segdir}/{nm}_30fps.mp4"
            sh(f"ffmpeg -y -v error -i {seg} -vf \"{vf}:flags=lanczos,setsar=1,fps=30\" "
               f"-r 30 -c:v libx264 -pix_fmt yuv420p -crf 18 -an {norm}", check=True)
            odir = f"{OUT}/{vid}_{nm}_30fps"
            sh(f"rm -rf {odir}")
            r = sh(f"gvhmr demo {norm} -o {OUT} -s --f-mm {fmm} --no-render")
            pt = f"{odir}/hmr4d_results.pt"
            if not os.path.exists(pt):
                raise RuntimeError("sin .pt")
            d = torch.load(pt, map_location="cpu")
            g = d["smpl_params_global"]
            n = g["body_pose"].shape[0]
            rep.append(f"frames={n}")
            key = f"{R}/{nm}_key.jpg"
            sh(f"ffmpeg -y -v error -ss {n/60:.2f} -i {norm} -frames:v 1 {key}", check=True)
            img = cv2.imread(key)
            tt = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                  ).permute(2, 0, 1).float().cuda() / 255
            out = moge.infer(tt[None])
            P = out["points"][0].cpu().numpy()
            N = out["normal"][0].cpu().numpy()
            M = out["mask"][0].cpu().numpy() > 0
            H, Wd, _ = P.shape
            ys, xs = np.mgrid[0:H, 0:Wd]
            cand = M & (ys > H * 0.45) & (np.abs(N[..., 1]) > 0.9)
            cand[:, Wd // 3:2 * Wd // 3] = False
            pts = P[cand].reshape(-1, 3)
            pts = pts[np.isfinite(pts).all(1)]
            reg = RANSACRegressor(min_samples=3, residual_threshold=0.05,
                                  max_trials=500).fit(pts[:, :2], pts[:, 2])
            a, b = reg.estimator_.coef_
            c = float(reg.estimator_.intercept_)
            nv = np.array([-a, -b, 1])
            nv /= np.linalg.norm(nv)
            d0 = -c * nv[2]
            inl = int(reg.inlier_mask_.sum())
            pct = 100 * inl / len(pts)
            rep.append(f"plano: pts={len(pts)} inl={pct:.0f}%")
            k2 = d["kp2d"]
            k2 = k2.numpy() if torch.is_tensor(k2) else k2
            mid = n // 2
            ank = []
            for j, lab in [(15, "L"), (16, "R")]:
                x, y, cf = [float(v) for v in k2[mid, j]]
                if cf < 0.3:
                    ank.append((lab, cf, None, None))
                    continue
                u0 = min(max(int(x), 0), Wd - 1)
                v0 = min(max(int(y), 0), H - 1)
                win = []
                for dv in range(-3, 4):
                    for du in range(-3, 4):
                        uu = min(max(u0 + du, 0), Wd - 1)
                        vv = min(max(v0 + dv, 0), H - 1)
                        win.append(float(nv @ P[vv, uu] + d0))
                win = np.array(win)
                ank.append((lab, cf, float(np.median(win)), float(win.std())))
            tp = []
            for lab, cf, med, std in ank:
                ms = "x" if med is None else str(round(med, 3))
                ss = "x" if std is None else str(round(std, 3))
                tp.append(lab + " cf=" + str(round(cf, 2)) + " med=" + ms + " std=" + ss)
            rep.append("tobillos: " + " ".join(tp))
            ok = [(lab, med, std) for lab, cf, med, std in ank
                  if med is not None and std is not None and std <= 0.06]
            flags = []
            if pct < 30:
                flags.append("WARN:inliers<30")
            if not ok:
                flags.append("WARN:ambos-std>6")
            if ok:
                lab, med, std = sorted(ok, key=lambda z: z[2])[0]
                delta = med - 0.10
                if abs(delta) > 0.10:
                    flags.append("WARN:delta>10cm")
                g["transl"][:, 1] -= float(delta)
                sgn = "+" if delta >= 0 else ""
                rep.append(f"delta={sgn}{round(delta, 4)}m (pie {lab}) CORREGIDO")
            else:
                rep.append("delta=0 (SIN corregir: velocidad anotada)")
            rep.append("FLAGS: " + (" ".join(flags) if flags else "OK"))
            torch.save(d, f"{R}/{nm}_corrected.pt")
            escribir_bvh(d, f"{R}/{nm}.bvh")
            rep.append(f"bvh={nm}.bvh")
            bb = d["bbx_xys"]
            bb = (bb.numpy() if torch.is_tensor(bb) else bb).astype(float)
            cap = cv2.VideoCapture(norm)
            deps = []
            step = max(1, n // 11)
            for f_ in range(0, n, step)[:11]:
                cap.set(1, f_)
                okf, im = cap.read()
                if not okf:
                    break
                cx, cy, s = bb[min(f_, n - 1)]
                side = s * 200 * 0.5
                x0 = int(max(cx - side / 2, 0))
                y0 = int(max(cy - side / 2, 0))
                x1 = int(min(cx + side / 2, im.shape[1]))
                y1 = int(min(cy + side / 2, im.shape[0]))
                crop = im[y0:y1, x0:x1]
                t2 = torch.from_numpy(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                                      ).permute(2, 0, 1).float().cuda() / 255
                dd = moge.infer(t2[None])["depth"].cpu().numpy()[0]
                h, w = dd.shape
                core = dd[h // 4:3 * h // 4, w // 4:3 * w // 4]
                deps.append(float(np.median(core)))
            cap.release()
            np.savetxt(f"{R}/{nm}_torso_depth.txt", np.array(deps), fmt="%.4f")
            rep.append(f"depths[{len(deps)}]")
            with open(f"{R}/report.txt", "w") as f:
                f.write("\n".join(rep) + "\n")
            index.append((nm, rep[-2]))
            print(f"  {nm}: OK frames={n} {rep[-2]}")
        except Exception as e:
            with open(f"{R}/report.txt", "w") as f:
                f.write("\n".join(rep) + f"\nFAIL: {type(e).__name__}: {str(e)[:300]}\n")
            index.append((nm, "FAIL"))
            print(f"  {nm}: FAIL {str(e)[:200]}")
    with open(f"{res}/index.csv", "w") as f:
        f.write("\n".join(a + ";" + b for a, b in index))
    print(f"  LOTE FIN: {len(index)} segmentos -> {res}")
    return res


# -------------------------------------------------------------------- ZIP
def fase_zip(vid):
    phase("[5/5] ZIP (SOLO el zip vive en Resultados/)")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    zf = f"{W}/Resultados/LOTE_{vid}_{ts}.zip"
    sh(f"cd {W} && rm -f Resultados/LOTE_{vid}_*.zip && zip -r {zf} "
       f"Resultados/{vid} segments_{vid}/contact_sheet.jpg "
       f"segments_{vid}/{vid}-Scenes.csv "
       "-x '*/mesh*' '*/camera*' '*preprocess*' '*0_input_video*' "
       "'*_geo.pt' '*_30fps.mp4' '*_mid.jpg' 2>&1 | tail -2")
    sh(f"ls -lh {W}/Resultados/")
    print(f"  ZIP-OK: {zf}  <-- baja este archivo con 1 clic")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    print("== PIPELINE GANADOR seg01 | GVHMR preciso + piso MoGe-2 ==")
    if argv[:1] == ["--instalar"]:
        fase_modelos()
        print("MODELOS-OK (instalacion completa, nada mas que hacer aqui)")
        return
    if not argv:
        raise SystemExit("*** Sin VID: corre como !python3 ff/run.py $VID "
                         "(celda aparte con VID = 'nombre') ***")
    if argv[0].startswith("--"):
        raise SystemExit("*** Flags validos: --instalar | $VID [FMM] [T]. "
                         "Ej: !python3 ff/run.py $VID ***")
    vid = argv[0]
    if vid.endswith(".mp4"):
        vid = vid[:-4]
    try:
        fmm = int(argv[1]) if len(argv) > 1 else FMM_DEFAULT
    except Exception:
        fmm = FMM_DEFAULT
    try:
        t = int(argv[2]) if len(argv) > 2 else T_DEFAULT
    except Exception:
        t = T_DEFAULT
    print(f"VID={vid} FMM={fmm} T={t} (argv; sin prompts)")
    fase_modelos()
    vids = video_in()
    if vid not in vids:
        raise SystemExit(f"*** VID '{vid}' no esta en inputs_demo/. Opciones: "
                         f"{', '.join(vids)} (revisa letra por letra) ***")
    segdir, segs = fase_split(vid, t)
    res = fase_lote(vid, fmm, segdir, segs)
    fase_zip(vid)
    print(f"\nFIN: revisa {res}/index.csv (FLAGS por segmento). "
          "Sin report OK no va a BVH.")


if __name__ == "__main__":
    main()
