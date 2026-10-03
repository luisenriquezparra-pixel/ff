#!/usr/bin/env python3
"""Pipeline ganador seg01 (GVHMR preciso + piso MoGe-2) — mega-script 1-archivo.

Uso en Kaggle (2 lineas a mano, nada mas):
    !git clone https://github.com/luisenriquezparra-pixel/ff.git
    !python3 ff/run.py
El script pregunta VID? -> escribes el nombre del video (sin .mp4, debe estar
en /kaggle/working/inputs_demo/ o adjuntado en /kaggle/input) y corre TODO:
modelos -> split -> lote por segmento -> zip en Resultados/ (SOLO el zip).

Receta locked: -s camara fija, --f-mm con guion (24 = 1x iPhone), --no-render
siempre (bug vertical), PySceneDetect -t 27, MoGe-2 vitb-normal (NO v3),
tobillos ventana 7x7 (NUNCA 1 pixel), regla std>6cm = tobillo descartado.

Publico-seguro: SIN secretos, SIN claves, SIN pesos adentro. SMPLX (gated MPI)
llega via Kaggle API desde TU dataset privado (Secrets KAGGLE_USERNAME/KEY);
si no hay secrets, pide adjuntar el dataset a mano. Todo lo demas es publico.
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

    # kaggle.json desde Secrets (o el que ya exista)
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

    # SMPLX via API desde dataset privado propio (legal: tu copia, tu cuenta)
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

    sh("pip install \"gvhmr[preproc]\" \"scenedetect[opencv]\" huggingface_hub 2>&1 | tail -1")
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
    # Reservado: VID siempre llega por argv (celda del usuario). Sin prompts.
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
    with open(csvp) as f:
        for r in csv.DictReader(f):
            if float(r["Length (seconds)"]) < 2.0:
                continue
            nm = f"seg{len(kept) + 1:02d}"
            sh(f"ffmpeg -y -v error -ss {r['Start Timecode']} -to {r['End Timecode']} "
               f"-i {IN}/{vid}.mp4 -c copy {segdir}/{nm}.mp4", check=True)
            kept.append((nm, r["Start Timecode"], r["End Timecode"]))
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
        # CELDA INSTALAR: solo modelos, termina limpio. Sin prompts.
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
    # Modelos: verifica rapido, instala solo lo que falte (pip tarda segundos
    # si ya esta; SMPLX/API igual con checkpoints de archivos).
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
