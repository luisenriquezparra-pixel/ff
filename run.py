#!/usr/bin/env python3
"""Pipeline multi-persona: GVHMR + piso MoGe-2 por ID. Uso:
    !python3 ff/run.py --instalar   (solo modelos)
    VID = 'nombre'                   (celda aparte)
    !python3 ff/run.py $VID          (todo -> SOLO BVH en /kaggle/working/)
Receta: -s, --f-mm 24, --no-render, PySceneDetect -t 27, MoGe-2 vitb,
tobillos 7x7, std>6cm = descartado. Salida: {VID}_{seg}_id{ID}_{n}f.bvh
y nada mas. 1 persona = 1 ID = 1 BVH. <60 frames = descartado con aviso.
EDOD: ningun path se asume. Todo se DESCRIBE (nombre + candidatos +
validador por CONTENIDO) y la ley resolver() lo encuentra donde este.
Personas = INDICES en arrays IDS[10], no objetos. Leyes: TRACK, TRACKLET,
POSE-ID, PISO-ID, EXPORT-ID. Sin clases, sin self: datos + leyes.
SMPLX via API (Secrets); resto publico. Sin secretos ni pesos adentro.
Codigo libre: si falta algo (yaml, ultralytics) se autocrea o avisa VACIO.
Nada revienta a mitad del lote: cada ID falla solo, los demas siguen."""
import csv
import datetime
import glob
import json
import os
import re
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
MAX_IDS = 10
MIN_FRAMES = 60


# ---------------------------------------- CAPA IDS[10] (EDOD SoA)
# Que existe: hasta MAX_IDS personas por segmento. Cada persona es un
# INDICE i en estos arrays paralelos preasignados. Sin objetos, sin clases.
# TRK = numero de track YOLO, NFR = frames visible, CNF = confianza media,
# ACT = sigue viva en el lote. ID_N = cuantas posiciones estan en uso
# (nunca se agrega, solo se reescriben posiciones: preasignacion total).
ID_TRK = [0] * MAX_IDS
ID_NFR = [0] * MAX_IDS
ID_CNF = [0.0] * MAX_IDS
ID_ACT = [False] * MAX_IDS
ID_N = 0
# Cajas por ID para recortar tracklets: CAJAS[i] = lista de
# (frame_idx, x0, y0, x1, y1) en pixeles del segmento normalizado.
ID_CAJAS = [[] for _ in range(MAX_IDS)]


def ley_ids_limpiar():
    """Ley previa: deja IDS[10] en cero. Toda corrida empieza aqui."""
    global ID_N
    for i in range(MAX_IDS):
        ID_TRK[i] = 0
        ID_NFR[i] = 0
        ID_CNF[i] = 0.0
        ID_ACT[i] = False
        ID_CAJAS[i] = []
    ID_N = 0


def ley_rankear(hallazgos):
    """Ley pura (sin GPU, sin archivos): recibe {trk: (frames, conf_media,
    cajas)} y llena IDS[10] con el top por frames. Devuelve cuantos quedan
    activos. Se prueba sin Kaggle, sin video, sin nada."""
    global ID_N
    ley_ids_limpiar()
    orden = sorted(hallazgos.keys(),
                   key=lambda t: hallazgos[t][0], reverse=True)[:MAX_IDS]
    for i, t in enumerate(orden):
        nfr, cnf, cajas = hallazgos[t]
        ID_TRK[i] = int(t)
        ID_NFR[i] = int(nfr)
        ID_CNF[i] = float(cnf)
        ID_ACT[i] = True
        ID_CAJAS[i] = list(cajas)
    ID_N = len(orden)
    return ID_N


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


def id_valido(n_frames):
    """Ley de 2 segundos: solo la racha con >=MIN_FRAMES genera BVH.
    Pregunta por la propiedad frames, nunca por identidad. Lo demas
    se descarta con aviso y no genera ningun archivo."""
    try:
        return int(n_frames) >= MIN_FRAMES
    except Exception:
        return False


def need(path, what):
    ok = os.path.exists(path)
    print(f"  [{'OK' if ok else '--'}] {what}: {path}")
    return ok


# --------------------------------------- DESCRIBIR LO QUE EXISTE (EDOD)
# No se asume DONDE estan las cosas. Se declara QUE existe:
#   (nombre + candidatos + validador por CONTENIDO)
# y UNA sola ley las resuelve todas. Un cambio futuro = una fila,
# nunca diez lineas acopladas. Sin clases, sin self: datos + leyes.
HALLADO = {}


def resolver(nombre, candidatos, valida, ayuda, clave=None, reversa=False):
    """Ley universal de descubrimiento: recorre candidatos (globs),
    valida por CONTENIDO (no por nombre) y devuelve el primero valido.
    Si nada existe, error con la lista exacta de donde busco."""
    vistos = []
    for patron in candidatos:
        vistos.extend(sorted(glob.glob(patron, recursive=True)))
    if clave is not None:
        try:
            vistos = sorted(vistos, key=clave, reverse=reversa)
        except Exception:
            pass
    for p in vistos:
        try:
            ok, detalle = valida(p)
        except Exception as e:
            ok, detalle = False, f"validador-fallo:{str(e)[:60]}"
        if ok:
            print(f"  [ENCONTRADO] {nombre}: {p} ({detalle})")
            HALLADO[nombre] = p
            return p
    donde = vistos if vistos else candidatos
    raise SystemExit(f"*** NO EXISTE {nombre}. Busque en: {donde}. {ayuda} ***")


def _mb(p):
    try:
        return f"{os.path.getsize(p) / 1e6:.0f}MB"
    except Exception:
        return "?"


def v_npz(p):
    ok = p.endswith(".npz") and os.path.isfile(p) \
        and os.path.getsize(p) > 1000000
    return (ok, _mb(p))


def v_moge(p):
    ok = p.endswith("model.pt") and os.path.isfile(p) \
        and os.path.getsize(p) > 10000000
    return (ok, _mb(p))


def v_mp4(p):
    if not (os.path.isfile(p) and os.path.getsize(p) > 10000):
        return (False, "vacio/ausente")
    pr = subprocess.run(
        f"ffprobe -v error -select_streams v:0 "
        f"-show_entries stream=width,height -of csv=p=0 \"{p}\"",
        shell=True, capture_output=True, text=True)
    nums = re.findall(r"\d+", pr.stdout or "")
    if len(nums) >= 2:
        return (True, f"{nums[0]}x{nums[1]}")
    return (False, f"ffprobe-vacio rc={pr.returncode}")


def v_pt_gvhmr(p, t0=None):
    if not (p.endswith(".pt") and os.path.isfile(p)):
        return (False, "no-pt")
    if t0 is not None:
        try:
            if os.path.getmtime(p) < t0 - 5:
                return (False, "viejo")
        except Exception:
            return (False, "sin-fecha")
    try:
        import torch
        d = torch.load(p, map_location="cpu")
        if isinstance(d, dict) and "smpl_params_global" in d:
            n = d["smpl_params_global"]["body_pose"].shape[0]
            return (True, f"frames={n}")
        return (False, "sin-smpl_params_global")
    except Exception as e:
        return (False, f"ilegible:{str(e)[:60]}")


def buscar_smplx():
    return resolver(
        "SMPLX",
        [f"{BM}/smplx/*.npz", f"{BM}/*.npz",
         "./**/SMPLX_NEUTRAL.npz",
         os.path.join(os.environ.get("HOME", "/root"),
                      "**/SMPLX_NEUTRAL.npz")],
        v_npz, "Re-corre --instalar (dataset smplx-neutral-npz).")


def buscar_moge():
    return resolver(
        "MoGe-vitb",
        [MOGE, f"{W}/moge/**/model.pt", "./moge/**/model.pt"],
        v_moge, "Re-corre --instalar.")


def buscar_video(vid):
    return resolver(
        f"VIDEO {vid}",
        [f"{IN}/{vid}.mp4", f"/kaggle/input/**/{vid}.mp4",
         f"./**/{vid}.mp4"],
        v_mp4, "Adjunta el dataset o sube el .mp4 a inputs_demo/.")


def buscar_pt(t0, odir_vid, odir_stm):
    return resolver(
        "PT-gvhmr",
        [f"{odir_vid}/hmr4d_results.pt",
         f"{odir_stm}/hmr4d_results.pt",
         f"{OUT}/*/hmr4d_results.pt",
         f"{OUT}/*/*/hmr4d_results.pt"],
        lambda p: v_pt_gvhmr(p, t0),
        "gvhmr rc=0 pero ningun .pt trae smpl_params_global.",
        clave=os.path.getmtime, reversa=True)


INTENTOS_DETECTOR = [
    {"nombre": "yolo26x", "flags": "--detector yolo26x"},
    {"nombre": "defecto-8x", "flags": ""},
]


def validar_entorno():
    phase("[0] VALIDAR (ley previa: nada pesado corre sin esto)")
    faltan = []
    for cmd in ["ffmpeg", "ffprobe", "gvhmr", "scenedetect"]:
        hay = shutil.which(cmd) is not None
        print(f"  [{'OK' if hay else 'FALTA'}] cmd-{cmd}")
        if not hay:
            faltan.append(cmd)
    try:
        import torch
        hay_gpu = torch.cuda.is_available()
        print(f"  [{'OK' if hay_gpu else '--'}] cuda-GPU"
              f" ({torch.cuda.get_device_name(0) if hay_gpu else 'sin-gpu'})")
        if not hay_gpu:
            faltan.append("cuda-GPU")
    except Exception:
        print("  [--] torch: aun no instalado (lo trae [1] MODELOS)")
    if faltan:
        raise SystemExit(f"*** ENTORNO INCOMPLETO, falta: {faltan}. "
                         "Activa GPU o re-corre --instalar ***")


# ------------------------------------------------------------------ SHIM
# utils3d_moge EMBEBIDO (todo dentro de este run.py, cero archivos extra).
# Replica exacta (MIT, EasternJournalist/utils3d) de lo unico que MoGe usa
# en inferencia. MoGe lo busca PRIMERO (try import utils3d_moge), asi no
# choca con nada real. Puro torch: sin open3d, sin pip imposible.
def _shim_ensure_utils3d():
    try:
        import utils3d_moge  # noqa: F401 (si existe de verdad, se usa)
        return
    except ImportError:
        pass
    import torch
    import torch.nn.functional as _F
    from itertools import chain as _chain
    from numbers import Integral as _Integral
    import types as _types

    def _sliding_window(x, window_size, stride=None, dilation=None,
                        pad_size=None, pad_mode='constant', pad_value=0,
                        dim=None):
        if dim is None:
            dim = tuple(range(x.ndim))
        if isinstance(dim, _Integral):
            dim = (dim,)
        dim = [dim[i] % x.ndim for i in range(len(dim))]
        if isinstance(window_size, _Integral):
            window_size = (window_size,) * len(dim)
        if stride is None:
            stride = (1,) * len(dim)
        elif isinstance(stride, _Integral):
            stride = (stride,) * len(dim)
        if dilation is None:
            dilation = (1,) * len(dim)
        elif isinstance(dilation, _Integral):
            dilation = (dilation,) * len(dim)
        assert len(window_size) == len(stride) == len(dim)
        if pad_size is not None:
            if isinstance(pad_size, _Integral):
                pad_size = ((pad_size, pad_size),) * len(dim)
            elif isinstance(pad_size, tuple) and len(pad_size) == 2 \
                    and all(isinstance(p, _Integral) for p in pad_size):
                pad_size = (pad_size,) * len(dim)
            elif isinstance(pad_size, tuple) and all(
                    isinstance(p, tuple) and 1 <= len(p) <= 2
                    for p in pad_size):
                pad_size = pad_size * len(dim) if len(pad_size) == 1 \
                    else pad_size
                assert len(pad_size) == len(dim)
                pad_size = tuple(p * 2 if len(p) == 1 else p
                                 for p in pad_size)
            else:
                raise ValueError(f"Invalid pad_size {pad_size}")
            full_pad = [(0, 0) if i not in dim else pad_size[dim.index(i)]
                        for i in range(x.ndim)]
            x = _F.pad(x, tuple(_chain(*reversed(full_pad))),
                       mode=pad_mode, value=pad_value)
        for i in range(len(window_size)):
            x = x.unfold(dim[i], (window_size[i] - 1) * dilation[i] + 1,
                         stride[i])[..., ::dilation[i]]
        return x

    def _uv_map(height, width=None, top=0., left=0., bottom=1., right=1.,
                dtype=torch.float32, device=None):
        if isinstance(height, tuple):
            height, width = height
        u = torch.linspace(left + 0.5 / width * (right - left),
                           right - 0.5 / width * (right - left),
                           width, dtype=dtype, device=device)
        v = torch.linspace(top + 0.5 / height * (bottom - top),
                           bottom - 0.5 / height * (bottom - top),
                           height, dtype=dtype, device=device)
        return torch.stack(torch.meshgrid(u, v, indexing='xy'), dim=-1)

    def _pixel_coord_map(height, width=None, top=0, left=0,
                         convention='integer-center',
                         dtype=torch.float32, device=None):
        if isinstance(height, tuple):
            height, width = height
        u = torch.arange(left, left + width, dtype=dtype, device=device)
        v = torch.arange(top, top + height, dtype=dtype, device=device)
        if convention == 'integer-corner':
            u = u + 0.5
            v = v + 0.5
        u, v = torch.meshgrid(u, v, indexing='xy')
        return torch.stack([u, v], dim=2)

    def _masked_min(x, mask, dim=None, keepdim=False):
        f = torch.where(mask, x, torch.tensor(torch.inf, dtype=x.dtype,
                                              device=x.device))
        return f.min() if dim is None else f.min(dim=dim, keepdim=keepdim)

    def _masked_max(x, mask, dim=None, keepdim=False):
        f = torch.where(mask, x, torch.tensor(-torch.inf, dtype=x.dtype,
                                              device=x.device))
        return f.max() if dim is None else f.max(dim=dim, keepdim=keepdim)

    def _safe_inv(A):
        inv, info = torch.linalg.inv_ex(A)
        if info.any():
            inv = torch.where((info > 0)[..., None, None],
                              torch.full_like(inv, float('nan')), inv)
        return inv

    def _unproject_cv(uv, depth, intrinsics, extrinsics=None):
        K = torch.cat([
            torch.cat([intrinsics,
                       torch.zeros((*intrinsics.shape[:-2], 3, 1),
                                   dtype=intrinsics.dtype,
                                   device=intrinsics.device)], dim=-1),
            torch.tensor([[0, 0, 0, 1]], dtype=intrinsics.dtype,
                         device=intrinsics.device).expand(
                             *intrinsics.shape[:-2], 1, 4),
        ], dim=-2)
        transform = K @ extrinsics if extrinsics is not None else K
        pts = torch.cat([uv, torch.ones((*uv.shape[:-1], 1),
                                        dtype=uv.dtype,
                                        device=uv.device)],
                        dim=-1) * depth[..., None]
        pts = torch.cat([pts, torch.ones((*pts.shape[:-1], 1),
                                         dtype=uv.dtype,
                                         device=uv.device)],
                        dim=-1)
        return (pts @ _safe_inv(transform).mT)[..., :3]

    def _depth_map_to_point_map(depth, intrinsics, extrinsics=None):
        height, width = depth.shape[-2:]
        uv = _uv_map(height, width, dtype=depth.dtype, device=depth.device)
        return _unproject_cv(uv, depth,
                             intrinsics=intrinsics[..., None, :, :],
                             extrinsics=extrinsics[..., None, :, :]
                             if extrinsics is not None else None)

    def _t(a, like):
        if torch.is_tensor(a):
            return a.to(device=like.device, dtype=like.dtype)
        return torch.tensor(a, device=like.device, dtype=like.dtype)

    def _intrinsics_from_focal_center(fx, fy, cx, cy):
        like = fx if torch.is_tensor(fx) else fy if torch.is_tensor(fy) \
            else cx if torch.is_tensor(cx) else cy
        if not torch.is_tensor(like):
            like = torch.tensor(0.0)
        fx, fy, cx, cy = torch.broadcast_tensors(_t(fx, like), _t(fy, like),
                                                 _t(cx, like), _t(cy, like))
        zeros, ones = torch.zeros_like(fx), torch.ones_like(fx)
        return torch.stack([fx, zeros, cx,
                            zeros, fy, cy,
                            zeros, zeros, ones],
                           dim=-1).unflatten(-1, (3, 3))

    class _PT:
        sliding_window = staticmethod(_sliding_window)
        uv_map = staticmethod(_uv_map)
        pixel_coord_map = staticmethod(_pixel_coord_map)
        masked_min = staticmethod(_masked_min)
        masked_max = staticmethod(_masked_max)
        unproject_cv = staticmethod(_unproject_cv)
        depth_map_to_point_map = staticmethod(_depth_map_to_point_map)
        intrinsics_from_focal_center = staticmethod(
            _intrinsics_from_focal_center)

    mod = _types.ModuleType("utils3d_moge")
    mod.pt = _PT()
    sys.modules["utils3d_moge"] = mod


# ------------------------------------------------------------------ FASE 0
def fase_modelos():
    phase("[1/4] MODELOS (todo auto)")
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

    sh("pip install \"gvhmr[preproc]\" \"scenedetect[opencv]\" huggingface_hub scipy "
       "scikit-learn ultralytics 2>&1 | tail -1")
    if not os.path.exists(f"{W}/MoGe"):
        sh(f"git clone https://github.com/microsoft/MoGe.git {W}/MoGe 2>&1 | tail -1")
    # El clone sin install era el hueco: el modulo `moge` debe importarse.
    sh(f"pip install --no-deps {W}/MoGe 2>&1 | tail -1")
    os.environ["GVHMR_BODY_MODELS"] = BM
    sh(f"export GVHMR_BODY_MODELS={BM}; gvhmr download 2>&1 | tail -2")
    if not os.path.exists(MOGE):
        sh(f"cd {W} && python3 -c \"from huggingface_hub import snapshot_download; "
           "snapshot_download('Ruicheng/moge-2-vitb-normal', "
           "local_dir='./moge/moge-2-vitb-normal')\" 2>&1 | tail -1")
    sh(f"export GVHMR_BODY_MODELS={BM}; gvhmr info 2>&1 | grep -iE 'smplx|smpl |yolo|hmr2'")
    assert os.path.exists(MOGE), "*** MoGe no se descargo ***"
    # BLINDAJE TOTAL: probar los 8 imports + 4 comandos que el pipeline usa.
    # Si algo va a fallar, falla AQUI con nombre y apellido, no a mitad del LOTE.
    _shim_ensure_utils3d()
    _ffdir = os.path.dirname(os.path.abspath(__file__))
    faltan = []
    for nombre, imp in [
            ("moge-v2",
             f"import sys; sys.path.insert(0, '{_ffdir}'); "
             "import run as R; R._shim_ensure_utils3d(); "
             "from moge.model.v2 import MoGeModel"),
            ("torch", "import torch"),
            ("numpy", "import numpy"),
            ("scipy", "from scipy.spatial.transform import Rotation"),
            ("smplx", "import smplx"),
            ("sklearn", "from sklearn.linear_model import RANSACRegressor"),
            ("ultralytics", "from ultralytics import YOLO"),
            ("cv2", "import cv2"),
            ("huggingface_hub", "import huggingface_hub")]:
        r = subprocess.run(f"python3 -c \"{imp}; print('OK-{nombre}')\"",
                           shell=True, capture_output=True, text=True)
        if f"OK-{nombre}" in r.stdout:
            print(f"  VERIFY {nombre}: OK")
        else:
            print(f"  VERIFY {nombre}: FALTA")
            print(f"    -> {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else '?'}")
            faltan.append(nombre)
    for cmd in ["ffmpeg", "ffprobe", "gvhmr", "scenedetect"]:
        if shutil.which(cmd):
            print(f"  VERIFY cmd-{cmd}: OK")
        else:
            print(f"  VERIFY cmd-{cmd}: FALTA")
            faltan.append(f"cmd-{cmd}")
    try:
        r = subprocess.run("python3 -c \"import torch; "
                           "print('OK-cuda' if torch.cuda.is_available() "
                           "else 'SIN-CUDA')\"",
                           shell=True, capture_output=True, text=True)
        print(f"  VERIFY cuda: {r.stdout.strip() or 'FALTA'}")
        if "OK-cuda" not in r.stdout:
            faltan.append("cuda-GPU")
    except Exception:
        faltan.append("cuda-GPU")
    if faltan:
        raise SystemExit(f"*** INSTALAR INCOMPLETO, falta: {faltan}. "
                         "Re-corre esta celda ***")
    print("  MODELOS-OK (todo importado y verificado)")


# --------------------------------------------------------------- VIDEO-IN
def video_in():
    phase("[2/4] VIDEO-IN (descubre .mp4, valida contenido)")
    found = subprocess.run("find /kaggle/input -type f -iname '*.mp4' 2>/dev/null",
                           shell=True, capture_output=True, text=True).stdout.split()
    for s in found:
        shutil.copy(s, IN)
    vids = sorted(s[:-4] for s in os.listdir(IN) if s.endswith(".mp4"))
    # Ley: el mismo validador para todos, el corrupto se descarta con aviso.
    buenos = []
    for v in vids:
        ok, detalle = v_mp4(f"{IN}/{v}.mp4")
        if ok:
            buenos.append(v)
        else:
            print(f"  [--] {v}.mp4 descartado ({detalle})")
    vids = buenos
    if not vids:
        raise SystemExit("*** SIN VIDEO: adjunta tu dataset (Add Input) o sube el "
                         ".mp4 a inputs_demo/ y re-corre ***")
    print("  VIDEOS (escribe uno tal cual en VID?):")
    for v in vids:
        print(f"   - {v}")
    return vids


# ------------------------------------------------------------------ SPLIT
def fase_split(vid, t):
    phase("[3/4] SPLIT (cortes, >=2s)")
    vpath = buscar_video(vid)
    segdir = f"{W}/segments_{vid}"
    os.makedirs(segdir, exist_ok=True)
    sh(f"cd {W} && scenedetect -i \"{vpath}\" detect-content -t {t} "
       f"list-scenes -o segments_{vid}/", check=False)
    csvp = f"{segdir}/{vid}-Scenes.csv"
    if not os.path.exists(csvp):
        raise SystemExit("*** SIN CSV: scenedetect no detecto nada "
                         "(video corrupto o -t muy alto) ***")
    kept = []

    def _col(row, *aliases):
        norm = {}
        for k in row.keys():
            if not isinstance(k, str):
                continue
            norm[k.strip().lower()] = k
        for a in aliases:
            if a in norm:
                return row[norm[a]]
        return None

    with open(csvp) as f:
        lines = f.read().splitlines()
    # El CSV de list-scenes trae una 1ra linea portada ("Timecode List:,...");
    # se lee desde la linea que empiece con Scene (probado en vivo 2026-10-03).
    hi = 0
    while hi < len(lines) and not lines[hi].strip().lower().startswith("scene"):
        hi += 1
    if hi >= len(lines):
        raise SystemExit("*** CSV sin linea 'Scene Number'. Primeras 3 lineas: "
                         f"{lines[:3]}. Pega esto para ajustar ***")
    rows = list(csv.DictReader(lines[hi:]))
    hdr = list(rows[0].keys()) if rows else []
    if rows and (_col(rows[0], "start time (seconds)", "start time", "start_time",
                      "start (seconds)", "start") is None
                 or _col(rows[0], "start timecode", "start_timecode",
                         "start_tc", "start tc") is None):
        raise SystemExit("*** CSV con columnas desconocidas. Vistas: "
                         f"{hdr}. Pega esto para ajustar alias ***")
    for r in rows:
        t0 = _col(r, "start time (seconds)", "start time", "start_time",
                  "start (seconds)", "start")
        t1 = _col(r, "end time (seconds)", "end time", "end_time",
                  "end (seconds)", "end")
        s0 = _col(r, "start timecode", "start_timecode", "start_tc", "start tc")
        s1 = _col(r, "end timecode", "end_timecode", "end_tc", "end tc")
        if not t0 or not t1 or not s0 or not s1:
            continue
        try:
            dur = float(t1) - float(t0)
        except ValueError:
            continue
        if dur < 2.0:
            continue
        nm = f"seg{len(kept) + 1:02d}"
        sh(f"ffmpeg -y -v error -ss {s0} -to {s1} "
           f"-i \"{vpath}\" -c copy {segdir}/{nm}.mp4", check=True)
        kept.append((nm, s0, s1))
    if not kept:
        raise SystemExit("*** 0 SEGMENTOS >=2s: baja el umbral o revisa el video ***")
    print("  SEGMENTOS:", [k[0] for k in kept])
    # Sin JPG: salida final solo .bvh. El frame medio para piso se lee
    # en memoria (cv2 directo) dentro de ley_piso, sin guardar nada.
    return segdir, [k[0] for k in kept]


# ------------------------------------------------- LEYES MULTI (EDOD)
# Tablas, no ramas. Todo lo que puede fallar (pesos, yaml, video raro)
# tiene fila de repuesto o aviso VACIO. Nada revienta el lote.
TABLA_YOLO_TRACK = ["yolo26x.pt", "yolo11x.pt"]
MARGEN_TRACKLET = 1.3
YAML_BOTSORT_DEFECTO = """# BoT-SORT para IDs estables (autocreado por run.py si falta).
tracker_type: botsort
track_high_thresh: 0.5
track_low_thresh: 0.1
new_track_thresh: 0.6
track_buffer: 90
match_thresh: 0.8
fuse_score: True
proximity_thresh: 0.5
appearance_thresh: 0.25
with_reid: False
gmc_method: none
"""


def buscar_tracker_yaml():
    """El yaml vive junto a run.py. Si no existe se AUTOCREA desde la
    tabla embebida: el codigo nunca depende del archivo. Libre."""
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "custom_tracker.yaml")
    for cand in [base, "./custom_tracker.yaml",
                 f"{W}/custom_tracker.yaml"]:
        try:
            if os.path.isfile(cand) and os.path.getsize(cand) > 100:
                print(f"  [ENCONTRADO] TRACKER-YAML: {cand}")
                return cand
        except Exception:
            continue
    try:
        with open(base, "w") as f:
            f.write(YAML_BOTSORT_DEFECTO)
        print(f"  [TRACKER-YAML] autocreado: {base}")
        return base
    except Exception as e:
        print(f"  [TRACKER-YAML] sin archivo ({e}): uso botsort de fabrica")
        return None


def ley_trackear(seg_mp4):
    """Ley TRACK: YOLO + BoT-SORT sobre el segmento, solo clase 0
    (persona). Llena IDS[10] via ley_rankear (top por frames).
    0 personas o sin ultralytics = VACIO con aviso, nunca error."""
    try:
        from ultralytics import YOLO
    except Exception:
        print("  [TRACK] sin ultralytics (re-corre --instalar): VACIO")
        return 0
    try:
        yamlp = buscar_tracker_yaml() or "botsort.yaml"
        modelo = None
        for peso in TABLA_YOLO_TRACK:
            try:
                modelo = YOLO(peso)
                print(f"  [TRACK] peso: {peso}")
                break
            except Exception as e:
                print(f"  [TRACK] peso {peso} no cargo "
                      f"({str(e)[:80]}), siguiente...")
        if modelo is None:
            print("  [TRACK] ningun peso YOLO cargo: VACIO")
            return 0
        hall = {}
        for fidx, r in enumerate(modelo.track(
                source=seg_mp4, tracker=yamlp, classes=[0],
                verbose=False, persist=True)):
            b = r.boxes
            if b is None or b.id is None:
                continue
            for tid, xy, cf in zip(b.id.int().tolist(),
                                   b.xyxy.tolist(), b.conf.tolist()):
                e = hall.get(tid)
                if e is None:
                    hall[tid] = [1, float(cf),
                                 [(fidx, float(xy[0]), float(xy[1]),
                                   float(xy[2]), float(xy[3]))]]
                else:
                    e[0] += 1
                    e[1] += float(cf)
                    e[2].append((fidx, float(xy[0]), float(xy[1]),
                                 float(xy[2]), float(xy[3])))
        hallazgos = {t: (v[0], v[1] / v[0], v[2]) for t, v in hall.items()}
        n = ley_rankear(hallazgos)
        if n == 0:
            print("  [TRACK] 0 personas: VACIO")
            return 0
        print(f"  [TRACK] {n} persona(s):")
        for i in range(n):
            print(f"    i={i} track={ID_TRK[i]} frames={ID_NFR[i]} "
                  f"conf={ID_CNF[i]:.2f}")
        return n
    except Exception as e:
        print(f"  [TRACK] fallo ({type(e).__name__}: {str(e)[:120]}): VACIO")
        return 0


def _tam_video(path):
    pr = subprocess.run(
        f"ffprobe -v error -select_streams v:0 "
        f"-show_entries stream=width,height -of csv=p=0 \"{path}\"",
        shell=True, capture_output=True, text=True)
    nums = re.findall(r"\d+", pr.stdout or "")
    if len(nums) < 2:
        raise RuntimeError(f"ffprobe vacio para {path}")
    return int(nums[0]), int(nums[1])


def ley_tracklets(seg_mp4, segdir, nm):
    """Ley TRACKLET: un mini-video por ID activo con >=MIN_FRAMES.
    Recorte = union de sus cajas x MARGEN_TRACKLET, tiempo = su racha
    (fidx/30). Sin codec raro: mismo normalize del pipeline. El ID corto
    se apaga (muerte perezosa) sin archivo y con aviso. Devuelve
    [(i, mini_mp4)] de los que si generan."""
    Wd, Hd = _tam_video(seg_mp4)
    vivos = []
    for i in range(ID_N):
        if not ID_ACT[i]:
            continue
        trk = ID_TRK[i]
        if not id_valido(ID_NFR[i]):
            ID_ACT[i] = False
            print(f"    id={trk}: {ID_NFR[i]}f <60: descartado (sin archivo)")
            continue
        try:
            cajas = ID_CAJAS[i]
            f0 = min(c[0] for c in cajas)
            f1 = max(c[0] for c in cajas)
            x0 = min(c[1] for c in cajas)
            y0 = min(c[2] for c in cajas)
            x1 = max(c[3] for c in cajas)
            y1 = max(c[4] for c in cajas)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            hw = max((x1 - x0) * MARGEN_TRACKLET / 2, 32)
            hh = max((y1 - y0) * MARGEN_TRACKLET / 2, 32)
            ax = max(int(cx - hw), 0)
            ay = max(int(cy - hh), 0)
            bx = min(int(cx + hw), Wd)
            by = min(int(cy + hh), Hd)
            ax -= ax % 2
            ay -= ay % 2
            bw = (bx - ax) // 2 * 2
            bh = (by - ay) // 2 * 2
            if bw < 16 or bh < 16:
                raise RuntimeError(f"recorte degenerado {bw}x{bh}")
            out = f"{segdir}/{nm}_id{trk}.mp4"
            t0, t1 = f0 / FPS, (f1 + 1) / FPS
            sh(f"ffmpeg -y -v error -ss {t0:.3f} -to {t1:.3f} -i \"{seg_mp4}\" "
               f"-vf \"crop={bw}:{bh}:{ax}:{ay},setsar=1,fps=30\" -r 30 "
               f"-c:v libx264 -pix_fmt yuv420p -crf 18 -an \"{out}\"",
               check=True)
            ok, _ = v_mp4(out)
            if not ok:
                raise RuntimeError("tracklet ilegible tras corte")
            print(f"    id={trk}: TRACKLET-OK {ID_NFR[i]}f -> {out}")
            vivos.append((i, out))
        except Exception as e:
            ID_ACT[i] = False
            print(f"    id={trk}: tracklet FAIL "
                  f"({type(e).__name__}: {str(e)[:100]}), apagado")
    return vivos


def ley_piso(d, video_mp4, moge):
    """Ley PISO por ID: plano MoGe-2 en frame medio + correccion por
    tobillos (misma matematica del pipeline ganador). Total: si algo
    falla devuelve ['piso=omitido:...'] y delta=0, nunca revienta."""
    import torch
    import numpy as np
    try:
        import cv2
        from sklearn.linear_model import RANSACRegressor
        g = d["smpl_params_global"]
        n = g["body_pose"].shape[0]
        cap = cv2.VideoCapture(video_mp4)
        total = int(cap.get(7) or 1)
        cap.set(1, total // 2)
        ok, img = cap.read()
        cap.release()
        if not ok or img is None:
            raise RuntimeError("sin frame medio")
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
        rep = [f"plano: pts={len(pts)} inl={pct:.0f}%"]
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
            tp.append(lab + " cf=" + str(round(cf, 2)) + " med=" + ms
                      + " std=" + ss)
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
        return rep
    except Exception as e:
        return [f"piso=omitido:{type(e).__name__}:{str(e)[:100]}"]


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
    phase("[4/4] LOTE-MULTI (TRACK->TRACKLET->POSE/PISO/EXPORT por ID->SOLO BVH)")
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("*** SIN GPU: activa acelerador (Settings -> Accelerator -> GPU) ***")
    # Shim embebido (ver arriba): MoGe lo busca PRIMERO.
    _shim_ensure_utils3d()
    from moge.model.v2 import MoGeModel
    res = f"{W}/Resultados/{vid}"
    os.makedirs(res, exist_ok=True)
    os.environ["GVHMR_BODY_MODELS"] = BM
    # Sin SMPL-X no hay BVH: se busca donde este, no donde se supone.
    buscar_smplx()
    buscar_moge()
    moge = MoGeModel.from_pretrained(MOGE).cuda().eval()
    bvhs = []
    for nm in segs:
        R = f"{res}/{nm}"
        os.makedirs(R, exist_ok=True)
        try:
            seg = f"{segdir}/{nm}.mp4"
            if not os.path.exists(seg):
                raise RuntimeError(f"no existe segmento: {seg}")
            ww, hh = _tam_video(seg)
            vf = "scale=1280:720" if ww > hh else "scale=720:1280"
            norm = f"{segdir}/{nm}_30fps.mp4"
            sh(f"ffmpeg -y -v error -i \"{seg}\" -vf \"{vf}:flags=lanczos,setsar=1,fps=30\" "
               f"-r 30 -c:v libx264 -pix_fmt yuv420p -crf 18 -an \"{norm}\"", check=True)
            # Leyes en orden: TRACK llena IDS[10], TRACKLET recorta por ID,
            # POSE/PISO/EXPORT corren por ID. Un ID falla solo, los demas siguen.
            n_ids = ley_trackear(norm)
            if n_ids == 0:
                print(f"  {nm}: VACIO (0 personas), sigue")
                continue
            vivos = ley_tracklets(norm, segdir, nm)
            for i, tracklet in vivos:
                trk = ID_TRK[i]
                tag = f"{nm}_id{trk}"
                try:
                    corr = f"{R}/{tag}_corrected.pt"
                    if os.path.exists(corr):
                        print(f"    id={trk}: YA HECHO (regenero BVH del .pt)")
                        d = torch.load(corr, map_location="cpu")
                        n = d["smpl_params_global"]["body_pose"].shape[0]
                        bvh = f"{W}/{vid}_{tag}_{n}f.bvh"
                        escribir_bvh(d, bvh)
                        bvhs.append(bvh)
                        print(f"    id={trk}: BVH-OK -> {bvh}")
                        continue
                    odir = f"{OUT}/{os.path.splitext(os.path.basename(tracklet))[0]}"
                    sh(f"rm -rf {odir}")
                    t0 = datetime.datetime.now().timestamp()
                    # Tabla, no ramas: se prueba cada detector en orden, gana el 1ro OK.
                    r = None
                    for intento in INTENTOS_DETECTOR:
                        fl = intento["flags"]
                        r = sh(f"gvhmr demo \"{tracklet}\" -o {OUT} -s --f-mm {fmm} "
                               f"{fl} --no-render")
                        if r.returncode == 0:
                            print(f"    id={trk} detector: {intento['nombre']}")
                            break
                        print(f"    id={trk} detector {intento['nombre']} fallo "
                              f"(rc={r.returncode}), siguiente...")
                    if r.returncode != 0:
                        raise RuntimeError(
                            "gvhmr fallo rc=%d cola=%s" %
                            (r.returncode,
                             (r.stderr or r.stdout or "")[-500:]))
                    # El .pt se describe (que trae) y se busca donde este.
                    pt = buscar_pt(t0, odir, odir)
                    d = torch.load(pt, map_location="cpu")
                    n = d["smpl_params_global"]["body_pose"].shape[0]
                    for linea in ley_piso(d, tracklet, moge):
                        print(f"    id={trk}: {linea}")
                    torch.save(d, corr)
                    bvh = f"{W}/{vid}_{tag}_{n}f.bvh"
                    escribir_bvh(d, bvh)
                    bvhs.append(bvh)
                    print(f"    id={trk}: BVH-OK frames={n} -> {bvh}  <-- baja con 1 clic")
                except Exception as e:
                    ID_ACT[i] = False
                    print(f"    id={trk}: FAIL {type(e).__name__}: "
                          f"{str(e)[:200]} (los demas siguen)")
        except Exception as e:
            print(f"  {nm}: FAIL {type(e).__name__}: {str(e)[:300]}")
    print(f"  LOTE FIN: {len(bvhs)} BVH en {W}/")
    return bvhs


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    print("== PIPELINE MULTI | 1 video -> N personas -> N BVH (piso MoGe-2) ==")
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
    validar_entorno()
    fase_modelos()
    vids = video_in()
    if vid not in vids:
        raise SystemExit(f"*** VID '{vid}' no esta en inputs_demo/. Opciones: "
                         f"{', '.join(vids)} (revisa letra por letra) ***")
    segdir, segs = fase_split(vid, t)
    bvhs = fase_lote(vid, fmm, segdir, segs)
    print(f"\nFIN: {len(bvhs)} BVH en /kaggle/working/ (baja con 1 clic). "
          "Piso ya horneado; en local pasales limpiar_bvh.py para in-place.")


if __name__ == "__main__":
    main()
