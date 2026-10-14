# ff — GVHMR + piso MoGe-2 por ID → BVH + PT + NPZ

Pipeline Kaggle: 1 video → N personas → N archivos. 1 persona = 1 ID = 1 `.bvh` + 1 `.pt` + 1 `.npz`.

## .bvh vs .pt / .npz (para tu pana)

- `.bvh`: jerarquía de huesos para Unity / Unreal / Blender. Para motor 3D usa este.
- `.pt` / `.npz`: tensores puros (`body_pose, global_orient, transl, betas`) listos para PyTorch / TensorFlow sin parser ni pérdida. Para entrenar usa estos.

## Instalar (1 vez, con GPU T4)

```bash
!python3 ff/run.py --instalar
Instala gvhmr[preproc], scenedetect, ultralytics, baja SMPLX + moge-2-vitb, verifica 9 imports + ffmpeg/ffprobe/gvhmr/scenedetect + CUDA.
1 video suelto (modo clásico)
VID = 'tiro_libre_01'
!python3 ff/run.py $VID
# opcional: !python3 ff/run.py $VID 24 27
Salida: /kaggle/working/{VID}_{seg}_id{ID}_{n}f.bvh
1 video por ruta (para loop / dataset)
!python3 ff/run.py --source "/kaggle/input/dataset_futbol/tiro_libre_01.mp4" --out_dir "/kaggle/working/dataset_procesado/tiro_libre_01"
# con params: --f-mm 24 --threshold 27 --device cuda
Si --out_dir ya termina en el nombre del video no duplica carpeta.
Dataset masivo en lote (recomendado fútbol)
Sube a Kaggle Datasets:
/kaggle/input/dataset_futbol/
  tiro_libre_01.mp4
  pase_02.mp4
  regate_03.mp4
Corre:
!python3 ff/run.py --lote --input-dir /kaggle/input/dataset_futbol --output-dir /kaggle/working/dataset_procesado
# sin zip: añade --no-zip
# zip custom: --zip-path /kaggle/working/mio.zip
Hace todo solo: SPLIT (scenedetect -t 27, descarta <2s) → TRACK (YOLO + BoT-SORT) → TRACKLET por ID (descarta <60 frames) → gvhmr demo -s --f-mm 24 --no-render → piso MoGe-2 → exporta. Un ID falla solo, un video falla solo, el resto sigue.
Estructura final:
/kaggle/working/dataset_procesado/
  tiro_libre_01/
    tiro_libre_01_seg01_id1_120f.bvh
    tiro_libre_01_seg01_id1_120f.pt
    tiro_libre_01_seg01_id1_120f.npz
  manifest.csv
/kaggle/working/dataset_futbol_final.zip
manifest.csv: vid,seg,id,frames,bvh,pt,npz
Multi-jugador
YOLO + Tracker asigna id=1,2,... por persona. Cada ID genera su carpeta de archivos. Dile a tu pana: IA → .pt/.npz, motor → .bvh.
Flags
--instalar
$VID [FMM] [T]
--source <mp4> --out_dir <carpeta> [--f-mm 24] [--threshold 27]
--lote [--input-dir D] [--output-dir D] [--f-mm 24] [-t 27] [--zip / --no-zip]
--lote acepta: --batch, --todos
--out_dir = --out-dir = --out
--input-dir acepta .mp4,.avi,.mov,.mkv
