# ff — pipeline ganador seg01 (GVHMR preciso + piso MoGe-2)

## Uso (Kaggle, 2 líneas a mano + responder preguntas, nada más)
```text
!git clone https://github.com/luisenriquezparra-pixel/ff.git
!python3 ff/run.py
```
El script pregunta `VID?` → escribes el nombre del video (sin `.mp4`; debe estar
adjuntado en `/kaggle/input` o subido a `/kaggle/working/inputs_demo/`).
Luego pregunta `FMM?` (Enter = 24, 1x iPhone) y `T?` (Enter = 27, umbral de corte).
Al terminar: **baja el `LOTE_<vid>_<fecha>.zip` de `/kaggle/working/Resultados/`** (1 clic).
Otro video = correr `!python3 ff/run.py` de nuevo (sin reinstalar, con checkpoints).

## Qué hace (5 fases, full-auto con checkpoints)
1. **Modelos**: pip (gvhmr, scenedetect, huggingface_hub) + clone MoGe + `gvhmr download`
   + pesos MoGe-2 + SMPLX vía Kaggle API desde TU dataset privado
   (Secrets `KAGGLE_USERNAME`/`KAGGLE_KEY`, una vez en la vida; sin secrets pide
   adjuntar el dataset a mano). Re-corre sin reinstalar lo ya hecho.
2. **VIDEO-IN**: descubre `.mp4` solo, los deja en `inputs_demo/`, imprime nombres exactos.
3. **SPLIT**: PySceneDetect `-t 27` → segmentos ≥2s + hoja de contactos.
4. **LOTE** por segmento: normaliza 30fps → `gvhmr demo -s --f-mm 24 --no-render`
   → keyframe medio → MoGe-2 plano RANSAC → tobillos ventana 7×7 → delta
   (regla std>6cm) → `*_corrected.pt` + depths torso + `report.txt` con FLAGS.
5. **ZIP**: solo el zip vive en `Resultados/` (excluye regenerables:
   `mesh/camera/preprocess/0_input_video/_geo.pt/_30fps.mp4/_mid.jpg`).

## Receta locked (no tocar sin medir)
`-s` cámara fija · `--f-mm` con guion · `--no-render` (bug vertical) ·
MoGe-2 vitb-normal (NO v3) · RANSAC tol 0.05 · tobillos NUNCA 1 píxel ·
`delta = mediana − 0.10` · sin `report.txt` OK no va a BVH.

## FLAGS del report (por segmento)
- `OK`: entra a conversión BVH.
- `WARN:inliers<30`: plano dudoso, cambiar keyframe.
- `WARN:ambos-std>6`: sin tobillo confiable → velocidad anotada, no referencia.
- `WARN:delta>10cm`: corrección grande, revisar.
- `FAIL`: fase que murió + motivo (re-corre: hace skip de lo ya hecho).

## Mapa del repo
- `run.py`: todo el pipeline en 1 archivo (público-seguro: SIN secretos, SIN claves, SIN pesos).
- Los pesos/modelos JAMÁS van al repo: GVHMR/MoGe son públicos (auto),
  SMPL-X es gated MPI (vía tu dataset privado en Kaggle, legal: tu copia).
