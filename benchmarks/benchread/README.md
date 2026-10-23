# BenchReAD (fondo de ojo, one-class)

Comparación de imgad con [BenchReAD](https://github.com/DopamineLcy/BenchReAD) (Lian et al., MICCAI 2025) bajo su
propio protocolo one-class: entrenamiento con las 7.786 imágenes normales etiquetadas de EDDFS + BRSET, selección
en su conjunto de validación y evaluación en dos conjuntos externos, RIADD (RFMiD) y JSIEC, con anomalías vistas y
no vistas. Qué evalúa BenchReAD, cómo define las particiones, sus cifras publicadas, licencias y diferencias de
protocolo: [`NOTES.md`](NOTES.md).

| Categoría | Entrenamiento (normales) | Test normal | Test anómalo | Uso |
|---|---:|---:|---:|---|
| `benchread-val` | 7.786 | 100 | 100 (`abnormal/`) | elegir la configuración |
| `benchread-riadd` | 7.786 | 669 | 1.634 en 15 carpetas (DR, AMD, ODC, MYA, DN, MH, BRVO, …) | informar |
| `benchread-jsiec` | 7.786 | 38 | 686 en 14 carpetas (DR, LOC, PM, BRVO, …) | informar |

## Datos (descarga manual)

BenchReAD no redistribuye imágenes ni listas de partición; `prepare_benchread.py` reconstruye su partición
(`seed = 0`) a partir de las etiquetas de cada conjunto.

| Conjunto | Dónde | Licencia | Para `--…` |
|---|---|---|---|
| EDDFS | enlaces del README de <https://github.com/xia-xx-cv/EDDFS_dataset> + `datas/EDDFS/Annotation/{train,test}.csv` | no comercial, sin redistribución | `--eddfs` (imágenes y los dos CSV en cualquier subcarpeta) |
| BRSET 1.0.1 | <https://physionet.org/content/brazilian-ophthalmological/1.0.1/> | PhysioNet Credentialed Health Data License 1.5.0: cuenta acreditada, curso CITI y DUA | `--brset` (`fundus_photos/` y `labels.csv` o `labels_brset.csv`) |
| RIADD / RFMiD | <https://riadd.grand-challenge.org/download-all-classes/> (las tres particiones) | CC BY 4.0 | `--riadd` (carpetas `Training`, `Validation`, `Test` y sus CSV) |
| JSIEC | <https://www.kaggle.com/datasets/linchundan/fundusimage1000> (447 MB) | ODbL | `--jsiec` (`1000images/0.0.Normal`, …) |

## Pasos

```bash
# 1. Reconstruir la partición de BenchReAD en el formato MVTec (recorte al campo de visión a 512 px, como BenchReAD;
#    JSIEC sin recortar, como BenchReAD). Para si los recuentos no coinciden con los del artículo (--force escribe igual).
python benchmarks/benchread/prepare_benchread.py --eddfs /data/EDDFS --brset /data/brset/1.0.1 \
  --riadd /data/RIADD --jsiec /data/JSIEC --out /data/benchread_imgad -j 8

# 2. Poner data.root = /data/benchread_imgad en configs/benchread_benchmark.yaml y configs/benchread_anomalib.yaml.
#    Extractores + detectores (mismo grid que PALM, entrenados con las 7.786 normales):
imgad benchmark --config configs/benchread_benchmark.yaml --out benchmarks/benchread/results.csv \
  --scores-out benchmarks/benchread/scores.joblib --report benchmarks/benchread/report --stats \
  -j 8 --devices auto --hf-token-file token
#    Modelos de Anomalib (submuestra de 1.000 normales por semilla; ver el YAML):
imgad benchmark --config configs/benchread_anomalib.yaml --out benchmarks/benchread/results_anomalib.csv \
  --scores-out benchmarks/benchread/scores_anomalib.joblib -j 2 --devices auto

# 3. Selección en validación, evaluación en RIADD y JSIEC y tabla frente a las cifras publicadas:
python benchmarks/benchread/compare_benchread.py benchmarks/benchread/results.csv benchmarks/benchread/results_anomalib.csv \
  --scores benchmarks/benchread/scores.joblib benchmarks/benchread/scores_anomalib.joblib \
  --data /data/benchread_imgad --out benchmarks/benchread/comparison.md
```

`compare_benchread.py` informa: la configuración con mejor AUROC en validación (global y por familia: extractor +
detector, Anomalib, *prompts*), una configuración fijada de antemano que no usa ninguna anomalía de BenchReAD (por
defecto la elegida en PALM, RetiZero + L2 + LOF; `--preselected` la cambia), el oráculo de cada conjunto de test, la
tabla de AUROC junto a la Fig. 2 del artículo y, con `--scores`, F1/especificidad/sensibilidad por categoría con el
umbral de imgad junto a la Tabla 1.

## Coste y requisitos

- Extracción: ~11.000 imágenes distintas por extractor. La caché de imgad usa la ruta de cada enlace simbólico, de
  modo que las 7.786 de entrenamiento se extraen una vez por categoría (~31.000 extracciones por extractor).
- Detectores con 7.786 imágenes de entrenamiento: OCSVM, KDE y LOF son los más lentos, pero caben en memoria.
- PatchCore y otros modelos de banco de memoria no caben con 7.786 imágenes (~50 GB de parches a 256 px), por eso
  `benchread_anomalib.yaml` usa 1.000 imágenes por semilla.

## Prueba sin datos

`prepare_benchread.py` se probó sobre una copia simulada de la estructura esperada (etiquetas reales de EDDFS, CSV
sintéticos de BRSET, RFMiD y JSIEC con los recuentos del artículo, imágenes diminutas): reproduce los recuentos de
BenchReAD (7.786 / 100 + 100 / 669 + 1.634 / 38 + 686) y da mensajes claros con rutas erróneas, carpetas sin
etiquetas o descargas aún comprimidas. El recorte coincide con la implementación de BenchReAD (OpenCV +
scikit-image) salvo por el filtro de remuestreo.

**Pendiente:** la prueba de humo de `imgad benchmark` con ambos YAML (extractor `stats` de `tests/conftest.py`) sobre
esos datos simulados **no se completó** (el primer intento falló por un fallo del propio guion de prueba, ver el PR, y
se detuvo antes de repetirla). `compare_benchread.py` tampoco se ha ejecutado aún sobre resultados reales ni simulados.
