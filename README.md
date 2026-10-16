# imgad — Detección de anomalías en imágenes con modelos fundacionales + algoritmos clásicos (+ Anomalib)

[![CI](https://github.com/joheras/imgad/actions/workflows/ci.yml/badge.svg)](https://github.com/joheras/imgad/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Framework modular para detección de anomalías a nivel de imagen en el escenario **one-class**
(se entrena solo con imágenes normales):

```
imágenes ──► FeatureExtractor ──► preprocesado ──► Detector clásico ──► score de anomalía
            (CLIP, SigLIP, DINOv2…   (l2, PCA,       (kNN, Mahalanobis,     (+ umbral calibrado)
             cualquier modelo HF)     standard…)      LOF, OCSVM, IF…)
```

## Instalación

```bash
pip install -e .                 # núcleo
pip install -e ".[anomalib]"     # + modelos de Anomalib (PatchCore, PaDiM, EfficientAD…)
pip install -e ".[all]"          # + open_clip, pyod, pandas, anomalib, pytest
```

## Uso rápido

```python
from imgad import AnomalyDetectionPipeline, HFFeatureExtractor, build_detector, data
from imgad.evaluation import evaluate_scores

ds = data.mvtec_ad("/ruta/mvtec_anomaly_detection", "bottle")

pipe = AnomalyDetectionPipeline(
    extractor=HFFeatureExtractor("openai/clip-vit-base-patch16"),
    detector=build_detector("knn", k=1),
    preprocessing=["l2"],
    cache_dir=".feature_cache",          # cachea features por imagen
)
pipe.fit(ds.train_paths)                 # solo imágenes normales
scores = pipe.score(ds.test_paths)       # mayor = más anómalo
preds  = pipe.predict(ds.test_paths)     # 0/1 con umbral calibrado
print(evaluate_scores(ds.test_labels, scores, preds))

pipe.save("bottle.joblib")               # guarda detector + config (no los pesos)
pipe = AnomalyDetectionPipeline.load("bottle.joblib")
```

## Inferencia con un modelo ya entrenado

El `.joblib` guarda el detector entrenado, el preprocesado, el umbral, la configuración del
extractor y metadatos (nº de imágenes de entrenamiento, fecha, origen del umbral). Los pesos del
modelo de HF no se guardan: se vuelven a cargar desde el Hub (o la caché local de HF) al hacer `load`.

```python
from imgad import AnomalyPredictor

predictor = AnomalyPredictor.load("bottle.joblib")   # device auto (cuda/mps/cpu); device="cpu" para forzar
r = predictor.predict_one("nueva.png")               # ruta, PIL.Image o array numpy
r.is_anomaly, r.score, r.normal_percentile

results = predictor.predict("carpeta/")              # carpeta, lista de rutas/imágenes, o mezcla
predictor.save_results(results, "preds.csv")         # .csv o .json
for r in predictor.predict_iter("carpeta_enorme/", chunk_size=256): ...   # memoria constante
```

Cada `Prediction` incluye:

| Campo | Significado |
|---|---|
| `score` | score bruto (escala dependiente del detector; mayor = más anómalo) |
| `is_anomaly` | `score > threshold` |
| `threshold` | umbral usado |
| `normal_percentile` | fracción de imágenes normales de entrenamiento con score menor (0–1). Comparable entre detectores: ~0.5 = normal típica, 1.0 = más rara que cualquier normal vista |

**Ajustar el umbral** con un conjunto de validación etiquetado (sin reentrenar):

```python
pipe = AnomalyDetectionPipeline.load("bottle.joblib")
pipe.calibrate_threshold(val_paths, val_labels, method="f1")   # f1 | youden | fpr | tpr
pipe.save("bottle_cal.joblib")
# Solo con imágenes normales: method="fpr", target_fpr=0.01  -> ≤1% de falsas alarmas
```

O al vuelo: `AnomalyPredictor.load("bottle.joblib", threshold=0.02)`.

Atajos: el extractor puede ser un id de HF (`"facebook/dinov2-base"`) y el detector un nombre
(`"mahalanobis"`) o un dict (`{"name": "knn", "k": 5}`).

## Extractores de características

`HFFeatureExtractor(model_name, pooling="auto", layer=None, normalize=True, ...)` funciona con
cualquier modelo de visión del Hub cargable con `AutoModel`:

| Familia | Ejemplos | Pooling por defecto |
|---|---|---|
| Duales tipo CLIP | `openai/clip-*`, `google/siglip-*`, `google/siglip2-*`, `facebook/metaclip-*`, `kakaobrain/align-base`, `laion/CLIP-*` (formato HF) | `projection` → `get_image_features` |
| Backbones de visión | `facebook/dinov2-*`, `google/vit-*`, `microsoft/beit-*`, `facebook/convnext-*`, `microsoft/resnet-*` | `pooler` |

Opciones de `pooling`: `auto`, `projection`, `pooler`, `cls`, `mean` (media de tokens de parche),
`cls+mean`. Con `layer` se usan capas intermedias (`layer=-4`, o `layer=[6, 9, 12]` para concatenar
varias), que suelen capturar mejor defectos locales que la capa final, más semántica.

Para checkpoints en formato **open_clip** (`hf-hub:laion/...`, EVA-CLIP, DataComp…):
`OpenCLIPFeatureExtractor("hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K")`.

## Detectores

Todos siguen la misma API (`fit`, `score`, `predict`) y la convención **mayor score = más anómalo**.

| Nombre | Método | Notas |
|---|---|---|
| `knn` | Distancia a los k vecinos normales | Línea base muy fuerte con CLIP/DINO; `metric="cosine"` |
| `mahalanobis` / `gaussian` | Distancia de Mahalanobis (Ledoit-Wolf, OAS, MCD) | Muy buena con features de modelos fundacionales |
| `centroid` | Distancia al embedding medio | Mínimo coste |
| `kmeans` | Distancia al centroide más cercano | Normalidad multimodal |
| `lof` | Local Outlier Factor (modo novelty) | |
| `gmm` | −log-verosimilitud de una mezcla gaussiana | |
| `kde` | −log-densidad KDE | Mejor tras `pca:<n>` a pocas dimensiones |
| `ocsvm` | One-Class SVM | |
| `iforest` | Isolation Forest | |
| `pca` | Error de reconstrucción PCA | |
| `ensemble` | Media de detectores normalizados (z-score o rango) | |
| `sklearn` | Envuelve cualquier estimador sklearn | `SklearnDetector(estimator=EllipticEnvelope())` |
| `pyod` | Cualquier modelo de PyOD | `{"name": "pyod", "model": "ECOD"}` |

**Umbral**: tras `fit`, `threshold_` se fija en el cuantil `threshold_quantile` (0.95 por defecto)
de scores *fuera de muestra* de los datos normales: leave-one-out para kNN/LOF y validación cruzada
de 5 particiones para el resto. Así se evita el umbral optimista de detectores que sobreajustan el
entrenamiento (KDE, OCSVM, PCA…). Se puede desactivar con `det.calibration = "train"`.

## Modelos de Anomalib (con mapas de anomalía)

`AnomalibPipeline` envuelve cualquier modelo **one-class** de [Anomalib](https://github.com/open-edge-platform/anomalib)
(`patchcore`, `padim`, `efficient_ad`, `fastflow`, `stfpm`, `reverse_distillation`, `cflow`, `draem`…;
`imgad list` los muestra) con la misma API que los pipelines de features, y añade **mapas de anomalía
a nivel de píxel**:

```python
from imgad import AnomalibPipeline, evaluate_maps

pc = AnomalibPipeline("patchcore", backbone="wide_resnet50_2", layers=["layer2", "layer3"], image_size=256)
pc.fit(ds.train_paths)                              # solo normales
scores, maps = pc.score_and_maps(ds.test_paths)     # maps: lista de arrays (resolución del modelo)
evaluate_maps(ds.test_masks, maps)                  # pixel_auroc, pixel_aupr, pixel_f1_max, aupro
pc.predict_masks(maps)                              # máscaras binarias (map > pixel_threshold_)
pc.save("bottle_pc.joblib")                         # AnomalyPredictor.load / imgad predict lo cargan igual
```

- Anomalib aporta los modelos y el bucle de entrenamiento; su post-procesado (normalización y umbral
  aprendidos con un conjunto de validación etiquetado) está desactivado. Scores y mapas son los valores
  **brutos** del modelo (mayor = más anómalo).
- **Umbral**: igual que en los detectores, cuantil `threshold_quantile` (0.95) de los scores de normales
  *no vistas*: `calibration_split=0.2` reserva ese 20% de las imágenes de entrenamiento (no se usan para
  entrenar). `calibration_split=0` usa scores de entrenamiento (optimista, sobre todo en PatchCore).
  El umbral de píxel (`pixel_threshold_`) es el mismo cuantil del máximo de cada mapa normal reservado;
  con máscaras de validación se puede ajustar con `calibrate_pixel_threshold(images, masks)` (F1 de píxel).
  `calibrate_threshold` funciona igual que en los otros pipelines.
- Las imágenes se redimensionan a `image_size` **sin recorte**, así que los mapas cubren la imagen entera.
  `score_and_maps(..., size="original")` los devuelve al tamaño de cada imagen.
- El `.joblib` guarda la configuración **y los pesos** (incluido p. ej. el memory bank de PatchCore): no
  hace falta descargar nada para cargarlo.
- Kwargs del modelo: los de su constructor en anomalib (`backbone`, `layers`, `coreset_sampling_ratio`,
  `pre_trained`…). Para modelos que se entrenan varias épocas: `max_epochs` o `trainer_kwargs={"max_steps": ...}`.
  EfficientAD entrena con batch 1 y descarga imagenette.
- Solo modelos one-class; los zero/few-shot (WinCLIP, AnomalyDINO…) no están soportados.

Inferencia con mapas:

```python
predictor = AnomalyPredictor.load("bottle_pc.joblib")
res = predictor.predict("carpeta/", return_maps=True)   # r.anomaly_map (tamaño original), r.anomalous_area
predictor.save_maps(res, "mapas/")                      # <carpeta>_<nombre>_heatmap.png + _mask.png
predictor.save_results(res, "preds.csv")                # incluye las rutas de heatmap y mask
```

En los heatmaps la escala es fija por modelo (azul = normal típico, mitad de la escala = umbral de píxel,
rojo = muy por encima), de modo que las imágenes normales se ven azules en vez de reescaladas.

## Preprocesado

Lista de pasos aplicada antes del detector (se ajusta con los datos normales):
`"l2"`, `"standard"`, `"center"`, `"pca:0.95"` (varianza explicada) o `"pca:64"` (componentes),
`"pca_whiten:<n>"`, o cualquier transformador de sklearn.

## Datos

```python
data.mvtec_ad(root, "screw")                     # MVTec AD / formato MVTec (VisA convertido, etc.)
data.from_folders(train_normal, test_normal, [test_anom_1, test_anom_2])
data.one_vs_rest(root, normal_class="cat")       # AD semántica desde un dataset de clasificación
```

Cada uno devuelve un `AnomalyDataset` con `train_paths`, `test_paths`, `test_labels`, `test_types` y,
en MVTec, `test_masks` (ruta de la máscara ground-truth en `ground_truth/<tipo>/<nombre>_mask.png`, o
`None` para imágenes normales).

## Evaluación y benchmarks

`evaluate_scores` devuelve AUROC, AUPR, F1-max y FPR@95%TPR (y F1/TPR/FPR en el umbral si se pasan
predicciones). `per_type_auroc` desglosa por tipo de defecto.

`evaluate_maps(masks, maps)` da las métricas de localización: pixel-AUROC, pixel-AUPR, pixel-F1-max y
AUPRO (FPR ≤ 0.3, la métrica estándar de MVTec AD). Las máscaras se reescalan a la resolución de cada mapa.

`benchmark(extractors, detectors, datasets, preprocessings, anomalib_models=[...])` evalúa la rejilla
completa extrayendo las features una sola vez por (extractor, dataset), y además cada modelo de anomalib
por dataset (con métricas de píxel si hay máscaras). En YAML, clave `anomalib:` (ver
`configs/mvtec_benchmark.yaml`).

## Análisis estadístico: repeticiones, intervalos de confianza y tests

**Repeticiones.** `benchmark(..., seeds=[0, 1, 2, 3, 4], train_subsample=0.8)` ejecuta cada configuración
una vez por semilla (columna `seed`). La semilla fija la `random_state` de los detectores (salvo que el spec
la fije), los folds de calibración, la semilla de anomalib y el **subconjunto de entrenamiento**
(`train_subsample`: fracción si es float, número de imágenes si es int). Sin submuestreo, los detectores
deterministas (kNN, Mahalanobis…) dan el mismo resultado con cualquier semilla. Las features se extraen
una sola vez y se reutilizan en todas las semillas.

**Intervalos de confianza.** Hay dos fuentes de incertidumbre y se miden por separado:

| Fuente | Cómo | Dónde |
|---|---|---|
| Muestra de test | bootstrap estratificado de las imágenes de test (`bootstrap=1000`) | `auroc_ci_low/high`, `aupr_ci_low/high` en cada fila |
| Entrenamiento | repeticiones con semillas + `train_subsample` | `stats.summarize_seeds(rows)`: media, desviación típica e IC t de la media |

**Tests estadísticos** (`imgad.stats`):

| Pregunta | Test | Función |
|---|---|---|
| ¿A tiene mayor AUROC que B **en este dataset**? | DeLong (AUROCs correladas, mismas imágenes) | `delong_test(y, sa, sb)`, `delong_vs_baseline(rows, base)` |
| Igual para AUPR, F1-max u otra métrica | bootstrap pareado | `paired_bootstrap_test(y, sa, sb, metric="aupr")` |
| ¿Hay diferencias entre configuraciones **a lo largo de los datasets**? | Friedman (+ corrección de Iman-Davenport) | `compare_configs(rows)` |
| ¿Qué pares difieren? | Nemenyi (diferencia crítica de rangos) y Wilcoxon de rangos con signo + Holm | `compare_configs(rows)["pairs"]`, `plot_cd_diagram` |

Para comparar entre datasets, las semillas se promedian primero en cada dataset, y cada dataset (p. ej.
cada categoría de MVTec) cuenta como un bloque, siguiendo a Demšar (2006). Wilcoxon-Holm suele tener más
potencia con pocas configuraciones, mientras que Nemenyi es el post-hoc clásico tras Friedman y el que se
dibuja en el diagrama CD. DeLong necesita los scores por imagen (`keep_scores=True`).

```python
from imgad import benchmark, stats

rows = benchmark(extractors, detectors, datasets, seeds=range(5), train_subsample=0.8,
                 bootstrap=1000, keep_scores=True, anomalib_models=["patchcore"])
stats.summarize_seeds(rows)                          # media ± std por (dataset, config)
res = stats.compare_configs(rows, metric="auroc")    # también aupr, pixel_auroc, aupro, fpr@95tpr…
print(stats.report_markdown(res))                    # rangos medios, Friedman, CD, pares con Holm
stats.plot_cd_diagram(res, "cd.png")
stats.delong_vs_baseline(rows, baseline=res["configs"][0])
```

Desde la CLI:

```bash
imgad benchmark --config configs/mvtec_benchmark.yaml --seeds 0 1 2 3 4 --train-subsample 0.8 \
                --bootstrap 1000 --out results.csv --scores-out scores.joblib --stats
imgad stats --results results.csv --metric auroc --cd-plot cd.png --out informe.md
imgad stats --scores scores.joblib --baseline "openai/clip-vit-base-patch16 | l2 | knn(k=1)"   # + DeLong
```

## CLI

```bash
imgad list
imgad fit --model openai/clip-vit-base-patch16 --detector knn -p k=1 --prep l2 \
          --train data/bottle/train/good --out bottle.joblib

# inferencia: imágenes y/o carpetas; salida por pantalla, .csv o .json
imgad predict --model bottle.joblib nueva.png data/bottle/test --out preds.csv
imgad predict --model bottle.joblib carpeta/ --only-anomalies --threshold 0.02 --device cpu

# reajustar el umbral con validación etiquetada (f1 | youden | fpr | tpr)
imgad calibrate --model bottle.joblib --normal val/good --anomalous val/defectos --method f1 --out bottle_cal.joblib
imgad info --model bottle_cal.joblib         # configuración, umbral y metadatos

imgad benchmark --config configs/mvtec_benchmark.yaml --out results.csv

# modelos de anomalib: -p son los kwargs del modelo; predict guarda heatmaps y máscaras
imgad fit --anomalib patchcore -p backbone=wide_resnet50_2 "layers=[layer2,layer3]" \
          --image-size 256 --train data/bottle/train/good --out bottle_pc.joblib
imgad predict --model bottle_pc.joblib data/bottle/test --maps-dir mapas/ --out preds.csv
```

## Extender

Registrar un detector o extractor propio lo hace disponible por nombre en Python, YAML y CLI
(ver `examples/extending.py`):

```python
from imgad import BaseDetector, register_detector

@register_detector("mi_detector")
class MiDetector(BaseDetector):
    def _fit(self, X): ...
    def _score(self, X): ...   # mayor = más anómalo
```

## Estructura

```
src/imgad/
  features/      base.py (interfaz, registro, caché) · huggingface.py · open_clip.py
  detectors/     base.py (interfaz, registro, calibración) · classical.py
  anomalib_backend.py (AnomalibPipeline) · maps.py (máscaras, heatmaps)
  preprocessing.py · pipeline.py · inference.py · data.py · evaluation.py · stats.py · cli.py
configs/         benchmarks YAML de ejemplo
examples/        quickstart.py · inference.py · compare_models.py · extending.py · anomalib_models.py
tests/           pytest (sin conexión: CLIP diminuto generado localmente; ResNet-18 sin preentrenar para anomalib)
```

## Próximos pasos posibles

- Detección **zero-/few-shot** con prompts de texto (WinCLIP, AnomalyCLIP) aprovechando el encoder de texto.
- PatchCore nativo sobre features de parche de modelos fundacionales (CLIP/DINOv2) para comparar con el de anomalib (CNN).
- Aumentado de datos en test (TTA) y agregación multi-escala.
- Barrido few-shot (`train_subsample` = 1, 2, 4, 8… imágenes) reutilizando las semillas y los tests de `imgad.stats`.

## Licencia

Distribuido bajo la licencia [Apache 2.0](LICENSE).
