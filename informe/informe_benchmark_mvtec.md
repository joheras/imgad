# Benchmark de detección de anomalías: modelos fundacionales + detectores clásicos

*Framework `imgad` · MVTec AD (4 categorías) · 16 de octubre de 2026*

## Resumen

**Pregunta:** con un extractor de características preentrenado y un detector clásico entrenado solo con imágenes normales, ¿qué combinación detecta mejor las anomalías a nivel de imagen?

- **Mejor combinación global:** SigLIP B/16 + Ensemble kNN+Mahal., con AUROC medio **0.974** en las 4 categorías.
- **Mejor extractor** (con su mejor detector): SigLIP B/16 0.974, DINOv2-B · capas 6/9/12 0.966, DINOv2-B 0.959, CLIP ViT-B/16 0.933, CLIP ViT-B/16 · capas 6/9/12 0.918.
- **Mejor detector** (media sobre extractores): Mahalanobis 0.946, Ensemble kNN+Mahal. 0.945, PCA (90% var.) 0.944, kNN (k=1) 0.926; los peores: OCSVM 0.828, Centroide 0.836.
- **Categoría más difícil:** screw (mejor AUROC 0.927, SigLIP B/16 + Ensemble kNN+Mahal.).
- Eligiendo la mejor combinación por categoría el AUROC medio sería 0.978 (cota optimista: la selección se hace sobre test).


**Conclusiones principales**

1. **SigLIP B/16 es el mejor extractor** en las 4 categorías (empata en *bottle*, donde casi todo da 1,000), y también es el mejor en media sobre todos los detectores (0,936 frente a 0,911 de DINOv2-B y 0,885 de CLIP).
2. **Los detectores gaussianos son los mejores con cualquier extractor.** Mahalanobis, PCA y el ensemble kNN+Mahalanobis ocupan los tres primeros puestos con 4 de los 5 extractores; con SigLIP, kNN (k=1) adelanta a PCA por 0,001. Isolation Forest, One-Class SVM y la distancia al centroide son los tres últimos con los 5 extractores.
3. **El detector importa casi tanto como el extractor.** Con CLIP, el AUROC medio va de 0,830 (OCSVM) a 0,933 (PCA) según el detector. Con Mahalanobis, cambiar de extractor lo mueve entre 0,916 y 0,968.
4. **Las capas intermedias con pooling global no ayudan.** Con DINOv2 apenas cambian el resultado (+0,001 de media) y con CLIP lo empeoran (−0,016), aunque mejoran *hazelnut*.
5. **Recomendación por defecto:** SigLIP + Mahalanobis (0,968 de media), más simple que el ensemble y solo 0,006 por debajo.

## Configuración experimental

| | |
|---|---|
| Datos | MVTec AD (copia `foersben/mvtec-ad` en HF Hub): bottle, hazelnut, screw, carpet |
| Entrenamiento | solo imágenes normales (`train/good`): bottle 209, hazelnut 391, screw 320, carpet 280 |
| Test | normales / anómalas: bottle 20/63, hazelnut 40/70, screw 41/119, carpet 28/89 |
| Extractores | CLIP ViT-B/16; CLIP ViT-B/16 · capas 6/9/12; SigLIP B/16; DINOv2-B; DINOv2-B · capas 6/9/12 |
| Preprocesado | normalización L2 |
| Detectores | kNN (k=1), kNN (k=5), Mahalanobis, Centroide, k-means (8), LOF, GMM (4), OCSVM, Isolation Forest, PCA (90% var.), Ensemble kNN+Mahal. |
| Métrica principal | AUROC a nivel de imagen (también AUPR, F1-max, FPR@95%TPR) |
| Hardware | CPU (2 núcleos), sin GPU |

Pooling: CLIP y SigLIP usan el embedding proyectado (`get_image_features`); DINOv2-B concatena token CLS y media de parches; las variantes «capas 6/9/12» concatenan la media de tokens de parche de esas capas del ViT (dimensión 3×768). Umbral de decisión: cuantil 95% de scores normales fuera de muestra.

## AUROC medio (4 categorías): extractor × detector

| Detector | CLIP ViT-B/16 | CLIP ViT-B/16 · capas 6/9/12 | SigLIP B/16 | DINOv2-B | DINOv2-B · capas 6/9/12 |
|---|---:|---:|---:|---:|---:|
| Mahalanobis | 0.924 | 0.916 | 0.968 | 0.959 | 0.966 |
| Ensemble kNN+Mahal. | 0.921 | 0.911 | **0.974** | 0.959 | 0.960 |
| PCA (90% var.) | 0.933 | 0.918 | 0.964 | 0.956 | 0.951 |
| kNN (k=1) | 0.902 | 0.877 | 0.965 | 0.949 | 0.938 |
| kNN (k=5) | 0.898 | 0.864 | 0.955 | 0.935 | 0.929 |
| LOF | 0.897 | 0.818 | 0.950 | 0.911 | 0.923 |
| k-means (8) | 0.873 | 0.813 | 0.946 | 0.901 | 0.903 |
| GMM (4) | 0.869 | 0.803 | 0.926 | 0.896 | 0.897 |
| Isolation Forest | 0.846 | 0.777 | 0.904 | 0.864 | 0.808 |
| Centroide | 0.844 | 0.787 | 0.867 | 0.844 | 0.837 |
| OCSVM | 0.830 | 0.770 | 0.881 | 0.843 | 0.818 |
| *Media del extractor* | *0.885* | *0.841* | *0.936* | *0.911* | *0.903* |

## Resultados por categoría

AUROC de cada extractor con su mejor detector en esa categoría (detector entre paréntesis).

| Extractor | bottle | hazelnut | screw | carpet |
|---|---:|---:|---:|---:|
| CLIP ViT-B/16 | 1.000 (Mahalanobis) | 0.949 (PCA (90% var.)) | 0.814 (PCA (90% var.)) | 0.976 (Mahalanobis) |
| CLIP ViT-B/16 · capas 6/9/12 | 1.000 (Mahalanobis) | 0.971 (PCA (90% var.)) | 0.775 (Mahalanobis) | 0.929 (Mahalanobis) |
| SigLIP B/16 | 1.000 (kNN (k=1)) | 0.998 (PCA (90% var.)) | 0.927 (Ensemble kNN+Mahal.) | 0.986 (Mahalanobis) |
| DINOv2-B | 1.000 (kNN (k=1)) | 0.975 (Mahalanobis) | 0.900 (kNN (k=1)) | 0.984 (Mahalanobis) |
| DINOv2-B · capas 6/9/12 | 0.998 (Mahalanobis) | 0.979 (Mahalanobis) | 0.905 (Mahalanobis) | 0.980 (Mahalanobis) |

## Análisis por tipo de defecto

Mejor combinación global (SigLIP B/16 + ensemble kNN+Mahalanobis). AUROC de cada tipo de defecto frente a todas las imágenes normales de test de su categoría.

| Categoría | Tipo de defecto | Nº test | AUROC |
|---|---|---:|---:|
| bottle | broken_large | 20 | 1.000 |
| bottle | broken_small | 22 | 1.000 |
| bottle | contamination | 21 | 1.000 |
| hazelnut | crack | 18 | 1.000 |
| hazelnut | cut | 17 | 0.991 |
| hazelnut | hole | 18 | 1.000 |
| hazelnut | print | 17 | 0.957 |
| screw | manipulated_front | 24 | 0.986 |
| screw | scratch_head | 24 | 0.933 |
| screw | scratch_neck | 25 | 0.969 |
| screw | thread_side | 23 | 0.874 |
| screw | thread_top | 23 | 0.870 |
| carpet | color | 19 | 0.998 |
| carpet | cut | 17 | 0.996 |
| carpet | hole | 17 | 0.996 |
| carpet | metal_contamination | 17 | 0.966 |
| carpet | thread | 19 | 0.955 |

Los fallos se concentran en defectos **pequeños y locales**: las roscas de *screw* (`thread_side` 0,874 y `thread_top` 0,870), `thread` y `metal_contamination` en *carpet*, y `print` en *hazelnut*. Los defectos grandes o que cambian el aspecto global (roturas, agujeros, cambios de color) se detectan casi perfectamente. Es lo esperable con un único embedding por imagen: un defecto que ocupa una fracción mínima de la imagen apenas desplaza el vector global. En *screw* se suma además que los tornillos aparecen en orientaciones arbitrarias, así que buena parte de la variación del embedding se debe a la pose y no a los defectos.

## Efecto de usar capas intermedias

Diferencia de AUROC (capas intermedias − última capa), con el mejor detector de cada variante.

| Backbone | bottle | hazelnut | screw | carpet | Media |
|---|---:|---:|---:|---:|---:|
| CLIP ViT-B/16 | +0.000 | +0.023 | -0.039 | -0.047 | -0.016 |
| DINOv2-B | -0.002 | +0.004 | +0.005 | -0.004 | +0.001 |

## Punto de operación (umbral automático)

Para la mejor combinación global (SigLIP B/16 + Ensemble kNN+Mahal.) con el umbral calibrado automáticamente (cuantil 95% de scores normales fuera de muestra, sin usar anomalías):

| Categoría | AUROC | AUPR | F1-max | FPR@95%TPR | TPR@umbral | FPR@umbral | F1@umbral |
|---|---:|---:|---:|---:|---:|---:|---:|
| bottle | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 |
| hazelnut | 0.987 | 0.993 | 0.971 | 0.050 | 0.886 | 0.050 | 0.925 |
| screw | 0.927 | 0.973 | 0.924 | 0.390 | 0.639 | 0.024 | 0.776 |
| carpet | 0.982 | 0.995 | 0.972 | 0.071 | 1.000 | 0.464 | 0.932 |

## Discusión

**Por qué ganan los detectores gaussianos.** Con embeddings de modelos fundacionales normalizados en L2, las imágenes normales de una categoría forman una nube compacta y aproximadamente unimodal. Mahalanobis (con covarianza Ledoit-Wolf) y la reconstrucción PCA modelan explícitamente las direcciones en las que varían los datos normales. Así, una desviación pequeña en una dirección en la que lo normal casi no varía cuenta mucho más que la misma desviación en una dirección de alta varianza. kNN y la distancia al centroide tratan todas las direcciones por igual. Isolation Forest hace cortes aleatorios eje a eje, poco adecuados para vectores densos de 512–2304 dimensiones en los que la información está repartida. El One-Class SVM con kernel RBF y `gamma="scale"` es sensible a la escala y no se ajustó.

**Capas intermedias.** La literatura muestra que las capas intermedias ayudan con defectos locales, pero ese resultado viene de métodos que comparan **parches** individuales (p. ej. PatchCore). Aquí se promedian los tokens de parche de cada capa en un único vector, de modo que la información local se diluye igualmente. Además, la dimensión sube a 2304 con solo 209–391 imágenes de entrenamiento, lo que perjudica a los detectores que estiman densidades o covarianzas. Con CLIP las capas intermedias mejoran *hazelnut* (+0,023) pero empeoran *carpet* (−0,047) y *screw* (−0,039).

**Umbral automático.** Para la mejor combinación, el umbral fijado en el cuantil 95% de los scores normales de entrenamiento (sin usar ninguna anomalía) da un FPR en test de 0–5% en *bottle*, *hazelnut* y *screw*, como se esperaba. En *carpet* el FPR sube al 46% (13 de 28 normales de test): los scores de las imágenes normales de test son más altos que los de entrenamiento. El ranking sigue siendo bueno (AUROC 0,982), por lo que es un problema de calibración y no de separabilidad. En producción conviene recalibrar el umbral con un pequeño conjunto de validación (`imgad calibrate --method fpr`).

**Contexto.** Los métodos específicos a nivel de parche (p. ej. PatchCore, Roth et al., CVPR 2022) obtienen AUROC de imagen en torno a 0,99 en MVTec AD. El enfoque de embedding global + detector clásico es más simple, entrena en segundos y permite cambiar de backbone trivialmente, pero su techo está en los defectos pequeños. El 0,974 medio obtenido aquí (en 4 categorías) es una línea base sólida.

## Dimensión de las características

| Extractor | Dimensión |
|---|---:|
| CLIP ViT-B/16 | 512 |
| CLIP ViT-B/16 · capas 6/9/12 | 2304 |
| SigLIP B/16 | 768 |
| DINOv2-B | 1536 |
| DINOv2-B · capas 6/9/12 | 2304 |

En la CPU de 2 núcleos usada, la extracción con un ViT-B/16 a 224 px fue de unas 2,5–4 imágenes/s (incluida la decodificación de PNG de 900–1024 px); con GPU es del orden de cientos de imágenes/s. Ajustar y puntuar cualquier detector tarda menos de unos segundos por categoría.

## Limitaciones

- **Solo 4 de las 15 categorías** de MVTec AD. *bottle* resulta trivial y *screw* es de las más difíciles; las conclusiones pueden cambiar con el conjunto completo.
- **Conjuntos de test pequeños** (20–41 imágenes normales por categoría): diferencias de AUROC de ±0,01–0,02 entre combinaciones no son significativas.
- **Sin ajuste de hiperparámetros:** todos los detectores usan valores por defecto. Elegir la mejor combinación mirando el test es optimista; para una comparación rigurosa habría que seleccionar en validación.
- **Solo métricas a nivel de imagen**: no se evalúa la localización (pixel-AUROC).
- **Solo modelos base (ViT-B)** por la falta de GPU; los modelos *large* podrían cambiar el orden.

## Siguientes pasos propuestos

1. Completar las 15 categorías de MVTec AD y añadir VisA, con GPU.
2. Probar backbones mayores: SigLIP-L / SigLIP2, DINOv2-L, CLIP ViT-L/14.
3. Implementar un extractor **a nivel de parche** con memory bank (estilo PatchCore) para atacar los defectos locales y obtener mapas de anomalía.
4. Separar validación de test para seleccionar detector e hiperparámetros sin sesgo.
5. Detección zero-shot con prompts de texto (estilo WinCLIP), aprovechando el encoder de texto de CLIP/SigLIP.

## Reproducibilidad

Archivos en esta carpeta:

- `resultados_mvtec.csv`: las 220 filas (5 extractores × 11 detectores × 4 categorías) con AUROC, AUPR, F1-max, FPR@95%TPR, métricas en el umbral y dimensión.
- `config_mvtec4.yaml`: configuración exacta del benchmark (cambiar `root` a la ruta local de MVTec AD).

```bash
# datos: copia de MVTec AD en el Hub (estructura original)
huggingface-cli download foersben/mvtec-ad --repo-type dataset --local-dir mvtec \
    --include "bottle/*" "hazelnut/*" "screw/*" "carpet/*"

imgad benchmark --config informe/config_mvtec4.yaml --out resultados.csv
```

Entorno: `imgad` 0.1.0, torch 2.14, transformers 5.20, scikit-learn 1.9, Python 3.13; CPU de 2 núcleos sin GPU.

## Anexo: AUROC por categoría, extractor y detector

#### bottle

| Detector | CLIP ViT-B/16 | CLIP ViT-B/16 · capas 6/9/12 | SigLIP B/16 | DINOv2-B | DINOv2-B · capas 6/9/12 |
|---|---:|---:|---:|---:|---:|
| kNN (k=1) | 0.994 | 0.995 | 1.000 | 1.000 | 0.989 |
| kNN (k=5) | 0.992 | 0.992 | 1.000 | 0.998 | 0.990 |
| Mahalanobis | 1.000 | 1.000 | 1.000 | 1.000 | 0.998 |
| Centroide | 0.993 | 0.983 | 0.992 | 0.991 | 0.976 |
| k-means (8) | 0.985 | 0.979 | 1.000 | 0.998 | 0.984 |
| LOF | 0.992 | 0.994 | 1.000 | 0.997 | 0.988 |
| GMM (4) | 0.983 | 0.994 | 1.000 | 0.990 | 0.980 |
| OCSVM | 0.985 | 0.981 | 0.986 | 0.994 | 0.973 |
| Isolation Forest | 0.991 | 0.993 | 0.997 | 0.990 | 0.986 |
| PCA (90% var.) | 1.000 | 1.000 | 1.000 | 1.000 | 0.995 |
| Ensemble kNN+Mahal. | 1.000 | 1.000 | 1.000 | 1.000 | 0.997 |

#### hazelnut

| Detector | CLIP ViT-B/16 | CLIP ViT-B/16 · capas 6/9/12 | SigLIP B/16 | DINOv2-B | DINOv2-B · capas 6/9/12 |
|---|---:|---:|---:|---:|---:|
| kNN (k=1) | 0.884 | 0.891 | 0.966 | 0.917 | 0.907 |
| kNN (k=5) | 0.898 | 0.866 | 0.966 | 0.919 | 0.909 |
| Mahalanobis | 0.942 | 0.961 | 0.996 | 0.975 | 0.979 |
| Centroide | 0.871 | 0.790 | 0.878 | 0.885 | 0.846 |
| k-means (8) | 0.890 | 0.849 | 0.959 | 0.921 | 0.919 |
| LOF | 0.905 | 0.860 | 0.976 | 0.918 | 0.932 |
| GMM (4) | 0.893 | 0.841 | 0.949 | 0.924 | 0.920 |
| OCSVM | 0.878 | 0.792 | 0.930 | 0.914 | 0.879 |
| Isolation Forest | 0.862 | 0.863 | 0.919 | 0.902 | 0.790 |
| PCA (90% var.) | 0.949 | 0.971 | 0.998 | 0.965 | 0.974 |
| Ensemble kNN+Mahal. | 0.927 | 0.947 | 0.987 | 0.959 | 0.963 |

#### screw

| Detector | CLIP ViT-B/16 | CLIP ViT-B/16 · capas 6/9/12 | SigLIP B/16 | DINOv2-B | DINOv2-B · capas 6/9/12 |
|---|---:|---:|---:|---:|---:|
| kNN (k=1) | 0.754 | 0.700 | 0.925 | 0.900 | 0.881 |
| kNN (k=5) | 0.729 | 0.679 | 0.896 | 0.846 | 0.847 |
| Mahalanobis | 0.776 | 0.775 | 0.889 | 0.875 | 0.905 |
| Centroide | 0.541 | 0.495 | 0.679 | 0.529 | 0.575 |
| k-means (8) | 0.647 | 0.526 | 0.869 | 0.716 | 0.757 |
| LOF | 0.715 | 0.512 | 0.864 | 0.755 | 0.809 |
| GMM (4) | 0.629 | 0.479 | 0.806 | 0.696 | 0.727 |
| OCSVM | 0.484 | 0.448 | 0.670 | 0.504 | 0.484 |
| Isolation Forest | 0.570 | 0.458 | 0.753 | 0.597 | 0.637 |
| PCA (90% var.) | 0.814 | 0.772 | 0.882 | 0.881 | 0.865 |
| Ensemble kNN+Mahal. | 0.780 | 0.769 | 0.927 | 0.895 | 0.904 |

#### carpet

| Detector | CLIP ViT-B/16 | CLIP ViT-B/16 · capas 6/9/12 | SigLIP B/16 | DINOv2-B | DINOv2-B · capas 6/9/12 |
|---|---:|---:|---:|---:|---:|
| kNN (k=1) | 0.974 | 0.921 | 0.971 | 0.979 | 0.973 |
| kNN (k=5) | 0.973 | 0.917 | 0.959 | 0.978 | 0.971 |
| Mahalanobis | 0.976 | 0.929 | 0.986 | 0.984 | 0.980 |
| Centroide | 0.972 | 0.882 | 0.920 | 0.970 | 0.951 |
| k-means (8) | 0.970 | 0.899 | 0.955 | 0.970 | 0.953 |
| LOF | 0.976 | 0.906 | 0.961 | 0.974 | 0.965 |
| GMM (4) | 0.970 | 0.898 | 0.948 | 0.973 | 0.959 |
| OCSVM | 0.974 | 0.857 | 0.938 | 0.961 | 0.936 |
| Isolation Forest | 0.962 | 0.796 | 0.948 | 0.964 | 0.819 |
| PCA (90% var.) | 0.970 | 0.928 | 0.975 | 0.976 | 0.971 |
| Ensemble kNN+Mahal. | 0.976 | 0.929 | 0.982 | 0.982 | 0.978 |
