# BenchReAD: notas para la comparación con imgad

Lian, C., Zhou, H.-Y., Hu, Z., Qin, J. *BenchReAD: A systematic benchmark for retinal anomaly detection.*
MICCAI 2025, LNCS 15961, pp. 35–45, doi:10.1007/978-3-032-04937-7_4; arXiv:2507.10492 (v1, 14-07-2025) [1, 2, 3].
Código: <https://github.com/DopamineLcy/BenchReAD> (AGPL-3.0; último push 14-07-2025) [4].

Todo lo que sigue sale del texto del artículo (versión HTML y PDF de arXiv) y del código del repositorio
(cuadernos `fundus/*.ipynb`, `OCT/*.ipynb` y `NFM-DRA/`), leídos el 23-10-2026. Donde el artículo y el código
no dicen algo, se indica.

## 1. Conjuntos de datos y modalidades

Dos *benchmarks* independientes: retinografía en color (fondo de ojo) y OCT [1, §2.1]. En ambos, el
entrenamiento y la validación salen de unos conjuntos y el test de **otros conjuntos distintos** (cambio de
dominio), y el test contiene anomalías **vistas** (presentes en entrenamiento) y **no vistas**.

### Fondo de ojo

| Papel | Conjunto | Normales | Anómalas | Notas |
|---|---|---:|---:|---|
| Entrenamiento | EDDFS [5] + BRSET [6] | 23.361 | 16.388 | anomalías retenidas: RD, DMAE, glaucoma, miopía (MYA) |
| Validación | EDDFS + BRSET | 100 | 100 | 50 + 50 de cada fuente |
| Test | RIADD (= RFMiD) [7] | 669 | 1.634 | vistas: DR, AMD, ODC, MYA, DN; no vistas: MH, BRVO, TSLN, CSR, CRVO, ODP, ODE, RS, CRS, RPEC |
| Test | JSIEC [8] | 38 | 686 | vistas: DR, LOC, PM (MYA); no vistas: BRVO, CRVO, RRD, MAC, ERM, MH, RP, YWSF, LS, BF sin/con RDP |

Se excluyen del test las categorías con menos de 20 imágenes [1, §2.1].

### OCT

| Papel | Conjunto | Normales | Anómalas | Notas |
|---|---|---:|---:|---|
| Entrenamiento | OCT 2017 (Kermany) [9] | 25.840 | 53.702 | CNV, DME, drusas |
| Validación | OCT 2017 | 50 | 150 | 50 por clase |
| Test | OCT 2017 (val + test oficiales) | 250 | 750 | 250 por clase; casi saturado |
| Test | OCTDL [10] | 332 | 1.732 | vistas: AMD, DME; no vistas: ERM, RAO, RVO, VID |
| Test | OCTID [11] | 206 | 336 | vistas: AMD, DR; no vistas: CSR, MH |

## 2. Cómo se definen clases y particiones (según el código)

**Normal / anómalo.**
- EDDFS: `normal == 1` es normal; anómalas = `normal == 0` **sin** RVO, LS (láser), hipertensión ni «others»
  (quedan DR, AMD, glaucoma, miopía; puede haber varias etiquetas). 15.000 normales y 10.691 anómalas.
- BRSET: normal = las 13 columnas de enfermedad (`diabetic_retinopathy` … `other`, posiciones 20–32 del CSV)
  a 0; anómalas = alguna a 1 pero sin edema macular, cicatriz, nevus, oclusión vascular, retinopatía
  hipertensiva, hemorragia, desprendimiento ni «other».
- RIADD: normal = `Disease_Risk == 0` en las tres particiones oficiales de RFMiD (entrenamiento, validación
  y test se usan **todas** como test); anómala = exactamente una etiqueta, y que esté entre las 15 retenidas.
- JSIEC: carpetas `0.0.Normal` (normal) y 16 carpetas de enfermedad con ≥ 20 imágenes; DR1–DR3 se agrupan en DR.

**Particiones (fondo).** Con `seed = 0` y `DataFrame.sample(random_state=0)` de pandas, por fuente:
1. 50 normales y 50 anómalas → validación;
2. del resto, 1/3 de las normales y 1/3 de las anómalas → *labeled*; 2/3 → *unlabeled* (sin etiqueta).

No hay partición por paciente en el fondo de ojo (EDDFS y BRSET se muestrean por imagen). En OCT 2017 sí se
parte por paciente, pero con `np.random.choice` sobre una lista construida desde un `set` y `os.listdir`, así que
**la partición OCT no es reproducible** (depende del orden del sistema de ficheros y de `PYTHONHASHSEED`).
La partición del fondo sí es reproducible: depende solo de los CSV de etiquetas.

**Preprocesado.** EDDFS, BRSET y RIADD se recortan al campo de visión (componente conexa mayor de
`suma RGB > 30`, recorte cuadrado centrado en su centroide, margen 5 px) y se redimensionan a 512 × 512.
**JSIEC no se recorta** (se lee tal cual de `1000images/`). Los métodos redimensionan después a 224 px
(PatchCore: WideResNet-50, capas 2–3, *coreset* 10 %, 1-NN).

## 3. Escenarios (niveles de supervisión)

El conjunto de entrenamiento se divide en 1/3 etiquetado y 2/3 sin etiquetar [1, §2.2]:

| Escenario | Datos de entrenamiento | Métodos evaluados |
|---|---|---|
| No supervisado | todo (normales + anómalas + sin etiquetar), sin etiquetas | SoftPatch (NeurIPS 2022) |
| **One-class** | solo normales etiquetadas (1/3 de las normales) | EDC (TMI 2023), SimpleNet (CVPR 2023), PatchCore (CVPR 2022) |
| Semisupervisado | normales etiquetadas + sin etiquetar | DDAD-ASR (MedIA 2023) |
| Supervisado (*open-set*) | normales y anómalas etiquetadas | DRA (CVPR 2022), **NFM-DRA** (propuesto) |

NFM-DRA = media de la puntuación de DRA (reescalada min–max con la **validación**, que contiene anomalías) y una
puntuación PatchCore («Normal Feature Memory») normalizada min–max **sobre el propio conjunto de test**.
En el código publicado el `Dataset` de entrenamiento de NFM devuelve `__len__ = 50`, de modo que la memoria
se construye con solo 50 imágenes normales. No hay escenarios *few-shot* ni *zero-shot*.

Selección: búsqueda en rejilla de tasa de aprendizaje y épocas «alrededor de los valores por defecto»; se
evalúa en test el mejor *checkpoint* en **validación** (que tiene 100 anomalías) [1, §3.2]. Una única RTX 3090.

## 4. Métricas

- **AUROC** normal vs. todas las anomalías por conjunto de test, con IC 95 % (método no indicado) [1, Fig. 2].
- **AUROC** normal vs. cada categoría anómala, con IC 95 % [1, Fig. 3, solo en gráfico].
- **F1, especificidad y sensibilidad** con un umbral por método y conjunto (la especificidad es la misma en
  todas las filas de un conjunto), normal vs. cada categoría y su media [1, Tablas 1–2]. La fila «Average» es la
  media de las categorías **y** de la fila agrupada «All abnormal» (exacto en RIADD; en JSIEC la fila agrupada no se
  imprime; la media de sus categorías no cuadra con la publicada y una fila agrupada estimada a partir de SPC y SEN sí,
  así que es una deducción). **Ni el artículo ni el
  código dicen cómo se fija el umbral**; los valores (p. ej. especificidad 0,0 % para EDC) sugieren un umbral
  fijo sobre puntuaciones min–max, no uno calibrado.

## 5. Resultados publicados

AUROC (%) normal vs. todas las anomalías, IC 95 % entre paréntesis [1, Fig. 2, leída del PDF]:

| Método | Supervisión | RIADD | JSIEC | OCT 2017 | OCTDL | OCTID |
|---|---|---|---|---|---|---|
| SoftPatch | no sup. | 75,4 (73,5–77,3) | 90,9 (86,7–94,7) | 95,3 (93,8–96,6) | 88,8 (87,2–90,3) | 96,2 (94,8–97,5) |
| EDC | one-class | 58,3 (55,6–60,9) | 80,0 (72,6–86,4) | 95,1 (93,8–96,3) | 80,6 (78,0–83,1) | 92,4 (90,1–94,6) |
| SimpleNet | one-class | 68,7 (66,4–70,7) | 76,6 (67,7–85,0) | 99,0 (98,5–99,4) | 87,1 (85,0–88,9) | 92,1 (89,7–94,2) |
| PatchCore | one-class | 77,0 (75,1–78,8) | 92,3 (88,1–95,8) | 99,7 (99,4–99,9) | 94,4 (93,4–95,3) | 97,3 (95,9–98,5) |
| DDAD-ASR | semisup. | 64,7 (62,1–67,2) | 68,2 (56,7–77,8) | 88,2 (86,1–90,3) | 62,8 (59,6–66,1) | 69,3 (65,0–73,4) |
| DRA | supervisado | 85,6 (84,1–87,1) | 89,2 (85,9–92,1) | 100,0 (100,0–100,0) | 97,7 (97,2–98,2) | 98,6 (97,6–99,5) |
| NFM-DRA | supervisado | **89,0** (87,7–90,2) | **96,7** (94,8–98,2) | 100,0 (100,0–100,0) | 97,7 (97,2–98,2) | 98,8 (97,8–99,6) |

F1 / especificidad / sensibilidad medias por categoría (%) en fondo de ojo [1, Tabla 1]:

| Método | RIADD F1 | SPC | SEN | JSIEC F1 | SPC | SEN |
|---|---:|---:|---:|---:|---:|---:|
| SoftPatch | 25,6 | 5,8 | 97,6 | 71,2 | 10,5 | 99,3 |
| EDC | 25,2 | 0,0 | 100,0 | 69,3 | 0,0 | 99,9 |
| SimpleNet | 25,3 | 0,9 | 99,7 | 72,0 | 13,2 | 99,9 |
| PatchCore | 26,3 | 13,6 | 95,2 | 73,5 | 21,1 | 99,5 |
| DDAD-ASR | 22,1 | 88,6 | 24,0 | 76,1 | 47,4 | 91,9 |
| DRA | 53,1 | 90,3 | 68,8 | 83,9 | 94,7 | 80,0 |
| NFM-DRA | 61,1 | 99,3 | 52,4 | 85,6 | 100,0 | 78,6 |

RIADD normal vs. todas las anomalías (fila «All abnormal», F1/SPC/SEN): PatchCore 82,4/13,6/94,9; DRA
79,3/90,3/68,3; NFM-DRA 70,0/99,3/54,0. Las cifras por categoría están en las Tablas 1–2 (y en la Fig. 3 para el
AUROC); `compare_benchread.py` las copia para RIADD y JSIEC.

Conclusiones del artículo: DRA (supervisado) es el mejor en conjunto pero cae en algunas anomalías no vistas
(MH, ODE, RS, CRS en RIADD; CRVO y fondos borrosos en JSIEC), donde PatchCore (one-class) es más robusto; OCT 2017
está saturado; las métricas con umbral fluctúan mucho más que el AUROC.

## 6. Licencias y obtención de los datos

BenchReAD **no redistribuye imágenes ni listas de partición**: hay que descargar cada conjunto y regenerar las
particiones con sus cuadernos (o con `prepare_benchread.py`, que las reproduce para el fondo de ojo).

| Conjunto | Fuente | Tamaño | Licencia / acceso |
|---|---|---|---|
| EDDFS (28.877 imágenes) | enlaces Google Drive / Mega / Baidu del README de <https://github.com/xia-xx-cv/EDDFS_dataset> [5]; etiquetas en `datas/EDDFS/Annotation/{train,test}.csv` | no indicado | solo investigación no comercial, prohibido redistribuir; consentimiento de descarga |
| BRSET v1.0.1 (16.266 imágenes) | PhysioNet, doi:10.13026/1pht-2b69 [6] | no indicado | **acceso con credenciales**: cuenta PhysioNet acreditada, curso CITI «Data or Specimens Only Research», firma del DUA (PhysioNet Credentialed Health Data License 1.5.0) |
| RFMiD / RIADD (3.200 imágenes) | <https://riadd.grand-challenge.org/download-all-classes/> (Google Drive) o IEEE Dataport [7] | ≈ 5 GB (estimado) | CC BY 4.0 según el espejo de Hugging Face; IEEE Dataport no lo muestra |
| JSIEC «1000 Fundus images with 39 categories» | Kaggle `linchundan/fundusimage1000` [8] | 447 MB | Open Database License (ODbL) / Database Contents License |
| OCT 2017 (Kermany) | Kaggle `paultimothymooney/kermany2018` o Mendeley Data `rscbjbr9sj` [9] | 5,8 GB | CC BY 4.0 |
| OCTDL | Mendeley Data `sncdhf53xc` v4 [10] | no comprobado | CC BY 4.0 (según el artículo de *Sci. Data*; no comprobado) |
| OCTID | Kaggle `saifurrahmanshatil/retinal-oct-dataset` [11] | 47 MB | no indicada |

## 7. Qué se puede comparar con imgad

**Mismo protocolo.** Solo el escenario **one-class** de **fondo de ojo**: entrenamiento con las 7.786 normales
etiquetadas (4.983 EDDFS + 2.803 BRSET), test en RIADD y JSIEC, AUROC global y por categoría, y
F1/especificidad/sensibilidad con umbral. Diferencias que no se pueden evitar:

- **Umbral.** imgad lo calibra con puntuaciones normales fuera de muestra (sin anomalías; objetivo 5 % de falsos
  positivos). BenchReAD no explica el suyo. Las tablas con umbral no son estrictamente comparables.
- **Selección.** BenchReAD elige *checkpoints* con la validación, que contiene 100 anomalías. imgad usa la misma
  validación solo para **elegir la configuración** (la pregunta es la misma: qué modelo informar). Además se
  informa una configuración fijada de antemano (la elegida en PALM: RetiZero + L2 + LOF), que no usa ninguna
  anomalía de BenchReAD.
- **Semillas.** BenchReAD da una ejecución; imgad da media ± d.e. sobre 3 semillas (relevantes para Anomalib y
  detectores estocásticos; con todo el entrenamiento, kNN/LOF/Mahalanobis son deterministas) e IC *bootstrap*.
- **Preprocesado.** `prepare_benchread.py` reproduce el recorte a 512 px (PNG sin pérdidas en lugar de JPEG; PIL
  bilineal en lugar de OpenCV) y deja JSIEC sin recortar, como BenchReAD.

**Fuera de alcance.** No supervisado (SoftPatch), semisupervisado (DDAD-ASR) y supervisado (DRA, NFM-DRA): usan
anomalías o datos sin etiquetar en entrenamiento, que imgad no usa por diseño; sus cifras se dan como
referencia, marcando el nivel de supervisión. OCT: la partición no es reproducible (§2) y el modelo de OCT del zoo
es secundario para el artículo; se puede añadir después con una partición propia por paciente.

**Métodos.** En común: solo **PatchCore** (imgad lo ejecuta vía Anomalib, WideResNet-50, 256 px; BenchReAD usa
224 px y *coreset* 10 %). SimpleNet no está en Anomalib 2 (sí su sucesor SuperSimpleNet). EDC, SoftPatch,
DDAD-ASR, DRA y NFM-DRA no están en imgad. BenchReAD no evalúa: *embeddings* congelados de modelos fundacionales
con detectores clásicos (kNN, Mahalanobis, LOF, GMM, KDE, OCSVM, IForest, PCA, conjunto), modelos fundacionales
retinianos (RETFound, RETFound-Green, VisionFM, RetiZero, FLAIR, UrFound), modelos generales y biomédicos
(DINOv2/v3, SigLIP2, BiomedCLIP, MedSigLIP), detectores con *prompts* de texto (*zero-shot*), los otros 18 modelos de
Anomalib, umbrales calibrados sin anomalías, varias semillas ni pruebas DeLong.

## Fuentes

1. Artículo, HTML de arXiv: <https://arxiv.org/html/2507.10492v1> (texto, Tablas 1–2).
2. Artículo, PDF de arXiv: <https://arxiv.org/pdf/2507.10492v1> (Fig. 2, AUROC con IC).
3. Ficha de arXiv (DOI, fecha): <https://arxiv.org/abs/2507.10492>; versión MICCAI:
   <https://papers.miccai.org/miccai-2025/0092-Paper1683.html>.
4. Repositorio: <https://github.com/DopamineLcy/BenchReAD> (README, `fundus/*.ipynb`, `OCT/*.ipynb`,
   `NFM-DRA/NFM/src/patchcore/datasets/fundus.py`, `NFM-DRA/DRA/datasets/fundus.py`, `NFM-DRA/NFM-DRA_evaluate.py`).
5. EDDFS: <https://github.com/xia-xx-cv/EDDFS_dataset> (README, licencia, etiquetas); Xia et al., *Signal Process.
   Image Commun.* 127, 117151 (2024).
6. BRSET: <https://physionet.org/content/brazilian-ophthalmological/1.0.1/>; Nakayama et al., *PLOS Digit. Health*
   3(7), e0000454 (2024).
7. RIADD/RFMiD: <https://riadd.grand-challenge.org/download-all-classes/>,
   <https://ieee-dataport.org/open-access/retinal-fundus-multi-disease-image-dataset-rfmid>,
   <https://huggingface.co/datasets/ctmedtech/RFMID/blob/main/README.md>; Pachade et al., *Data* 6(2), 14 (2021).
8. JSIEC: <https://www.kaggle.com/datasets/linchundan/fundusimage1000> (metadatos de la API de Kaggle: licencia y
   tamaño); Cen et al., *Nat. Commun.* 12, 4828 (2021).
9. OCT 2017: <https://data.mendeley.com/datasets/rscbjbr9sj/3>,
   <https://www.kaggle.com/datasets/paultimothymooney/kermany2018>; Kermany et al., *Cell* 172(5), 1122–1131 (2018).
10. OCTDL: <https://data.mendeley.com/datasets/sncdhf53xc/4>; Kulyabin et al., *Sci. Data* 11, 365 (2024).
11. OCTID: <https://www.kaggle.com/datasets/saifurrahmanshatil/retinal-oct-dataset>; Gholami et al., *Comput. Electr.
    Eng.* 81, 106532 (2020).
