# Zero- and few-shot detection

Sometimes there are no normal training images, or only a handful. This happens with a new device, a rare
modality, or data that cannot leave the hospital. Vision–language models (VLMs) such as CLIP place images
and text in the same space. An image can then be scored against sentences that describe normal and
anomalous findings, without any training image. This is *zero-shot* detection. With a few normal images
the score can also use their distance to the test image. This is *few-shot* detection.

imgad implements both with one detector, `zero_shot`, and measures how performance grows with the number
of normal images with the [few-shot protocol](#the-few-shot-protocol) of the benchmark.

## Quick start

```python
from imgad import AnomalyDetectionPipeline

pipe = AnomalyDetectionPipeline("zoo:flair-clip", {"name": "zero_shot", "modality": "fundus"})
pipe.fit([])                         # zero-shot: text prompts only
scores = pipe.score(test_images)     # probability of the anomalous prompts, in [0, 1]

pipe.fit(normal_images[:8])          # 8-shot: prompts + the 8 normal images
```

In a benchmark YAML:

```yaml
extractors: [zoo:flair-clip, zoo:biomedclip, zoo:siglip2-base]
detectors:
  - {name: zero_shot, modality: fundus}
  - {name: knn, k: 1}                # visual-only baseline on the same embeddings
k_shots: [0, 1, 2, 4, 8, 16]
seeds: [0, 1, 2]
```

## Which models

The detector needs an extractor whose image embeddings share a space with text:

| Extractor | Domain |
|---|---|
| `zoo:flair-clip`, `zoo:retizero-clip` | Retinal fundus photographs |
| `zoo:biomedclip` | Biomedical figures (PubMed Central), many modalities |
| `zoo:medsiglip` (gated) | Chest X-ray, CT/MRI, dermatology, ophthalmology, histopathology |
| `zoo:siglip2-base`, any CLIP/SigLIP (`hf` with `pooling="projection"`), `open_clip` | General images |

`imgad models -m <modality> --zero-shot` lists them per modality. Other extractors (DINOv2, RETFound, UNI…)
have no text encoder. The benchmark skips the zero-shot detector with them and runs the other detectors.

## How the score works

**Prompts.** The detector writes every *state* into every *template*:

- states such as `"a healthy retina"` and `"diabetic retinopathy"`;
- templates such as `"A fundus photograph of {}"`.

It averages the text embeddings per state, as in CLIP's prompt ensembles and
[WinCLIP](https://arxiv.org/abs/2303.14814). Then it applies a softmax over the cosine similarities
(scaled by the model's temperature). The score is the probability of the anomalous states, summed over
them.

**Few-shot.** Given `k` normal images, the detector also computes the visual score
`(1 − cosine similarity to the nearest normal image) / 2`. It returns
`(1 − w) · text + w · visual`, with `w = visual_weight`, 0.5 by default, like WinCLIP+.

**Modes.** `mode="auto"` uses the text only without images and both scores with them. `"text"`,
`"visual"` and `"both"` force one behaviour.

**Prompt sets.** Each modality has its own templates and states in `imgad.prompts`: `fundus`, `oct`,
`cxr`, `ct`, `mri`, `histopathology` and `dermatology`. `modality=None` gives a generic object/defect set
for industrial images. Replace any part to fit your data:

```python
{"name": "zero_shot", "modality": "oct",
 "anomalous": ["choroidal neovascularization", "diabetic macular edema", "drusen"],
 "templates": ["An OCT scan of {}"]}
```

Naming the findings that actually occur in your data usually helps more than generic words. Benchmark
several prompt sets like any other configuration.

**Threshold.** Zero-shot, the threshold is 0.5 on the probability of the anomalous prompts. With
training images, it is the usual quantile of their leave-one-out scores. Recalibrate it on normal images
of the target site as soon as there are some (`imgad calibrate`).

**Compatibility.** The text comparison needs the joint embeddings exactly as the model produces them.
The only compatible preprocessing is `l2`; the benchmark skips other preprocessing with this detector.
After fitting, the detector stores its text embeddings. A saved pipeline therefore needs only the image
encoder.

## The few-shot protocol

`benchmark(..., k_shots=[0, 1, 2, 4, 8, 16])`, YAML `k_shots:` or CLI `--k-shots 0 1 4 16` runs every
configuration with `k` normal training images drawn per seed:

- Each row gets a `k_shot` column, and the configuration name ends in `| k-shot`.
- `k = 0` runs only the zero-shot detector.
- A detector that cannot fit on `k` images is recorded as a failed run, and the rest go on (see
  [When a run fails](benchmarks.md#when-a-run-fails)). kNN works from one image; with a single image its
  threshold cannot be calibrated.
- The report adds a *Few-shot learning curves* section: the mean metric for each `k` and a plot.

With few images the result depends on which images were drawn. Use several seeds, and compare
configurations with the tests in [Statistical analysis](statistics.md).

## Caveats

- **Data contamination.** VLMs are trained on large collections of public images. A public test set may
  have been part of a model's training data, which inflates its zero-shot results. Check the training
  data of each model, and validate on private or recent data.
- **Prompts are hyper-parameters.** Choosing them on the test set is optimistic. Fix them on a
  validation split, as you would a threshold.
- **Global scores only.** The detector scores whole images. Small lesions may need patch-level methods:
  PatchCore through Anomalib, or WinCLIP-style windows (not yet in imgad).
- **Not included:** learned prompts and adapters trained on auxiliary anomaly data, such as
  [AnomalyCLIP](https://arxiv.org/abs/2310.18961), [AdaCLIP](https://arxiv.org/abs/2407.15795) and
  [MVFA](https://arxiv.org/abs/2403.12570). They need their authors' code and weights.
