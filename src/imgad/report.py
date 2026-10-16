"""Markdown report of a benchmark: dataset analysis, methods, results (with figures) and discussion.

    rows = benchmark(..., keep_scores=True)          # scores enable ROC curves, histograms, DeLong...
    generate_report(rows, datasets, "report/", config=cfg)

or from the CLI: ``imgad benchmark --config cfg.yaml --report report/``. The output folder gets
``report.md``, ``figures/*.png``, ``results.csv`` and a copy of the configuration. Figures need
matplotlib (``pip install "imgad[report]"``); without it the report is written with tables only.

The text of the "Summary" and "Discussion" sections is derived from the numbers by simple rules
(not by an LLM): treat it as a first draft to be checked and completed.
"""

from __future__ import annotations

import datetime as _dt
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .stats import compare_configs, delong_test, md_escape

METRIC_NAMES = {
    "auroc": "AUROC",
    "aupr": "AUPR",
    "f1_max": "F1-max",
    "fpr@95tpr": "FPR@95%TPR",
    "pixel_auroc": "pixel-AUROC",
    "pixel_aupr": "pixel-AUPR",
    "aupro": "AUPRO",
}
DETECTOR_DESC = {
    "knn": "mean distance to the *k* nearest normal samples",
    "mahalanobis": "Mahalanobis distance to a Gaussian fitted to the normals (shrunk covariance)",
    "gaussian": "Mahalanobis distance to a Gaussian fitted to the normals",
    "centroid": "distance to the mean normal embedding",
    "kmeans": "distance to the nearest k-means centroid of the normals",
    "lof": "Local Outlier Factor (local density relative to the neighbours)",
    "gmm": "negative log-likelihood under a Gaussian mixture",
    "kde": "negative log-density of a kernel density estimate",
    "ocsvm": "One-Class SVM (boundary enclosing the normals)",
    "iforest": "Isolation Forest (how easily random splits isolate a point)",
    "isolation_forest": "Isolation Forest",
    "pca": "reconstruction error in the PCA subspace of the normals",
    "pca_reconstruction": "PCA reconstruction error",
    "ensemble": "mean of the normalised scores of several detectors",
    "sklearn": "scikit-learn estimator",
    "pyod": "PyOD model",
}
ANOMALIB_DESC = {
    "patchcore": "memory bank of patch features from a pretrained CNN; score = distance to the nearest normal patch",
    "padim": "per-position multivariate Gaussian over CNN patch features",
    "efficientad": "lightweight student-teacher network + autoencoder; anomaly = their discrepancy",
    "fastflow": "2D normalizing flow over CNN features; anomaly = low likelihood",
    "stfpm": "student-teacher feature pyramid; anomaly = their discrepancy",
    "reversedistillation": "reverse distillation encoder-decoder; anomaly = feature reconstruction error",
    "cflow": "conditional normalizing flow over patch features",
    "draem": "reconstruction + segmentation trained with synthetic anomalies",
    "dfm": "density model on deep features (PCA + Gaussian)",
    "dfkde": "KDE on deep features",
}


# ============================================================================ entry point

def generate_report(
    rows: List[Dict[str, Any]],
    datasets: Sequence = (),
    out_dir="report",
    config: Optional[Dict[str, Any]] = None,
    title: Optional[str] = None,
    metric: str = "auroc",
    alpha: float = 0.05,
    command: Optional[str] = None,
    figures: bool = True,
) -> Path:
    """Write ``out_dir/report.md`` (+ figures, CSV and config). Returns the path of the markdown file.

    ``rows``: output of :func:`imgad.evaluation.benchmark` (ideally with ``keep_scores=True``).
    ``datasets``: the :class:`~imgad.data.AnomalyDataset` objects used (for the dataset analysis,
    per-defect-type results and example images); can be empty.
    """
    out = Path(out_dir)
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plt = _pyplot() if figures else None
    if plt is not None:
        fig_dir.mkdir(exist_ok=True)
    r = _Report(rows, list(datasets), metric, alpha, out, fig_dir, plt)
    md = r.build(title, config, command)
    path = out / "report.md"
    path.write_text(md)
    _write_csv(rows, out / "results.csv")
    if config is not None:
        import yaml

        (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    return path


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight", "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 9})
    return plt


def _write_csv(rows, path) -> None:
    import csv

    scalar = (str, int, float, bool, type(None), np.number)
    rows = [{k: v for k, v in r.items() if isinstance(v, scalar)} for r in rows]
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ============================================================================ helpers

def short_extractor(name: str) -> str:
    """``openai/clip-vit-base-patch16[mean@[6, 9, 12]]`` -> ``clip-vit-base-patch16 · mean@6/9/12``."""
    base, _, extra = str(name).partition("[")
    base = base.split("/")[-1].replace("hf-hub:", "")
    if extra:
        extra = extra.rstrip("]").replace("[", "").replace(", ", "/").replace(",", "/")
        return f"{base} · {extra}"
    return base


def _f(v, nd: int = 3) -> str:
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]], align: Optional[str] = None) -> List[str]:
    align = align or "l" + "r" * (len(header) - 1)
    sep = "|" + "|".join("---:" if a == "r" else "---" for a in align) + "|"
    return ["| " + " | ".join(map(str, header)) + " |", sep] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]


class _Report:
    def __init__(self, rows, datasets, metric, alpha, out: Path, fig_dir: Path, plt):
        self.raw = rows
        self.datasets = {d.name: d for d in datasets}
        self.metric, self.alpha, self.out, self.fig_dir, self.plt = metric, alpha, out, fig_dir, plt
        self.mname = METRIC_NAMES.get(metric, metric)
        self.has_scores = bool(rows) and all("scores" in r for r in rows)
        self.seeds = sorted({r.get("seed", 0) for r in rows})
        self.ds_names = list(dict.fromkeys(r["dataset"] for r in rows))
        self.preps = list(dict.fromkeys(r.get("preprocessing", "-") for r in rows))
        self.meta: Dict[str, Dict[str, Any]] = {}
        groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            cfg = r.get("config") or f"{r['extractor']} | {r['preprocessing']} | {r['detector']}"
            groups[(r["dataset"], cfg)].append(r)
            self.meta.setdefault(cfg, {"extractor": r.get("extractor", "-"), "preprocessing": r.get("preprocessing", "-"),
                                       "detector": r.get("detector", "-")})
        self.groups = groups
        self.configs = list(self.meta)
        # complete configs (present on every dataset) ranked by mean metric
        self.complete = [c for c in self.configs if all((d, c) in groups for d in self.ds_names)]
        sign = -1 if metric == "fpr@95tpr" else 1
        self.ranked = sorted(self.complete, key=lambda c: -sign * self.mean(c))
        self.best = self.ranked[0] if self.ranked else None
        self.figs: List[str] = []

    # ------------------------------------------------------------------ data access
    def val(self, ds: str, cfg: str, metric: Optional[str] = None) -> float:
        rs = self.groups.get((ds, cfg), [])
        v = [r.get(metric or self.metric) for r in rs]
        v = [x for x in v if x is not None]
        return float(np.mean(v)) if v else float("nan")

    def std_seeds(self, ds: str, cfg: str, metric: Optional[str] = None) -> float:
        v = [r.get(metric or self.metric) for r in self.groups.get((ds, cfg), [])]
        v = [x for x in v if x is not None]
        return float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")

    def mean(self, cfg: str, metric: Optional[str] = None) -> float:
        return float(np.nanmean([self.val(d, cfg, metric) for d in self.ds_names]))

    def label(self, cfg: str) -> str:
        m = self.meta[cfg]
        if m["extractor"] == "anomalib":
            return f"{m['detector']} (anomalib)"
        prep = f" [{m['preprocessing']}]" if len(self.preps) > 1 else ""
        return f"{short_extractor(m['extractor'])}{prep} + {m['detector']}"

    def rows_of(self, ds: str, cfg: str) -> List[Dict[str, Any]]:
        return self.groups.get((ds, cfg), [])

    def save(self, fig, name: str, caption: str) -> str:
        path = self.fig_dir / f"{name}.png"
        fig.savefig(path, dpi=130)
        self.plt.close(fig)
        rel = f"figures/{name}.png"
        self.figs.append(rel)
        return f"![{caption}]({rel})\n\n*{caption}*"

    # ------------------------------------------------------------------ assemble
    def build(self, title, config, command) -> str:
        if not self.raw:
            return "# Benchmark report\n\nNo results.\n"
        self._stats = self._run_stats()
        body: List[str] = []
        sections = [
            ("Data analysis", self.sec_data),
            ("Methods", self.sec_methods),
            ("Results", self.sec_results),
            ("Discussion", self.sec_discussion),
            ("Limitations", self.sec_limitations),
            ("Reproducibility", lambda: self.sec_repro(config, command)),
            ("Appendix: results per dataset", self.sec_annex),
        ]
        for name, fn in sections:
            lines = fn()
            if lines:
                body += [f"## {name}", "", *lines, ""]
        head = [
            f"# {title or 'Anomaly detection benchmark report'}",
            "",
            f"*Generated with `imgad` on {_dt.date.today():%Y-%m-%d} · {len(self.ds_names)} dataset(s) · "
            f"{len(self.configs)} configurations · main metric: {self.mname}*",
            "",
            "## Summary",
            "",
            *self.sec_summary(),
            "",
        ]
        return "\n".join(head + body).rstrip() + "\n"

    # ------------------------------------------------------------------ statistics shared by sections
    def _run_stats(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if len(self.ds_names) >= 2 and len(self.complete) >= 2:
            try:
                out["cmp"] = compare_configs([r for r in self.raw if (r.get("config") in self.complete)],
                                             metric=self.metric, alpha=self.alpha)
            except ValueError:
                pass
        if self.has_scores and self.best and self.metric == "auroc":
            delong = []
            for ds in self.ds_names:
                b = self.rows_of(ds, self.best)[0]
                for cfg in self.ranked[1:]:
                    o = self.rows_of(ds, cfg)
                    if o and o[0].get("seed") == b.get("seed"):
                        delong.append({"dataset": ds, "config": cfg, **delong_test(b["labels"], b["scores"], o[0]["scores"])})
            out["delong"] = delong
        return out

    # ------------------------------------------------------------------ summary
    def sec_summary(self) -> List[str]:
        if not self.best:
            return ["No configuration was evaluated on every dataset."]
        L = []
        b, m = self.best, self.mname
        L.append(f"- **Best configuration:** {self.label(b)}, mean {m} **{self.mean(b):.3f}**"
                 + (f" over the {len(self.ds_names)} datasets." if len(self.ds_names) > 1 else "."))
        if len(self.ranked) > 1:
            second = self.ranked[1]
            L.append(f"- Runner-up: {self.label(second)} ({self.mean(second):.3f}; difference "
                     f"{self.mean(b) - self.mean(second):+.3f}). {self._significance_sentence(b, second)}")
        ext = self._best_by("extractor")
        if len(ext) > 1:
            L.append("- **Extractors** (each with its best detector): " + ", ".join(f"{k} {v:.3f}" for k, v in ext))
        det = self._mean_by("detector")
        if len(det) > 1:
            L.append(f"- **Detectors** (mean over extractors): best {', '.join(f'{k} {v:.3f}' for k, v in det[:3])}; "
                     f"worst {', '.join(f'{k} {v:.3f}' for k, v in det[-2:])}.")
        if len(self.ds_names) > 1:
            hardest = min(self.ds_names, key=lambda d: max(self.val(d, c) for c in self.complete))
            best_h = max(self.complete, key=lambda c: self.val(hardest, c))
            L.append(f"- **Hardest dataset:** {hardest} (best {m} {self.val(hardest, best_h):.3f}, {self.label(best_h)}).")
            oracle = np.mean([max(self.val(d, c) for c in self.complete) for d in self.ds_names])
            L.append(f"- Picking the best configuration on each dataset would give a mean {m} of {oracle:.3f} "
                     "(optimistic bound: the choice is made on the test set).")
        if any("pixel_auroc" in r for r in self.raw):
            pix = [c for c in self.complete if not np.isnan(self.mean(c, "pixel_auroc"))]
            if pix:
                bp = max(pix, key=lambda c: self.mean(c, "pixel_auroc"))
                L.append(f"- **Localisation:** best pixel-AUROC {self.mean(bp, 'pixel_auroc'):.3f} and AUPRO "
                         f"{self.mean(bp, 'aupro'):.3f} with {self.label(bp)}.")
        return L

    def _significance_sentence(self, a: str, b: str) -> str:
        cmp = self._stats.get("cmp")
        if cmp:
            for p in cmp["pairs"]:
                if {p["a"], p["b"]} == {a, b}:
                    return ("The difference is statistically significant across datasets "
                            f"(Wilcoxon-Holm p={p['wilcoxon_p_holm']:.3g})." if p["significant"] else
                            f"The difference is **not** significant across datasets (Wilcoxon-Holm p={p['wilcoxon_p_holm']:.3g}).")
        dl = [t for t in self._stats.get("delong", []) if t["config"] == b]
        if dl:
            sig = sum(t["p_value"] < self.alpha for t in dl)
            return f"DeLong: significant difference on {sig} of {len(dl)} dataset(s) (α={self.alpha}, uncorrected)."
        return ""

    def _best_by(self, key: str) -> List[Tuple[str, float]]:
        best: Dict[str, float] = {}
        for c in self.complete:
            k = short_extractor(self.meta[c][key]) if self.meta[c]["extractor"] != "anomalib" else self.label(c)
            best[k] = max(best.get(k, -np.inf), self.mean(c))
        return sorted(best.items(), key=lambda kv: -kv[1])

    def _mean_by(self, key: str) -> List[Tuple[str, float]]:
        acc: Dict[str, List[float]] = defaultdict(list)
        for c in self.complete:
            if self.meta[c]["extractor"] != "anomalib":
                acc[self.meta[c][key]].append(self.mean(c))
        return sorted(((k, float(np.mean(v))) for k, v in acc.items()), key=lambda kv: -kv[1])

    # ------------------------------------------------------------------ data analysis
    def sec_data(self) -> List[str]:
        if not self.datasets:
            return [f"Evaluated datasets: {', '.join(self.ds_names)} (the dataset objects were not provided, "
                    "so their analysis is skipped)."]
        L = ["**One-class** setting: models are trained on normal images only and must separate normal from "
             "anomalous images at test time.", ""]
        info = {name: _dataset_info(d) for name, d in self.datasets.items()}
        header = ["Dataset", "Train (normal)", "Test normal", "Test anomalous", "% anomalous", "Defect types", "Masks", "Resolution"]
        rows = []
        for name, i in info.items():
            rows.append([name, i["n_train"], i["n_test_normal"], i["n_test_anom"], f"{100 * i['anom_frac']:.0f}%",
                         len(i["types"]), "yes" if i["masks"] else "no", i["resolution"]])
        L += _table(header, rows, "lrrrrrll") + [""]
        if self.plt is not None:
            L += [self.fig_counts(info), ""]
        # defect types
        if any(i["types"] for i in info.values()):
            L += ["<details><summary>Defect types per dataset</summary>", ""]
            L += _table(["Dataset", "Type", "Test images"],
                        [[n, t, c] for n, i in info.items() for t, c in i["types"].items()], "llr")
            L += ["", "</details>", ""]
        if self.plt is not None:
            for name, d in self.datasets.items():
                fig = self.fig_examples(d)
                if fig:
                    L += [fig, ""]
        L += self._data_observations(info)
        return L

    def _data_observations(self, info) -> List[str]:
        obs = []
        small = [n for n, i in info.items() if i["n_test_normal"] < 30]
        if small:
            obs.append(f"Test sets with few normal images (<30): {', '.join(small)}. With so few images, AUROC "
                       "differences of ±0.01–0.02 are usually within noise; see the confidence intervals.")
        few = [n for n, i in info.items() if i["n_train"] < 50]
        if few:
            obs.append(f"Few training images (<50) in {', '.join(few)}: detectors that estimate covariances or "
                       "densities in high dimension (Mahalanobis, GMM, KDE) may suffer.")
        imb = [n for n, i in info.items() if i["anom_frac"] > 0.7 or i["anom_frac"] < 0.2]
        if imb:
            obs.append(f"Highly imbalanced test set in {', '.join(imb)}: AUPR depends on the anomaly ratio, so it is "
                       "not comparable across datasets; AUROC is.")
        mixed = [n for n, i in info.items() if i["mixed_res"]]
        if mixed:
            obs.append(f"Mixed image resolutions in {', '.join(mixed)} (all are resized to each model's input).")
        if not obs:
            obs.append("The datasets have a reasonable size and balance for comparing configurations.")
        return ["**Observations about the data**", ""] + [f"- {o}" for o in obs]

    def fig_counts(self, info) -> str:
        plt = self.plt
        names = list(info)
        x = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(names) + 2), 3.2))
        w = 0.27
        for j, (key, lab, col) in enumerate([("n_train", "train (normal)", "#4c78a8"),
                                             ("n_test_normal", "test normal", "#72b7b2"),
                                             ("n_test_anom", "test anomalous", "#e45756")]):
            vals = [info[n][key] for n in names]
            ax.bar(x + (j - 1) * w, vals, w, label=lab, color=col)
        ax.set_xticks(x, names, rotation=30 if len(names) > 4 else 0, ha="right" if len(names) > 4 else "center")
        ax.set_ylabel("number of images")
        ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
        return self.save(fig, "data_sizes", "Number of images per dataset and split.")

    def fig_examples(self, d) -> Optional[str]:
        from .maps import load_mask
        from .utils import load_image

        plt = self.plt
        normals = d.train_paths[:4]
        y, types = np.asarray(d.test_labels), list(d.test_types)
        anom = []
        for t in dict.fromkeys(t for t, l in zip(types, y) if l == 1):
            i = next(i for i, (tt, l) in enumerate(zip(types, y)) if l == 1 and tt == t)
            anom.append(i)
            if len(anom) == 4:
                break
        if not normals and not anom:
            return None
        ncol = max(len(normals), len(anom), 1)
        fig, axes = plt.subplots(2, ncol, figsize=(2.0 * ncol, 4.3), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for j, p in enumerate(normals):
            axes[0, j].imshow(_thumb(load_image(p)))
            axes[0, j].set_title("normal (train)", fontsize=8)
        masks = d.test_masks if d.has_masks else [None] * len(d.test_paths)
        for j, i in enumerate(anom):
            img = _thumb(load_image(d.test_paths[i]))
            axes[1, j].imshow(img)
            if masks[i] is not None:
                m = load_mask(masks[i], img.shape[:2])
                axes[1, j].contour(m, levels=[0.5], colors="red", linewidths=1)
            axes[1, j].set_title(types[i], fontsize=8, color="#b22")
        fig.suptitle(d.name, fontsize=10)
        safe = d.name.replace("/", "_")
        return self.save(fig, f"data_examples_{safe}", f"Examples from {d.name}: normal training images (top) and one "
                         "example per defect type (bottom; ground-truth mask in red when available).")

    # ------------------------------------------------------------------ methods
    def sec_methods(self) -> List[str]:
        L = []
        feat = [c for c in self.configs if self.meta[c]["extractor"] != "anomalib"]
        anom = [c for c in self.configs if self.meta[c]["extractor"] == "anomalib"]
        if feat:
            L += ["**Feature pipelines + classical detector.** Each image is mapped to an embedding by a frozen "
                  "pretrained model; after preprocessing, a one-class detector fitted on the normal images assigns an "
                  "anomaly score (higher = more anomalous).", ""]
            exts = {}
            for c in feat:
                r = next(r for r in self.raw if r.get("config") == c)
                exts.setdefault(r["extractor"], r.get("feat_dim", "–"))
            L += ["Extractors:", ""] + _table(["Extractor", "Identifier", "Dimension"],
                                               [[short_extractor(e), f"`{e}`", d] for e, d in exts.items()], "llr") + [""]
            L.append("Preprocessing: " + ", ".join(f"`{p}`" for p in self.preps if p) + ".")
            L.append("")
            dets = list(dict.fromkeys(self.meta[c]["detector"] for c in feat))
            L += ["Detectors:", ""] + _table(["Detector", "Description"],
                                              [[d, DETECTOR_DESC.get(d.split("(")[0], "–")] for d in dets], "ll") + [""]
        if anom:
            L += ["**Anomalib models** (end-to-end, with pixel-level anomaly maps). Raw model scores and maps are "
                  "used; Anomalib's post-processing (which needs labelled anomalies) is disabled.", ""]
            ms = list(dict.fromkeys(self.meta[c]["detector"] for c in anom))
            L += _table(["Model", "Description"],
                        [[m, ANOMALIB_DESC.get(m.split("(")[0].replace("_", ""), "–")] for m in ms], "ll") + [""]
        q = sorted({r.get("threshold_quantile") for r in self.raw if r.get("threshold_quantile") is not None})
        qs = ", ".join(f"{x:g}" for x in q) or "0.95"
        L += ["**Protocol.**", "",
              "- Training on normal images only; testing on normal and anomalous images.",
              f"- Decision threshold: quantile {qs} of *out-of-sample* normal scores (cross-validation or leave-one-out "
              "for detectors; a held-out split for anomalib). No anomaly is used.",
              f"- Main metric: **{self.mname}**. Also AUPR, F1-max (best F1 over all thresholds), FPR@95%TPR and "
              "TPR/FPR/F1 at the automatic threshold."]
        if any("pixel_auroc" in r for r in self.raw):
            L.append("- Localisation (models with maps): pixel-AUROC, pixel-AUPR and AUPRO (area under the PRO curve up to FPR=0.3).")
        sub = sorted({r.get("n_train") for r in self.raw})
        L.append(f"- Repetitions: {len(self.seeds)} seed(s) ({', '.join(map(str, self.seeds))})."
                 + (" Each seed uses a different subset of the normal training images." if len(sub) > 1 or
                    any(r.get("n_train", 0) < len(self.datasets[r['dataset']].train_paths) for r in self.raw if r["dataset"] in self.datasets) else ""))
        if any("auroc_ci_low" in r for r in self.raw):
            L.append(f"- {1 - self.alpha:.0%} confidence intervals by stratified bootstrap of the test set.")
        tests = []
        if "cmp" in self._stats:
            tests.append("Friedman + Nemenyi and pairwise Wilcoxon with Holm correction, each dataset being a block "
                         "(Demšar, 2006)")
        if self._stats.get("delong"):
            tests.append("DeLong test to compare AUROCs on the same test images")
        if tests:
            L.append("- Statistical tests: " + "; ".join(tests) + f" (α={self.alpha}).")
        return L

    # ------------------------------------------------------------------ results
    def sec_results(self) -> List[str]:
        L: List[str] = []
        L += self.res_ranking()
        L += self.res_grid()
        L += self.res_per_dataset()
        L += self.res_curves()
        L += self.res_defect_types()
        L += self.res_operating_point()
        L += self.res_pixel()
        L += self.res_seeds()
        L += self.res_stats()
        L += self.res_times()
        return L

    def res_ranking(self, top: int = 15) -> List[str]:
        has_pix = any("pixel_auroc" in r for r in self.raw)
        multi_seed = len(self.seeds) > 1
        cmp = self._stats.get("cmp")
        ranks = dict(zip(cmp["configs"], cmp["avg_rank"])) if cmp else {}
        header = ["#", "Configuration", f"mean {self.mname}"] + (["± std over seeds"] if multi_seed else []) + \
                 ["AUPR", "F1-max"] + (["mean rank"] if ranks else []) + (["pixel-AUROC", "AUPRO"] if has_pix else [])
        rows = []
        for i, c in enumerate(self.ranked[:top], 1):
            row = [i, md_escape(self.label(c)), f"**{self.mean(c):.3f}**" if i == 1 else f"{self.mean(c):.3f}"]
            if multi_seed:
                row.append(_f(np.nanmean([self.std_seeds(d, c) for d in self.ds_names])))
            row += [_f(self.mean(c, "aupr")), _f(self.mean(c, "f1_max"))]
            if ranks:
                row.append(_f(ranks.get(c), 2))
            if has_pix:
                row += [_f(self.mean(c, "pixel_auroc")), _f(self.mean(c, "aupro"))]
            rows.append(row)
        L = ["### Ranking of configurations", "",
             f"Mean over {'the ' + str(len(self.ds_names)) + ' datasets' if len(self.ds_names) > 1 else 'the dataset'}"
             + (" and seeds" if multi_seed else "") + f"; top {min(top, len(self.ranked))} of {len(self.ranked)} shown.", ""]
        L += _table(header, rows, "rl" + "r" * (len(header) - 2)) + [""]
        if self.plt is not None and self.ranked:
            L += [self.fig_ranking(), ""]
        return L

    def fig_ranking(self, top: int = 15) -> str:
        plt = self.plt
        cfgs = self.ranked[:top][::-1]
        fig, ax = plt.subplots(figsize=(7.5, 0.35 * len(cfgs) + 1.2))
        means = [self.mean(c) for c in cfgs]
        ax.barh(range(len(cfgs)), means, color=["#e45756" if c == self.best else "#4c78a8" for c in cfgs], alpha=0.85)
        if len(self.ds_names) > 1:
            for i, c in enumerate(cfgs):
                vals = [self.val(d, c) for d in self.ds_names]
                ax.scatter(vals, np.full(len(vals), i) + np.linspace(-0.15, 0.15, len(vals)), s=9, color="black", zorder=3)
        elif any("auroc_ci_low" in r for r in self.raw) and self.metric in ("auroc", "aupr"):
            d = self.ds_names[0]
            lo = [np.mean([r[f"{self.metric}_ci_low"] for r in self.rows_of(d, c)]) for c in cfgs]
            hi = [np.mean([r[f"{self.metric}_ci_high"] for r in self.rows_of(d, c)]) for c in cfgs]
            ax.errorbar(means, range(len(cfgs)), xerr=[np.subtract(means, lo), np.subtract(hi, means)], fmt="none",
                        ecolor="black", capsize=2, lw=0.8)
        ax.set_yticks(range(len(cfgs)), [self.label(c) for c in cfgs])
        lo_x = min(min(means), *(self.val(d, c) for d in self.ds_names for c in cfgs))
        ax.set_xlim(max(0, lo_x - 0.05), 1.0 if self.metric != "fpr@95tpr" else None)
        ax.set_xlabel(self.mname)
        note = "dots = individual datasets" if len(self.ds_names) > 1 else "error bars = bootstrap CI"
        return self.save(fig, "results_ranking", f"Mean {self.mname} of the best configurations ({note}).")

    def res_grid(self) -> List[str]:
        feat = [c for c in self.complete if self.meta[c]["extractor"] != "anomalib"]
        exts = list(dict.fromkeys(self.meta[c]["extractor"] + (f" [{self.meta[c]['preprocessing']}]" if len(self.preps) > 1 else "") for c in feat))
        dets = list(dict.fromkeys(self.meta[c]["detector"] for c in feat))
        if len(exts) < 2 and len(dets) < 2:
            return []
        M = np.full((len(dets), len(exts)), np.nan)
        for c in feat:
            m = self.meta[c]
            e = m["extractor"] + (f" [{m['preprocessing']}]" if len(self.preps) > 1 else "")
            M[dets.index(m["detector"]), exts.index(e)] = max(np.nan_to_num(M[dets.index(m["detector"]), exts.index(e)], nan=-1), self.mean(c))
        order = np.argsort(-np.nanmean(M, axis=1))
        M, dets = M[order], [dets[i] for i in order]
        best = np.nanmax(M)
        rows = [[d] + [("**" + _f(v) + "**") if v == best else _f(v) for v in M[i]] for i, d in enumerate(dets)]
        rows.append(["*mean*"] + [f"*{_f(v)}*" for v in np.nanmean(M, axis=0)])
        L = [f"### Mean {self.mname}: extractor × detector", ""]
        L += _table(["Detector"] + [md_escape(short_extractor(e)) for e in exts], rows) + [""]
        self._grid = (dets, exts, M)
        if self.plt is not None:
            L += [self.fig_grid(dets, exts, M), ""]
        return L

    def fig_grid(self, dets, exts, M) -> str:
        plt = self.plt
        fig, ax = plt.subplots(figsize=(1.3 * len(exts) + 2.5, 0.42 * len(dets) + 1.5))
        im = ax.imshow(M, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(exts)), [short_extractor(e) for e in exts], rotation=30, ha="right")
        ax.set_yticks(range(len(dets)), dets)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]):
                    v = M[i, j]
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
                            color="white" if v < np.nanmean(M) else "black")
        fig.colorbar(im, ax=ax, label=self.mname, shrink=0.8)
        ax.spines[:].set_visible(False)
        return self.save(fig, "results_extractor_detector", f"Mean {self.mname} per extractor and detector.")

    def res_per_dataset(self) -> List[str]:
        if len(self.ds_names) < 2 and not any("auroc_ci_low" in r for r in self.raw):
            return []
        L = ["### Results per dataset", ""]
        rows = []
        for d in self.ds_names:
            cs = sorted(self.complete, key=lambda c: -self.val(d, c))
            b = cs[0]
            r0 = self.rows_of(d, b)
            ci = (f"[{np.mean([r['auroc_ci_low'] for r in r0]):.3f}, {np.mean([r['auroc_ci_high'] for r in r0]):.3f}]"
                  if self.metric == "auroc" and "auroc_ci_low" in r0[0] else "–")
            second = cs[1] if len(cs) > 1 else None
            rows.append([d, md_escape(self.label(b)), f"{self.val(d, b):.3f}", ci,
                         md_escape(self.label(second)) if second else "–", _f(self.val(d, second)) if second else "–",
                         _f(self.val(d, self.best))])
        L += _table(["Dataset", "Best configuration", self.mname, "CI (bootstrap)", "Runner-up", self.mname,
                     "Overall best"], rows, "lllllrr") + [""]
        if self.plt is not None and len(self.ds_names) > 1:
            L += [self.fig_per_dataset(), ""]
        return L

    def fig_per_dataset(self, k: int = 6) -> str:
        plt = self.plt
        cfgs = self.ranked[:k]
        x = np.arange(len(self.ds_names))
        w = 0.8 / len(cfgs)
        fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(self.ds_names) + 2), 3.6))
        cmap = plt.get_cmap("tab10")
        for j, c in enumerate(cfgs):
            ax.bar(x + (j - (len(cfgs) - 1) / 2) * w, [self.val(d, c) for d in self.ds_names], w, label=self.label(c), color=cmap(j))
        ax.set_xticks(x, self.ds_names, rotation=30 if len(self.ds_names) > 4 else 0, ha="right" if len(self.ds_names) > 4 else "center")
        lo = min(self.val(d, c) for d in self.ds_names for c in cfgs)
        ax.set_ylim(max(0, lo - 0.05), 1.0)
        ax.set_ylabel(self.mname)
        ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18 if len(self.ds_names) <= 4 else -0.3))
        return self.save(fig, "results_per_dataset", f"{self.mname} of the {len(cfgs)} best configurations on each dataset.")

    def res_curves(self) -> List[str]:
        if not (self.has_scores and self.plt is not None and self.best):
            return []
        from sklearn.metrics import roc_curve

        plt = self.plt
        n = len(self.ds_names)
        ncol = min(n, 4)
        nrow = int(np.ceil(n / ncol))
        cfgs = self.ranked[:4]
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.0 * nrow), squeeze=False)
        cmap = plt.get_cmap("tab10")
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        for ax, d in zip(axes.ravel(), self.ds_names):
            for j, c in enumerate(cfgs):
                r = self.rows_of(d, c)[0]
                fpr, tpr, _ = roc_curve(r["labels"], r["scores"])
                ax.plot(fpr, tpr, color=cmap(j), lw=1.2, label=f"{self.label(c)} ({r['auroc']:.3f})")
            ax.plot([0, 1], [0, 1], ls=":", color="grey", lw=0.8)
            ax.set_title(d, fontsize=9)
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.legend(fontsize=6, frameon=False, loc="lower right")
        L = ["### ROC curves", "", self.save(fig, "results_roc", "ROC curves of the 4 best configurations on each dataset "
                                             "(first seed; AUROC in brackets)."), ""]
        L += [self.fig_hist(), ""]
        return L

    def fig_hist(self) -> str:
        plt = self.plt
        n = len(self.ds_names)
        ncol = min(n, 4)
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow), squeeze=False)
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        for ax, d in zip(axes.ravel(), self.ds_names):
            r = self.rows_of(d, self.best)[0]
            s, y = np.asarray(r["scores"]), np.asarray(r["labels"])
            bins = np.histogram_bin_edges(s, bins=25)
            ax.hist(s[y == 0], bins=bins, alpha=0.6, color="#4c78a8", label="normal")
            ax.hist(s[y == 1], bins=bins, alpha=0.6, color="#e45756", label="anomalous")
            if r.get("threshold") is not None:
                ax.axvline(r["threshold"], color="black", ls="--", lw=1, label="threshold")
            ax.set_title(d, fontsize=9)
            ax.set_xlabel("score")
            ax.legend(fontsize=6, frameon=False)
        return self.save(fig, "results_scores", f"Test score distribution of the best configuration "
                         f"({self.label(self.best)}) with the automatic threshold.")

    def res_defect_types(self) -> List[str]:
        if not (self.has_scores and self.best and self.datasets):
            return []
        from .evaluation import per_type_auroc

        rows, data = [], []
        for d in self.ds_names:
            ds = self.datasets.get(d)
            if ds is None:
                continue
            r = self.rows_of(d, self.best)[0]
            types = list(ds.test_types)
            if len(types) != len(r["scores"]):
                continue
            pt = per_type_auroc(r["labels"], r["scores"], types)
            counts = Counter(types)
            for t, v in pt.items():
                rows.append([d, t, counts[t], f"{v:.3f}"])
                data.append((f"{d} · {t}", v))
        if not rows or len(rows) < 2:
            return []
        self._types = data
        L = ["### Results per defect type", "",
             f"Best configuration ({md_escape(self.label(self.best))}): AUROC of each defect type against all normal "
             "test images of its dataset.", ""]
        L += _table(["Dataset", "Defect type", "Test images", "AUROC"], rows, "llrr") + [""]
        if self.plt is not None:
            plt = self.plt
            data_s = sorted(data, key=lambda kv: kv[1])
            fig, ax = plt.subplots(figsize=(6.5, 0.28 * len(data_s) + 1))
            vals = [v for _, v in data_s]
            ax.barh(range(len(data_s)), vals, color=["#e45756" if v < 0.9 else "#4c78a8" for v in vals])
            ax.set_yticks(range(len(data_s)), [k for k, _ in data_s], fontsize=7)
            ax.set_xlim(max(0, min(vals) - 0.05), 1.0)
            ax.axvline(0.9, color="grey", ls=":", lw=0.8)
            ax.set_xlabel("AUROC")
            L += [self.save(fig, "results_defect_types", "AUROC per defect type (red: < 0.90)."), ""]
        return L

    def res_operating_point(self) -> List[str]:
        if not self.best or "tpr@thr" not in self.raw[0]:
            return []
        rows = []
        for d in self.ds_names:
            rows.append([d] + [_f(self.val(d, self.best, m)) for m in
                               ("auroc", "aupr", "f1_max", "fpr@95tpr", "tpr@thr", "fpr@thr", "f1@thr")])
        q = self.rows_of(self.ds_names[0], self.best)[0].get("threshold_quantile", 0.95)
        L = ["### Operating point with the automatic threshold", "",
             f"Best configuration ({md_escape(self.label(self.best))}); threshold = quantile {q:g} of out-of-sample "
             f"normal scores, so the expected FPR on normal images is ≈ {1 - q:.0%}.", ""]
        L += _table(["Dataset", "AUROC", "AUPR", "F1-max", "FPR@95%TPR", "TPR@thr", "FPR@thr", "F1@thr"], rows)
        return L + [""]

    def res_pixel(self) -> List[str]:
        pix = [c for c in self.configs if any("pixel_auroc" in r for d in self.ds_names for r in self.rows_of(d, c))]
        if not pix:
            return []
        rows = []
        for c in pix:
            for d in self.ds_names:
                if self.rows_of(d, c):
                    rows.append([md_escape(self.label(c)), d, _f(self.val(d, c, "pixel_auroc")),
                                 _f(self.val(d, c, "pixel_aupr")), _f(self.val(d, c, "aupro")), _f(self.val(d, c))])
        L = ["### Localisation (pixel level)", ""]
        L += _table(["Model", "Dataset", "pixel-AUROC", "pixel-AUPR", "AUPRO", "image AUROC"], rows, "llrrrr") + [""]
        if self.plt is not None:
            fig = self.fig_maps(pix)
            if fig:
                L += [fig, ""]
        return L

    def fig_maps(self, pix) -> Optional[str]:
        from .maps import colorize, resize_map
        from .utils import load_image

        items = []
        for c in pix:
            for d in self.ds_names:
                rs = self.rows_of(d, c)
                if rs and rs[0].get("example_maps") and d in self.datasets:
                    items.append((c, d, rs[0]))
        if not items:
            return None
        items = items[:4]
        ncol = max(len(it[2]["example_maps"]) for it in items)
        plt = self.plt
        fig, axes = plt.subplots(len(items), ncol, figsize=(2.1 * ncol, 2.3 * len(items)), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for i, (c, d, r) in enumerate(items):
            ds = self.datasets[d]
            for j, (idx, m) in enumerate(r["example_maps"].items()):
                img = _thumb(load_image(ds.test_paths[idx]))
                heat = colorize(resize_map(m, img.shape[:2]), r["map_reference"], r["pixel_threshold"])
                axes[i, j].imshow((0.5 * img + 0.5 * heat).astype(np.uint8))
                axes[i, j].set_title(ds.test_types[idx] if ds.test_labels[idx] else "normal", fontsize=8)
            name = self.label(c).replace(" (anomalib)", "")
            axes[i, 0].annotate(f"{d}\n{name.split('(')[0]}", xy=(-0.08, 0.5), xycoords="axes fraction", rotation=90,
                                ha="right", va="center", fontsize=7)
        return self.save(fig, "results_maps", "Example anomaly maps (blue = normal, red = above the pixel threshold): "
                         "one example per defect type and one normal image.")

    def res_seeds(self) -> List[str]:
        if len(self.seeds) < 2:
            return []
        L = ["### Variability across seeds", ""]
        stds = [self.std_seeds(d, c) for d in self.ds_names for c in self.complete]
        L.append(f"Standard deviation of {self.mname} across seeds: median {np.nanmedian(stds):.4f}, max {np.nanmax(stds):.4f}.")
        L.append("")
        if self.plt is not None:
            plt = self.plt
            cfgs = self.ranked[:10]
            data = []
            for c in cfgs:
                per_seed = defaultdict(list)
                for d in self.ds_names:
                    for r in self.rows_of(d, c):
                        per_seed[r.get("seed", 0)].append(r[self.metric])
                data.append([np.mean(v) for v in per_seed.values()])
            fig, ax = plt.subplots(figsize=(7, 0.4 * len(cfgs) + 1.2))
            ax.boxplot(data[::-1], vert=False, widths=0.5)
            ax.set_yticks(range(1, len(cfgs) + 1), [self.label(c) for c in cfgs[::-1]])
            ax.set_xlabel(f"mean {self.mname} over datasets (one sample per seed)")
            L += [self.save(fig, "results_seeds", "Spread across seeds of the best configurations."), ""]
        return L

    def res_stats(self) -> List[str]:
        L = []
        cmp = self._stats.get("cmp")
        if cmp:
            L += ["### Statistical comparison of configurations", ""]
            k, n = len(cmp["configs"]), len(cmp["datasets"])
            if "friedman" in cmp:
                f = cmp["friedman"]
                verdict = "there are significant differences" if f["iman_davenport_p"] < self.alpha else "no significant differences are detected"
                L.append(f"Friedman over {k} configurations and {n} datasets: χ²={f['chi2']:.2f}, p={f['p_value']:.3g} "
                         f"(Iman-Davenport p={f['iman_davenport_p']:.3g}): {verdict} between configurations. "
                         f"Nemenyi critical difference: CD={cmp['cd']:.2f} ranks.")
                L.append("")
            best = cmp["configs"][0]
            rows = []
            for p in cmp["pairs"]:
                if best not in (p["a"], p["b"]):
                    continue
                other = p["b"] if p["a"] == best else p["a"]
                sgn = 1 if p["a"] == best else -1
                w, l = (p["wins"], p["losses"]) if sgn == 1 else (p["losses"], p["wins"])
                rows.append([md_escape(self.label(other)), f"{sgn * p['mean_diff']:+.3f}", f"{w}/{p['ties']}/{l}",
                             f"{p['wilcoxon_p_holm']:.3g}" + (" *" if p["significant"] else ""),
                             _f(p.get("nemenyi_p"), 3)])
            L += [f"Best configuration by mean rank, **{md_escape(self.label(best))}**, against each of the others "
                  "(W/T/L = datasets where it wins/ties/loses; * = significant after Holm):", ""]
            L += _table(["Versus", f"Δ{self.mname}", "W/T/L", "Wilcoxon p (Holm)", "Nemenyi p"], rows[:25], "lrrrr") + [""]
            if self.plt is not None and "cd" in cmp:
                L += [self.fig_cd(), ""]
        dl = self._stats.get("delong")
        if dl and len(self.ds_names) == 1:
            L += ["### DeLong test against the best configuration", ""]
            rows = [[md_escape(self.label(t["config"])), f"{t['diff']:+.3f}", f"[{t['diff_low']:+.3f}, {t['diff_high']:+.3f}]",
                     f"{t['p_value']:.3g}" + (" *" if t["p_value"] < self.alpha else "")] for t in dl[:25]]
            L += _table(["Versus", "ΔAUROC (best − other)", "95% CI", "p"], rows, "lrrr") + [""]
        return L

    def fig_cd(self, top: int = 15) -> str:
        from .stats import compare_configs, plot_cd_diagram

        cfgs = set(self._stats["cmp"]["configs"][:top])
        rows = [dict(r, config=self.label(r["config"])) for r in self.raw if r.get("config") in cfgs]
        res = compare_configs(rows, metric=self.metric, alpha=self.alpha)
        fig = plot_cd_diagram(res)
        note = f" (top {top})" if len(self._stats["cmp"]["configs"]) > top else ""
        return self.save(fig, "results_cd", f"Critical difference diagram{note}: mean rank over datasets (1 = best); "
                         "configurations joined by a bar are not significantly different (Nemenyi).")

    def res_times(self) -> List[str]:
        acc: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        for r in self.raw:
            key = short_extractor(r["extractor"]) if r["extractor"] != "anomalib" else f"{r['detector']} (anomalib)"
            for k in ("feat_time_s", "det_time_s", "fit_time_s", "test_time_s"):
                if r.get(k) is not None:
                    acc[key][k].append(float(r[k]))
        if not acc:
            return []
        rows = [[md_escape(k), _f(np.mean(v["feat_time_s"]), 1) if v.get("feat_time_s") else "–",
                 _f(np.mean(v["det_time_s"]), 3) if v.get("det_time_s") else "–",
                 _f(np.mean(v["fit_time_s"]), 1) if v.get("fit_time_s") else "–",
                 _f(np.mean(v["test_time_s"]), 1) if v.get("test_time_s") else "–"] for k, v in acc.items()]
        return ["### Computational cost", "", "Mean times per dataset (seconds; feature extraction is cached and shared "
                "across detectors and seeds).", ""] + \
            _table(["Method", "Feature extraction", "Detector fit+test", "Training (anomalib)", "Test (anomalib)"], rows) + [""]

    # ------------------------------------------------------------------ discussion
    def sec_discussion(self) -> List[str]:
        if not self.best:
            return []
        P: List[str] = []
        m = self.mname
        # 1. extractor vs detector effect
        g = getattr(self, "_grid", None)
        if g is not None:
            dets, exts, M = g
            if M.shape[1] > 1 and M.shape[0] > 1:
                det_spread = np.nanmean(np.nanmax(M, axis=0) - np.nanmin(M, axis=0))
                ext_spread = np.nanmean(np.nanmax(M, axis=1) - np.nanmin(M, axis=1))
                which = "the detector" if det_spread > ext_spread else "the extractor"
                P.append(f"**What matters more, the extractor or the detector?** With a fixed extractor, changing the detector moves "
                         f"the mean {m} by {det_spread:.3f} on average (best minus worst detector); with a fixed detector, "
                         f"changing the extractor moves it by {ext_spread:.3f}. On these data **{which}** matters more.")
                best_det_per_ext = [dets[int(np.nanargmax(M[:, j]))] for j in range(M.shape[1])]
                top = Counter(best_det_per_ext).most_common(1)[0]
                P.append(f"The best detector is {top[0]} with {top[1]} of {len(exts)} extractors"
                         + ("; the detector ranking is stable across extractors." if top[1] == len(exts) else
                            f" (with the others, {', '.join(sorted(set(best_det_per_ext) - {top[0]}))} wins)."))
                worst = [dets[int(np.nanargmin(M[:, j]))] for j in range(M.shape[1])]
                w = Counter(worst).most_common(1)[0]
                P.append(f"The worst detector is {w[0]} with {w[1]} of {len(exts)} extractors.")
        # 2. consistency across datasets
        if len(self.ds_names) > 1:
            wins = sum(self.val(d, self.best) >= max(self.val(d, c) for c in self.complete) - 1e-12 for d in self.ds_names)
            close = sum(self.val(d, self.best) >= max(self.val(d, c) for c in self.complete) - 0.01 for d in self.ds_names)
            P.append(f"**Consistency.** The overall best configuration ({self.label(self.best)}) is the best on {wins} of "
                     f"{len(self.ds_names)} datasets and within 0.01 of the best on {close}.")
        # 3. significance
        cmp = self._stats.get("cmp")
        if cmp:
            sig = [p for p in cmp["pairs"] if cmp["configs"][0] in (p["a"], p["b"]) and p["significant"]]
            n_other = len(cmp["configs"]) - 1
            P.append(f"**Significance.** After Holm correction, the best configuration by mean rank is significantly better "
                     f"than {len(sig)} of the other {n_other}. "
                     + ("With few datasets the power is low: a non-significant difference does not mean that the "
                        "configurations are equivalent." if len(self.ds_names) < 8 else ""))
        dl = self._stats.get("delong")
        if dl and not cmp:
            sig = [t for t in dl if t["p_value"] < self.alpha]
            P.append(f"**Significance.** According to DeLong, the best configuration has a significantly higher AUROC "
                     f"(α={self.alpha}, uncorrected for multiple comparisons) than {len(sig)} of the other {len(dl)}.")
        ci = [r["auroc_ci_high"] - r["auroc_ci_low"] for r in self.raw if "auroc_ci_low" in r]
        if ci:
            P.append(f"The median width of the {1 - self.alpha:.0%} AUROC CI is {np.median(ci):.3f}: smaller differences between "
                     "configurations on the same dataset are not conclusive.")
        # 4. hardest defect types
        types = getattr(self, "_types", None)
        if types:
            bad = [k for k, v in sorted(types, key=lambda kv: kv[1]) if v < 0.9]
            if bad:
                P.append(f"**Hard defect types.** With the best configuration, {len(bad)} type(s) stay below AUROC 0.90: "
                         f"{', '.join(bad[:6])}. These are usually small or local defects, which barely change a global "
                         "image embedding; patch-level methods (e.g. PatchCore) capture them better.")
            else:
                P.append("**Defect types.** With the best configuration, every defect type exceeds AUROC 0.90.")
        # 5. anomalib vs features
        anom = [c for c in self.complete if self.meta[c]["extractor"] == "anomalib"]
        feat = [c for c in self.complete if self.meta[c]["extractor"] != "anomalib"]
        if anom and feat:
            ba, bf = max(anom, key=self.mean), max(feat, key=self.mean)
            diff = self.mean(ba) - self.mean(bf)
            verb = "matches" if abs(diff) < 5e-4 else "outperforms" if diff > 0 else "falls behind"
            P.append(f"**Anomalib vs. global features.** The best anomalib model ({self.label(ba)}, {self.mean(ba):.3f}) "
                     f"{verb} the best features + detector combination ({self.label(bf)}, {self.mean(bf):.3f}; "
                     f"difference {diff:+.3f}); it also produces anomaly maps that localise the defect.")
        # 6. threshold calibration
        if "fpr@thr" in self.raw[0]:
            q = self.rows_of(self.ds_names[0], self.best)[0].get("threshold_quantile", 0.95)
            nominal = 1 - q
            bad = [(d, self.val(d, self.best, "fpr@thr")) for d in self.ds_names if self.val(d, self.best, "fpr@thr") > max(0.15, 3 * nominal)]
            tpr = np.mean([self.val(d, self.best, "tpr@thr") for d in self.ds_names])
            txt = (f"**Automatic threshold.** Calibrated on normal images only (quantile {q:g}), it detects {tpr:.0%} of the "
                   f"anomalies on average with the best configuration.")
            if bad:
                txt += (" The test FPR clearly exceeds the expected one (≈" + f"{nominal:.0%}) on "
                        + ", ".join(f"{d} ({v:.0%})" for d, v in bad)
                        + ": normal test images score higher than training ones (distribution shift or a small training set). "
                          "The ranking can still be good; recalibrating on a validation set is advisable "
                          "(`imgad calibrate --method fpr`).")
            else:
                txt += f" The test FPR stays close to the expected one (≈{nominal:.0%})."
            P.append(txt)
        # 7. pixel
        if any("pixel_auroc" in r for r in self.raw):
            pix = [c for c in self.configs if not np.isnan(self.mean(c, "pixel_auroc"))]
            if pix:
                bp = max(pix, key=lambda c: self.mean(c, "pixel_auroc"))
                gap = self.mean(bp, "pixel_auroc") - self.mean(bp, "aupro")
                P.append(f"**Localisation.** {self.label(bp)} reaches pixel-AUROC {self.mean(bp, 'pixel_auroc'):.3f} and AUPRO "
                         f"{self.mean(bp, 'aupro'):.3f}. "
                         + ("The gap between them shows that large defects dominate pixel-AUROC while small ones are segmented "
                            "worse (AUPRO weights every region equally)." if gap > 0.05 else
                            "Both metrics agree: segmentation is good for small defects too."))
        return [p + "\n" for p in P]

    def sec_limitations(self) -> List[str]:
        L = []
        if self.datasets:
            small = [n for n, d in self.datasets.items() if int((np.asarray(d.test_labels) == 0).sum()) < 30]
            if small:
                L.append(f"Small test sets ({', '.join(small)}): small differences are not conclusive.")
        L.append("The best configuration is chosen on the test set, so its performance is an optimistic estimate. An "
                 "unbiased selection needs a separate validation set.")
        L.append("Default hyperparameters, no tuning.")
        if len(self.seeds) == 1:
            L.append("A single seed: variability due to training is not measured (`--seeds` and `--train-subsample`).")
        if not any("auroc_ci_low" in r for r in self.raw):
            L.append("No confidence intervals (`--bootstrap 1000`).")
        if not any("pixel_auroc" in r for r in self.raw):
            L.append("Image-level metrics only: defect localisation is not evaluated.")
        if len(self.ds_names) < 3:
            L.append("Too few datasets to generalise the conclusions or to run powerful across-dataset tests.")
        return [f"- {x}" for x in L]

    def sec_repro(self, config, command) -> List[str]:
        L = ["Files in this folder: `report.md`, `results.csv` (one row per dataset × configuration × seed)"
             + (", `config.yaml`" if config is not None else "") + (f" and {len(self.figs)} figures in `figures/`." if self.figs else "."), ""]
        if command:
            L += ["```bash", command, "```", ""]
        if config is not None:
            import yaml

            L += ["<details><summary>Configuration</summary>", "", "```yaml",
                  yaml.safe_dump(config, sort_keys=False, allow_unicode=True).rstrip(), "```", "", "</details>", ""]
        L.append("Environment: " + ", ".join(_versions()) + ".")
        return L

    def sec_annex(self) -> List[str]:
        if len(self.ds_names) < 2:
            return []
        L = []
        has_ci = self.metric == "auroc" and any("auroc_ci_low" in r for r in self.raw)
        for d in self.ds_names:
            cs = sorted([c for c in self.configs if self.rows_of(d, c)], key=lambda c: -self.val(d, c))
            rows = []
            for c in cs:
                rs = self.rows_of(d, c)
                ci = f"[{np.mean([r['auroc_ci_low'] for r in rs]):.3f}, {np.mean([r['auroc_ci_high'] for r in rs]):.3f}]" if has_ci else None
                rows.append([md_escape(self.label(c)), _f(self.val(d, c))] + ([ci] if has_ci else [])
                            + [_f(self.val(d, c, "aupr")), _f(self.val(d, c, "f1_max")), _f(self.val(d, c, "fpr@95tpr"))])
            L += [f"<details><summary>{d}</summary>", ""]
            L += _table(["Configuration", self.mname] + (["CI"] if has_ci else []) + ["AUPR", "F1-max", "FPR@95%TPR"], rows)
            L += ["", "</details>", ""]
        return L


# ============================================================================ dataset helpers

def _dataset_info(d) -> Dict[str, Any]:
    y = np.asarray(d.test_labels)
    types = Counter(t for t, l in zip(d.test_types, y) if l == 1)
    sizes = Counter()
    for p in list(d.train_paths[:15]) + list(d.test_paths[:15]):
        try:
            from PIL import Image

            with Image.open(p) as im:
                sizes[im.size] += 1
        except Exception:
            pass
    res = "–"
    if sizes:
        (w, h), _ = sizes.most_common(1)[0]
        res = f"{w}×{h}" + (" (variable)" if len(sizes) > 1 else "")
    return {
        "n_train": len(d.train_paths),
        "n_test_normal": int((y == 0).sum()),
        "n_test_anom": int((y == 1).sum()),
        "anom_frac": float((y == 1).mean()) if len(y) else 0.0,
        "types": dict(types),
        "masks": d.has_masks,
        "resolution": res,
        "mixed_res": len(sizes) > 1,
    }


def _thumb(img, size: int = 160) -> np.ndarray:
    img = img.copy()
    img.thumbnail((size, size))
    return np.asarray(img)


def _versions() -> List[str]:
    from . import __version__

    out = [f"imgad {__version__}", f"Python {sys.version.split()[0]}"]
    for mod in ("torch", "transformers", "sklearn", "anomalib"):
        try:
            out.append(f"{mod} {__import__(mod).__version__}")
        except Exception:
            pass
    try:
        import torch

        out.append(f"GPU {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "no GPU")
    except Exception:
        pass
    out.append(platform.platform(terse=True))
    return out
