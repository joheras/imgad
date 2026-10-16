"""Statistics for comparing configurations: bootstrap confidence intervals, paired tests on one test
set (DeLong, paired bootstrap) and tests across datasets (Friedman + Nemenyi, Wilcoxon-Holm).

Typical use, after ``rows = benchmark(..., seeds=[0, 1, 2], bootstrap=1000, keep_scores=True)``::

    from imgad import stats
    stats.summarize_seeds(rows)                       # mean, std and 95% CI over seeds
    res = stats.compare_configs(rows, metric="auroc") # ranks, Friedman, Nemenyi CD, Wilcoxon-Holm
    print(stats.report_markdown(res))
    stats.plot_cd_diagram(res, "cd.png")
    stats.delong_vs_baseline(rows, baseline=res["configs"][0])   # per-dataset paired AUROC tests

Rows are the dicts returned by :func:`imgad.evaluation.benchmark` (or read back from its CSV with
:func:`load_results`). Configurations are identified by the ``config`` column
(``"extractor | preprocessing | detector"``) and datasets by ``dataset``.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sps
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

# ----------------------------------------------------------------------------- metrics

def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUROC (ties count 1/2); much faster than sklearn inside bootstrap loops."""
    r = sps.rankdata(s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _f1_max(y, s) -> float:
    p, r, _ = precision_recall_curve(y, s)
    return float((2 * p * r / np.clip(p + r, 1e-12, None)).max())


def _fpr95(y, s) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    return float(fpr[np.searchsorted(tpr, 0.95, side="left")])


METRICS: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "auroc": _auroc,
    "aupr": lambda y, s: float(average_precision_score(y, s)),
    "f1_max": _f1_max,
    "fpr@95tpr": _fpr95,
}
LOWER_IS_BETTER = {"fpr@95tpr"}


def _stratified_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Resample normals and anomalies separately (keeps class balance; both classes always present)."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    return np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])


# ----------------------------------------------------------------------------- one test set

def bootstrap_ci(
    y_true,
    scores,
    metrics: Sequence[str] = ("auroc", "aupr"),
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Percentile bootstrap CI over **test images** (stratified by class).

    Returns ``{metric: {"value", "low", "high", "se"}}``. It reflects test-set sampling variability
    only; variability due to training (seeds / training subsets) is measured by repeating runs.
    """
    y, s = np.asarray(y_true).astype(int), np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    boots = {m: np.empty(n_boot) for m in metrics}
    for b in range(n_boot):
        idx = _stratified_indices(y, rng)
        for m in metrics:
            boots[m][b] = METRICS[m](y[idx], s[idx])
    return {
        m: {
            "value": METRICS[m](y, s),
            "low": float(np.quantile(boots[m], alpha / 2)),
            "high": float(np.quantile(boots[m], 1 - alpha / 2)),
            "se": float(boots[m].std(ddof=1)),
        }
        for m in metrics
    }


def delong_test(y_true, scores_a, scores_b, alpha: float = 0.05) -> Dict[str, float]:
    """DeLong test for two **correlated** AUROCs (same test images, two models).

    Fast O(n log n) implementation (Sun & Xu, 2014). Returns both AUROCs, their difference
    (a - b) with a ``1 - alpha`` CI, the z statistic and the two-sided p-value."""
    y = np.asarray(y_true).astype(int)
    S = np.vstack([np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)])
    pos, neg = S[:, y == 1], S[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    if m == 0 or n == 0:
        raise ValueError("delong_test needs both normal and anomalous samples")
    tx = np.apply_along_axis(sps.rankdata, 1, pos)
    ty = np.apply_along_axis(sps.rankdata, 1, neg)
    tz = np.apply_along_axis(sps.rankdata, 1, np.hstack([pos, neg]))
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    diff = float(aucs[0] - aucs[1])
    if var <= 1e-15:  # identical rankings
        z, p = 0.0, 1.0
    else:
        z = diff / math.sqrt(var)
        p = float(2 * sps.norm.sf(abs(z)))
    half = sps.norm.ppf(1 - alpha / 2) * math.sqrt(max(var, 0.0))
    return {"auroc_a": float(aucs[0]), "auroc_b": float(aucs[1]), "diff": diff,
            "diff_low": diff - half, "diff_high": diff + half, "z": float(z), "p_value": p}


def paired_bootstrap_test(
    y_true, scores_a, scores_b, metric: str = "aupr", n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> Dict[str, float]:
    """Paired bootstrap for any metric (e.g. AUPR, where DeLong does not apply): both models are
    evaluated on the same resampled test images. Two-sided p-value for ``metric(a) == metric(b)``."""
    y = np.asarray(y_true).astype(int)
    a, b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    f = METRICS[metric]
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for i in range(n_boot):
        idx = _stratified_indices(y, rng)
        d[i] = f(y[idx], a[idx]) - f(y[idx], b[idx])
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return {f"{metric}_a": f(y, a), f"{metric}_b": f(y, b), "diff": f(y, a) - f(y, b),
            "diff_low": float(np.quantile(d, alpha / 2)), "diff_high": float(np.quantile(d, 1 - alpha / 2)),
            "p_value": float(p)}


# ----------------------------------------------------------------------------- repetitions

def summarize_seeds(
    rows: List[Dict[str, Any]], metrics: Sequence[str] = ("auroc", "aupr"), alpha: float = 0.05
) -> List[Dict[str, Any]]:
    """Aggregate repeated runs: one row per (dataset, config) with mean, std, number of seeds and a
    t-based ``1 - alpha`` CI of the mean for each metric. Adds a ``dataset="mean"`` row per config
    with the across-dataset mean of the seed means."""
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], _config(r))].append(r)
    out = []
    for (ds, cfg), rs in groups.items():
        row: Dict[str, Any] = {"dataset": ds, "config": cfg, "n_seeds": len(rs)}
        for m in metrics:
            v = np.array([r[m] for r in rs if r.get(m) is not None], dtype=float)
            if not len(v):
                continue
            sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            half = float(sps.t.ppf(1 - alpha / 2, len(v) - 1) * sd / math.sqrt(len(v))) if len(v) > 1 else 0.0
            row.update({f"{m}_mean": float(v.mean()), f"{m}_std": sd, f"{m}_ci_low": float(v.mean()) - half,
                        f"{m}_ci_high": float(v.mean()) + half})
        out.append(row)
    by_cfg: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in out:
        by_cfg[r["config"]].append(r)
    for cfg, rs in by_cfg.items():
        if len(rs) > 1:
            out.append({"dataset": "mean", "config": cfg, "n_seeds": min(r["n_seeds"] for r in rs),
                        **{f"{m}_mean": float(np.mean([r[f"{m}_mean"] for r in rs if f"{m}_mean" in r])) for m in metrics}})
    return out


# ----------------------------------------------------------------------------- across datasets

def results_matrix(
    rows: List[Dict[str, Any]], metric: str = "auroc"
) -> Tuple[List[str], List[str], np.ndarray]:
    """``(datasets, configs, M)`` with ``M[i, j]`` = metric of config j on dataset i, averaged over
    seeds. Configs missing on some dataset are dropped (tests need complete blocks)."""
    acc: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for r in rows:
        if r.get(metric) is not None and r["dataset"] != "mean":
            acc[(r["dataset"], _config(r))].append(float(r[metric]))
    datasets = sorted({d for d, _ in acc})
    configs = list(dict.fromkeys(c for _, c in acc))
    configs = [c for c in configs if all((d, c) in acc for d in datasets)]
    M = np.array([[np.mean(acc[(d, c)]) for c in configs] for d in datasets], dtype=float)
    return datasets, configs, M


def holm(pvalues: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (controls the family-wise error rate)."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    """Critical difference of average ranks (Nemenyi, Demšar 2006) for k configs on n datasets."""
    q = sps.studentized_range.ppf(1 - alpha, k, np.inf) / math.sqrt(2)
    return float(q * math.sqrt(k * (k + 1) / (6 * n)))


def compare_configs(rows: List[Dict[str, Any]], metric: str = "auroc", alpha: float = 0.05) -> Dict[str, Any]:
    """Compare configurations across datasets (each dataset = one block), following Demšar (2006):

    * average ranks (1 = best; ties get the average rank),
    * Friedman test (+ Iman-Davenport F correction) of "all configs perform equally",
    * Nemenyi post-hoc: critical difference (CD) and pairwise p-values,
    * pairwise Wilcoxon signed-rank tests with Holm correction (more powerful with few configs).

    Needs >= 2 datasets; Friedman/Nemenyi need >= 3 configs. Seeds are averaged per dataset first.
    """
    datasets, configs, M = results_matrix(rows, metric)
    n, k = M.shape if M.size else (0, 0)
    if n < 2 or k < 2:
        raise ValueError(f"compare_configs needs >= 2 datasets and >= 2 complete configs (got {n} datasets, {k} configs)")
    sign = 1 if metric in LOWER_IS_BETTER else -1
    ranks = np.apply_along_axis(sps.rankdata, 1, sign * M)
    avg_rank = ranks.mean(axis=0)
    order = np.argsort(avg_rank)
    configs = [configs[i] for i in order]
    M, ranks, avg_rank = M[:, order], ranks[:, order], avg_rank[order]

    res: Dict[str, Any] = {"metric": metric, "alpha": alpha, "datasets": datasets, "configs": configs,
                           "matrix": M, "avg_rank": avg_rank, "mean": M.mean(axis=0)}
    if k >= 3:
        chi2, p = sps.friedmanchisquare(*M.T)
        ff = (n - 1) * chi2 / max(n * (k - 1) - chi2, 1e-12)
        res["friedman"] = {"chi2": float(chi2), "p_value": float(p), "iman_davenport_F": float(ff),
                           "iman_davenport_p": float(sps.f.sf(ff, k - 1, (k - 1) * (n - 1)))}
        res["cd"] = nemenyi_cd(k, n, alpha)
        se = math.sqrt(k * (k + 1) / (6 * n))
    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            d = M[:, i] - M[:, j]
            p_w = 1.0 if np.allclose(d, 0) else float(sps.wilcoxon(M[:, i], M[:, j]).pvalue)
            pair = {"a": configs[i], "b": configs[j], "mean_diff": float(d.mean()),
                    "wins": int((sign * d < 0).sum()), "ties": int(np.isclose(d, 0).sum()),
                    "losses": int((sign * d > 0).sum()), "wilcoxon_p": p_w}
            if k >= 3:
                z = abs(avg_rank[i] - avg_rank[j]) / se
                pair["nemenyi_p"] = float(min(1.0, sps.studentized_range.sf(z * math.sqrt(2), k, np.inf)))
            pairs.append(pair)
    for pair, adj in zip(pairs, holm([p["wilcoxon_p"] for p in pairs])):
        pair["wilcoxon_p_holm"] = float(adj)
        pair["significant"] = bool(adj < alpha)
    res["pairs"] = pairs
    return res


def delong_vs_baseline(rows: List[Dict[str, Any]], baseline: str, alpha: float = 0.05) -> List[Dict[str, Any]]:
    """Per (dataset, seed): DeLong test of every config against ``baseline`` (needs rows produced
    with ``keep_scores=True``). p-values are Holm-adjusted within each dataset and seed."""
    by_key: Dict[Tuple[str, Any], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        if "scores" not in r:
            raise ValueError("delong_vs_baseline needs per-image scores: run benchmark(..., keep_scores=True)")
        by_key[(r["dataset"], r.get("seed", 0))][_config(r)] = r
    out = []
    for (ds, seed), cfgs in by_key.items():
        if baseline not in cfgs:
            continue
        base = cfgs[baseline]
        block = []
        for cfg, r in cfgs.items():
            if cfg == baseline:
                continue
            t = delong_test(r["labels"], r["scores"], base["scores"], alpha)
            block.append({"dataset": ds, "seed": seed, "config": cfg, "baseline": baseline, **t})
        for b, adj in zip(block, holm([b["p_value"] for b in block])):
            b["p_holm"] = float(adj)
            b["significant"] = bool(adj < alpha)
        out += block
    return out


# ----------------------------------------------------------------------------- reporting

def report_markdown(res: Dict[str, Any], max_pairs: Optional[int] = None) -> str:
    """Markdown summary of :func:`compare_configs`."""
    m, n, k = res["metric"], len(res["datasets"]), len(res["configs"])
    lines = [f"### Comparison of configurations ({m}, {n} datasets, {k} configurations)", "",
             f"| mean rank | mean {m} | configuration |", "|---|---|---|"]
    lines += [f"| {r:.2f} | {v:.4f} | {md_escape(c)} |" for c, r, v in zip(res["configs"], res["avg_rank"], res["mean"])]
    if "friedman" in res:
        f = res["friedman"]
        lines += ["", f"Friedman: χ²={f['chi2']:.3f}, p={f['p_value']:.3g} · Iman-Davenport: "
                      f"F={f['iman_davenport_F']:.3f}, p={f['iman_davenport_p']:.3g} · "
                      f"Nemenyi CD (α={res['alpha']}) = {res['cd']:.3f}"]
    pairs = res["pairs"] if max_pairs is None else res["pairs"][:max_pairs]
    lines += ["", f"| A | B | mean Δ{m} | W/T/L | Wilcoxon p (Holm) | Nemenyi p |", "|---|---|---|---|---|---|"]
    for p in pairs:
        star = " *" if p["significant"] else ""
        nem = f"{p['nemenyi_p']:.3g}" if "nemenyi_p" in p else "-"
        lines.append(f"| {md_escape(p['a'])} | {md_escape(p['b'])} | {p['mean_diff']:+.4f} | {p['wins']}/{p['ties']}/{p['losses']} | "
                     f"{p['wilcoxon_p_holm']:.3g}{star} | {nem} |")
    lines += ["", "W/T/L: datasets where A wins/ties/loses against B. * = significant after Holm."]
    return "\n".join(lines)


def plot_cd_diagram(res: Dict[str, Any], path: Optional[str] = None, title: Optional[str] = None):
    """Critical-difference diagram (Demšar 2006): average ranks on an axis; configs joined by a bar
    are not significantly different (Nemenyi). Needs matplotlib. Returns the figure."""
    import matplotlib

    if path is not None:
        matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    if "cd" not in res:
        raise ValueError("CD diagram needs >= 3 configurations")
    names, ranks, cd, k = res["configs"], np.asarray(res["avg_rank"]), res["cd"], len(res["configs"])
    half = (k + 1) // 2
    fig_h = 1.6 + 0.32 * half
    fig, ax = plt.subplots(figsize=(9, fig_h))
    lo, hi = 1, k
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.axis("off")
    top = fig_h - 0.55
    ax.hlines(top, lo, hi, color="black", lw=1)
    for t in range(lo, hi + 1):
        ax.vlines(t, top, top + 0.1, color="black", lw=1)
        ax.text(t, top + 0.15, str(t), ha="center", va="bottom", fontsize=9)
    # CD bar
    ax.hlines(top + 0.45, lo, lo + cd, color="black", lw=2)
    ax.text(lo + cd / 2, top + 0.5, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=9)
    # labels: best half left, rest right
    for i, (name, r) in enumerate(zip(names, ranks)):
        left = i < half
        y = top - 0.35 - 0.3 * (i if left else k - 1 - i)
        x_text = lo - 0.45 if left else hi + 0.45
        ax.plot([r, r, x_text], [top, y, y], color="0.3", lw=0.8)
        ax.text(x_text, y, f"{name} ({r:.2f})", ha="right" if left else "left", va="center", fontsize=8)
    # cliques: maximal groups whose rank range < CD
    cliques = []
    for i in range(k):
        j = max(jj for jj in range(i, k) if ranks[jj] - ranks[i] < cd)
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    for c, (i, j) in enumerate(cliques):
        y = top - 0.12 - 0.08 * c
        ax.hlines(y, ranks[i] - 0.03, ranks[j] + 0.03, color="black", lw=3)
    ax.set_ylim(top - 0.35 - 0.3 * half, top + 0.8)
    ax.set_title(title or f"Critical difference ({res['metric']}, α={res['alpha']})", fontsize=10)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def md_escape(text: Any) -> str:
    """Escape ``|`` so config names (``"extractor | prep | detector"``) fit in markdown tables."""
    return str(text).replace("|", "\\|")


def load_results(path: str) -> List[Dict[str, Any]]:
    """Read a benchmark CSV back, converting numeric columns."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({k: _num(v) for k, v in r.items()})
    return rows


def _num(v: str):
    if v is None or v == "":
        return None
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def _config(r: Dict[str, Any]) -> str:
    return r.get("config") or f"{r.get('extractor', '-')} | {r.get('preprocessing', '-')} | {r.get('detector', '-')}"
