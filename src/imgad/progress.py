"""Progress reporting for long runs (benchmarks, Anomalib training, reports).

Everything goes through the standard ``logging`` module under the ``imgad`` logger, so it can be
redirected or silenced like any other library log::

    import logging, imgad
    imgad.progress.setup_logging(logging.DEBUG, log_file="run.log")   # console + file

Levels: INFO = phases and one line per finished run (with counter, elapsed time and ETA);
DEBUG = per-step timings, cache hits, bootstrap... Messages are written with ``tqdm.write`` so they
do not break the progress bars.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from tqdm.auto import tqdm

logger = logging.getLogger("imgad")


class TqdmHandler(logging.StreamHandler):
    """Console handler that prints through ``tqdm.write`` (keeps progress bars intact)."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=self.stream)
            self.flush()
        except Exception:  # pragma: no cover
            self.handleError(record)


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None, fmt: str = "%(asctime)s %(message)s") -> None:
    """Send ``imgad`` logs to the console (``level``) and optionally to ``log_file`` (always DEBUG,
    with the logger name). Calling it again replaces the previous handlers."""
    for h in list(logger.handlers):
        if getattr(h, "_imgad", False):
            logger.removeHandler(h)
            h.close()
    console = TqdmHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    console._imgad = True
    logger.addHandler(console)
    if log_file:
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
        fh._imgad = True
        logger.addHandler(fh)
    logger.setLevel(min(level, logging.DEBUG if log_file else level))
    logger.propagate = False


def ensure_console(verbose: bool) -> None:
    """For Python API calls with ``verbose=True``: show INFO messages if the user has not configured
    logging for ``imgad`` (or the root logger) themselves."""
    if verbose and not logger.handlers and not logging.getLogger().handlers:
        setup_logging(logging.INFO)


def fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


_KIND_NAMES = {"load": "model loading", "extract": "feature extraction", "detector": "detector runs",
               "anomalib": "anomalib runs"}


class BenchmarkProgress:
    """Counts the runs of a benchmark and estimates the remaining time.

    Work is split into *kinds* with very different costs (``"extract"``: features for one extractor
    and dataset; ``"detector"``: fit + score of one detector; ``"anomalib"``: training + testing of
    one Anomalib model). The ETA is the sum over kinds of (remaining units × mean observed time), so
    an expensive feature extraction does not distort the estimate for the cheap detector runs.
    """

    def __init__(self, totals: Dict[str, int]):
        self.totals = dict(totals)
        self.done: Dict[str, int] = defaultdict(int)
        self.spent: Dict[str, float] = defaultdict(float)
        self.t0 = time.time()
        self.n_runs = totals.get("detector", 0) + totals.get("anomalib", 0)
        self.runs_done = 0
        self.workers: Dict[str, int] = {}  # parallel workers per kind (the ETA divides by them)

    def add(self, kind: str, seconds: float) -> None:
        self.done[kind] += 1
        self.spent[kind] += seconds
        if kind in ("detector", "anomalib"):
            self.runs_done += 1

    @property
    def elapsed(self) -> float:
        return time.time() - self.t0

    def eta(self) -> Tuple[Optional[float], List[str]]:
        """``(seconds, kinds)``: estimated remaining time over the kinds of work already timed, and the
        kinds still pending that have not been timed yet (so the estimate excludes them)."""
        rest, untimed, timed = 0.0, [], False
        for kind, total in self.totals.items():
            left = total - self.done[kind]
            if left <= 0:
                continue
            if self.done[kind] == 0:
                untimed.append(kind)
                continue
            timed = True
            rest += left * self.spent[kind] / self.done[kind] / max(1, self.workers.get(kind, 1))
        return (rest if timed or not untimed else None), untimed

    def counter(self) -> str:
        w = len(str(self.n_runs))
        pct = 100 * self.runs_done / self.n_runs if self.n_runs else 100
        return f"[{self.runs_done:>{w}}/{self.n_runs} {pct:3.0f}%]"

    def timing(self) -> str:
        eta, untimed = self.eta()
        out = f"elapsed {fmt_duration(self.elapsed)}"
        if eta is not None:
            out += f", ETA ~{fmt_duration(eta)}"
            # model loading / feature extraction are short phases: only flag the run kinds
            pending = [_KIND_NAMES[k] for k in untimed if k in ("detector", "anomalib")]
            if pending:
                out += f" + {' and '.join(pending)} (not timed yet)"
        elif untimed:
            out += f" (ETA after the first of the {' and '.join(_KIND_NAMES[k] for k in untimed)})"
        return out


class step:
    """Context manager that logs ``msg`` at the start (INFO or DEBUG) and its duration at the end;
    on an exception it logs where the run failed before re-raising."""

    def __init__(self, msg: str, level: int = logging.DEBUG, log=logger):
        self.msg, self.level, self.log = msg, level, log

    def __enter__(self):
        self.t = time.time()
        self.log.log(self.level, f"{self.msg}...")
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - self.t
        self.seconds = dt
        if exc_type is not None:
            self.log.error(f"FAILED: {self.msg} ({type(exc).__name__}: {exc})")
            return False
        self.log.debug(f"done: {self.msg} ({fmt_duration(dt) if dt >= 1 else f'{dt:.2f}s'})")
        return False
