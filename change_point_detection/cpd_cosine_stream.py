"""

Streaming PELT for cosine similarity to the drug centroid.

The batch script (``cpd_cosine_similarity.py``) waits until every half-year is
already in a matrix. This file scores words as observations arrive. Each
observation is one word in one half-year:

    {"word": "fetty", "half_year": "2018H1", "cosine": 0.62, "n_occurrences": 400}

A whole half-year can also arrive at once:

    {"half_year": "2018H1", "observations": [
        {"word": "fetty", "cosine": 0.62, "n_occurrences": 400}
    ]}

Feed JSON lines on stdin. Alerts are JSON lines on stdout. When the stream
ends, a snapshot of every word seen so far is written to ``--out-dir``.

    python cpd_cosine_stream.py --out-dir results_cosine_stream < observations.jsonl

What "streaming" means here. PELT needs the series it is segmenting, and a
half-year series is short (about 20 points), so each new point re-fits PELT
on that word's observations so far. The fit is exact for the data in hand.
A point that arrives later, or a revised value for a half-year already seen,
can move an earlier change point. Half-years are ordered by their label, not
by arrival time, so a late 2017H2 is inserted before 2018H1.

Two different cutoffs:

  * A live alert uses that word's own permutation p-value. The other words
    are not known yet, so Benjamini-Hochberg cannot be applied to one point.
  * The snapshot at the end ranks the words seen so far and adds q-values,
    which is the same rule as the batch script. Those q-values change if
    more words arrive later.

Cells with fewer than ``--min-occurrences`` contexts are stored but left out
of the series. A word is not scored until it has ``--min-valid-periods``
usable half-years.
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd

from change_point_detection import (
    analyze_word,
    benjamini_hochberg,
    plot_words,
)

logger = logging.getLogger(__name__)

_HALF_YEAR = re.compile(r"(\d{4})H([12])$")


def half_year_key(label):
    """Sort key for a ``YYYYH1`` / ``YYYYH2`` label."""
    match = _HALF_YEAR.fullmatch(str(label).strip())
    if not match:
        raise ValueError(f"Expected a half-year like '2018H1', got {label!r}.")
    return int(match.group(1)) * 2 + int(match.group(2))


def _word_rng(seed, word):
    """Same word and seed always produce the same permutation p-value."""
    digest = hashlib.sha256(f"{seed}:{word}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


class StreamingCosinePELT:
    """Keep per-word cosine histories and re-fit PELT as points arrive."""

    def __init__(
        self,
        min_occurrences=30,
        min_valid_periods=6,
        min_size=3,
        pen_mult=3.0,
        n_perm=999,
        alpha=0.05,
        min_shift=0.0,
        noise_floor=0.001,
        seed=0,
        decreasing=False,
    ):
        if min_valid_periods < 2 * min_size:
            raise ValueError("min_valid_periods must be at least 2 * min_size.")
        self.min_occurrences = min_occurrences
        self.min_valid_periods = min_valid_periods
        self.alpha = alpha
        self.min_shift = min_shift
        self.noise_floor = noise_floor
        self.seed = seed
        self.args = SimpleNamespace(
            min_size=min_size,
            pen_mult=pen_mult,
            n_perm=n_perm,
            decreasing=decreasing,
        )
        # word -> {half_year: (cosine, n_occurrences)}
        self._points = {}
        # word -> last scored result, or None when it is not yet scoreable
        self._scored = {}
        self._flagged = set()

    def update(self, word, half_year, cosine, n_occurrences):
        """
        Ingest one observation.

        A repeated ``(word, half_year)`` replaces the previous value. Returns
        an alert dict when the live flag appears, changes, or goes away, and
        ``None`` when the call does not change the alert.
        """
        word = str(word)
        half_year = str(half_year).strip()
        half_year_key(half_year)
        cosine = float(cosine) if cosine is not None else float("nan")
        n_occurrences = int(n_occurrences)

        series_before = self._valid_series(word) if word in self._points else None
        self._points.setdefault(word, {})[half_year] = (cosine, n_occurrences)
        series_after = self._valid_series(word)
        if series_before == series_after and word in self._scored:
            return None
        return self._rescore(word, half_year)

    def update_many(self, records):
        """Ingest ``(word, half_year, cosine, n_occurrences)`` tuples. Returns alerts."""
        alerts = []
        for word, half_year, cosine, n_occurrences in records:
            alert = self.update(word, half_year, cosine, n_occurrences)
            if alert is not None:
                alerts.append(alert)
        return alerts

    def snapshot(self):
        """
        Rank every word that can currently be scored.

        ``emerging`` uses the Benjamini-Hochberg q-value across this table.
        ``live_alert`` is the per-word rule used while the stream is running.
        """
        rows = []
        for word, scored in self._scored.items():
            if scored is None:
                continue
            result, _rows, periods, _series = scored
            points = self._points[word]
            rows.append({
                **result,
                "word": word,
                "total_occurrences": int(sum(n for _c, n in points.values())),
                "first_valid_half_year": periods[0],
                "last_valid_half_year": periods[-1],
                "live_alert": word in self._flagged,
            })
        if not rows:
            return pd.DataFrame()

        results = pd.DataFrame(rows)
        results["q_value"] = benjamini_hochberg(results["p_value"])
        results["emerging"] = (
            (results["q_value"] < self.alpha)
            & (results["shift"] > max(self.min_shift, 0.0))
        )
        results = results.sort_values(
            ["emerging", "shift", "q_value"], ascending=[False, False, True]
        ).reset_index(drop=True)
        results.insert(0, "rank", np.arange(1, len(results) + 1))
        front = ["rank", "word", "emerging", "live_alert", "onset_half_year",
                 "shift", "q_value", "p_value"]
        return results[front + [c for c in results.columns if c not in front]]

    def write_snapshot(self, out_dir, n_plots=30, ylabel="cosine mean embedding",
                       signal="cosine similarity to the half-year drug centroid"):
        """Write the current ranking, change points, and plots of the top words."""
        os.makedirs(out_dir, exist_ok=True)
        results = self.snapshot()
        results.to_csv(os.path.join(out_dir, "cpd_results.csv"), index=False)

        cp_rows = []
        signal_of, cps_of, periods_of = {}, {}, {}
        for word, scored in self._scored.items():
            if scored is None:
                continue
            result, rows, periods, series = scored
            cp_rows.extend({"word": word, **row} for row in rows)
            signal_of[word] = series
            periods_of[word] = periods
            cps_of[word] = [
                periods.index(h)
                for h in (result["changepoints"].split(";") if result["changepoints"] else [])
            ]
        pd.DataFrame(cp_rows).to_csv(
            os.path.join(out_dir, "cpd_changepoints.csv"), index=False
        )

        if len(results):
            to_plot = list(results["word"].head(n_plots))
            plot_words(
                to_plot, signal_of, cps_of, periods_of,
                ylabel, os.path.join(out_dir, "plots"),
            )

        meta = {
            "script": "cpd_cosine_stream.py",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "signal": signal,
            "direction": "decrease" if self.args.decreasing else "increase",
            "live_alert": "p_value < alpha and shift > min_shift, for this word alone",
            "snapshot_emerging": "q_value < alpha and shift > min_shift, across words seen so far",
            "eligibility": {
                "min_occurrences_per_cell": self.min_occurrences,
                "min_valid_periods": self.min_valid_periods,
            },
            "pelt": {
                "model": "l2",
                "min_size": self.args.min_size,
                "penalty": f"{self.args.pen_mult} * log(n)",
                "noise_floor": self.noise_floor,
                "refit": "PELT is re-fit on the series received so far",
            },
            "significance": {
                "test": "one-sided permutation test of max standardized upward mean shift",
                "n_perm": self.args.n_perm,
                "multiple_testing_on_snapshot": "Benjamini-Hochberg",
                "alpha": self.alpha,
                "min_shift": self.min_shift,
                "seed": self.seed,
            },
            "counts": {
                "words_seen": len(self._points),
                "words_scored": int(sum(v is not None for v in self._scored.values())),
                "live_alerts": len(self._flagged),
                "snapshot_emerging": int(results["emerging"].sum()) if len(results) else 0,
            },
        }
        with open(os.path.join(out_dir, "cpd_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return results

    def _valid_series(self, word):
        usable = [
            (half_year, cosine)
            for half_year, (cosine, n) in self._points[word].items()
            if n >= self.min_occurrences and np.isfinite(cosine)
        ]
        usable.sort(key=lambda item: half_year_key(item[0]))
        return tuple(usable)

    def _rescore(self, word, arrived):
        usable = self._valid_series(word)
        previously = word in self._flagged
        if len(usable) < self.min_valid_periods:
            self._scored[word] = None
            self._flagged.discard(word)
            if previously:
                return self._alert(word, arrived, "cleared", None)
            return None

        periods = [h for h, _c in usable]
        series = np.array([c for _h, c in usable], dtype=np.float64)
        out = analyze_word(
            series, periods, self.args, self.noise_floor, _word_rng(self.seed, word)
        )
        if out is None:
            self._scored[word] = None
            self._flagged.discard(word)
            if previously:
                return self._alert(word, arrived, "cleared", None)
            return None

        result, rows = out
        self._scored[word] = (result, rows, periods, series)
        flagged = (
            result["p_value"] < self.alpha
            and result["shift"] > max(self.min_shift, 0.0)
        )
        if flagged:
            self._flagged.add(word)
            event = "emerged" if not previously else "revised"
            # A refit that leaves the onset and the shift unchanged is not news.
            if previously and not self._detection_changed(word, result):
                return None
            self._last_signature = getattr(self, "_last_signature", {})
            self._last_signature[word] = (
                result["changepoints"], round(result["shift"], 6)
            )
            return self._alert(word, arrived, event, result)
        self._flagged.discard(word)
        if previously:
            return self._alert(word, arrived, "cleared", result)
        return None

    def _detection_changed(self, word, result):
        previous = getattr(self, "_last_signature", {}).get(word)
        current = (result["changepoints"], round(result["shift"], 6))
        return previous != current

    def _alert(self, word, arrived, event, result):
        alert = {
            "word": word,
            "half_year": arrived,
            "event": event,
            "n_valid": 0 if result is None else result["n_valid"],
            "onset_half_year": "" if result is None else result["onset_half_year"],
            "shift": 0.0 if result is None else result["shift"],
            "p_value": None if result is None else result["p_value"],
            "changepoints": "" if result is None else result["changepoints"],
            "level_before": None if result is None else result["level_before"],
            "level_after": None if result is None else result["level_after"],
        }
        return alert


def iter_records(lines):
    """Yield observation tuples from JSON lines, one word or one half-year each."""
    for line_number, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Line {line_number} is not JSON: {exc}") from exc
        if "observations" in record:
            half_year = record["half_year"]
            for item in record["observations"]:
                yield (
                    item["word"], half_year, item["cosine"], item["n_occurrences"],
                )
            continue
        yield (
            record["word"], record["half_year"],
            record["cosine"], record["n_occurrences"],
        )


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Streaming PELT on cosine similarity observations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out-dir", default=os.path.join(here, "results_cosine_stream"))
    parser.add_argument("--min-occurrences", type=int, default=30)
    parser.add_argument("--min-valid-periods", type=int, default=6)
    parser.add_argument("--min-size", type=int, default=3)
    parser.add_argument("--pen-mult", type=float, default=3.0)
    parser.add_argument("--n-perm", type=int, default=999)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-shift", type=float, default=0.0)
    parser.add_argument("--noise-floor", type=float, default=0.001,
                        help="Fixed standardization floor. The batch cosine run "
                        "estimated about 0.0014; a fixed floor keeps a new word "
                        "from changing every other word's scale.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decreasing", action="store_true")
    parser.add_argument("--n-plots", type=int, default=30)
    return parser.parse_args()


def main():
    # ------------------------------------------------------------------
    # 1. Read options and build the detector. Nothing is loaded from the
    #    cosine matrix. Observations come from stdin, one JSON object per line.
    # ------------------------------------------------------------------
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    if sys.stdin.isatty():
        raise SystemExit(
            "Pipe a JSONL stream on stdin. Each line is either "
            '{"word", "half_year", "cosine", "n_occurrences"} or '
            '{"half_year", "observations": [...]}.'
        )
    detector = StreamingCosinePELT(
        min_occurrences=args.min_occurrences,
        min_valid_periods=args.min_valid_periods,
        min_size=args.min_size,
        pen_mult=args.pen_mult,
        n_perm=args.n_perm,
        alpha=args.alpha,
        min_shift=args.min_shift,
        noise_floor=args.noise_floor,
        seed=args.seed,
        decreasing=args.decreasing,
    )

    # ------------------------------------------------------------------
    # 2. Score each observation as it arrives and print an alert only when
    #    that word's live flag changes. stdout is line-buffered by the print
    #    flush, so a downstream process sees the alert immediately.
    # ------------------------------------------------------------------
    n_seen = 0
    n_alerts = 0
    for record in iter_records(sys.stdin):
        n_seen += 1
        alert = detector.update(*record)
        if alert is not None:
            n_alerts += 1
            print(json.dumps(alert), flush=True)
        if n_seen % 100000 == 0:
            logger.info("Ingested %s observations, %s alerts", f"{n_seen:,}", f"{n_alerts:,}")

    # ------------------------------------------------------------------
    # 3. The stream has ended. Rank the words received so far, apply
    #    Benjamini-Hochberg, and write the same kinds of outputs as the
    #    batch script. q-values describe this batch of words only.
    # ------------------------------------------------------------------
    results = detector.write_snapshot(args.out_dir, n_plots=args.n_plots)
    logger.info(
        "Stream ended: %s observations, %s live alerts, %s words in the snapshot. "
        "Wrote %s",
        f"{n_seen:,}", f"{n_alerts:,}", f"{len(results):,}", args.out_dir,
    )


if __name__ == "__main__":
    main()