#!/usr/bin/env python3
"""
Masked prediction shift analysis (script version of masked_prediction.ipynb).

Run from anywhere:
  cd /path/to/Emerging-and-New-Euphemism-Detection && python masked_prediction.py

Or:
  python /path/to/masked_prediction.py --second-pass second_pass_ccnews_top200.jsonl

Requires: pip install torch transformers pandas matplotlib
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Helpers (same as notebook)
# ---------------------------------------------------------------------------


def load_taboo_vocab(path: Path) -> set[str]:
    ns: dict = {}
    with open(path, "r") as f:
        exec(f.read(), ns)
    anchors = ns["TABOO_ANCHORS"]
    out: set[str] = set()
    for _cat, words in anchors.items():
        for w in words:
            w = str(w).strip().lower()
            if w:
                out.add(w)
                for part in re.split(r"[\s\-]+", w):
                    if len(part) > 2:
                        out.add(part)
    return out


def parse_timestamp(raw: str) -> datetime | None:
    if not raw or not str(raw).strip():
        return None
    s = str(raw).strip()
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d"):
        try:
            return datetime.strptime(s[:19], fmt) if len(s) >= 10 else datetime.strptime(s, fmt)
        except ValueError:
            continue
    m = re.search(r"(20\d{2}|19\d{2})", s)
    if m:
        return datetime(int(m.group(1)), 1, 1)
    return None


def bucket_label(dt: datetime, mode: str) -> str:
    if mode == "decade":
        d = dt.year - (dt.year % 10)
        return f"{d}s"
    if mode == "halfyear":
        # Jan–Jun = H1, Jul–Dec = H2
        half = "H1" if dt.month <= 6 else "H2"
        return f"{dt.year}-{half}"
    return str(dt.year)


def period_sort_key(period: object) -> tuple:
    """Chronological sort: year, decade (2010s), and half-year (2016-H1) labels."""
    s = str(period).strip()
    if re.fullmatch(r"\d{4}", s):
        return (0, int(s), 0)
    m_half = re.fullmatch(r"(\d{4})-([Hh][12])", s)
    if m_half:
        y = int(m_half.group(1))
        h = 0 if m_half.group(2).upper() == "H1" else 1
        return (0, y, h)
    m = re.fullmatch(r"(\d{4})s", s, re.IGNORECASE)
    if m:
        return (0, int(m.group(1)), 0)
    m2 = re.search(r"(?:19|20)\d{2}", s)
    if m2:
        return (0, int(m2.group(0)), 0)
    return (1, s, 0)


def prediction_hits_taboo(decoded_topk_strings: list[str], taboo_vocab: set[str]) -> bool:
    for s in decoded_topk_strings:
        for word in re.findall(r"[a-zA-Z]+", s.lower()):
            if word in taboo_vocab:
                return True
    return False


def mask_phrase_spans(
    tokenizer,
    sentence: str,
    char_start: int,
    char_end: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    enc = tokenizer(
        sentence,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=True,
        max_length=512,
    )
    input_ids = enc["input_ids"].to(device).clone()
    off = enc["offset_mapping"][0]
    off_list = off.tolist() if hasattr(off, "tolist") else list(off)
    mask_id = tokenizer.mask_token_id
    masked_positions: list[int] = []
    for i, (s, e) in enumerate(off_list):
        if s == 0 and e == 0:
            continue
        if s >= char_end:
            break
        if e <= char_start:
            continue
        input_ids[0, i] = mask_id
        masked_positions.append(i)
    if not masked_positions:
        return input_ids, []
    return input_ids, masked_positions


@torch.inference_mode()
def topk_at_masked_positions(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    masked_positions: list[int],
    k: int,
) -> list[str]:
    out: list[str] = []
    if not masked_positions:
        return out
    logits = model(input_ids).logits[0]
    for pos in masked_positions:
        _, topi = torch.topk(logits[pos], k=min(k, logits.shape[-1]))
        for tid in topi.tolist():
            tok = tokenizer.decode([tid]).strip()
            if tok:
                out.append(tok)
    return out


def row_taboo_hit(
    model,
    tokenizer,
    taboo_vocab: set[str],
    sentence: str,
    c0: int,
    c1: int,
    k: int,
    device: torch.device,
) -> bool:
    input_ids, positions = mask_phrase_spans(tokenizer, sentence, c0, c1, device)
    if not positions:
        return False
    toks = topk_at_masked_positions(model, tokenizer, input_ids, positions, k)
    return prediction_hits_taboo(toks, taboo_vocab)


def process_row_batch(
    rows_batch: list[tuple[int, dict]],
    model,
    tokenizer,
    taboo_vocab: set[str],
    top_k: int,
    device: torch.device,
) -> list[dict]:
    """Process a batch of rows in a worker thread. Returns list of result dicts."""
    results = []
    for idx, row in rows_batch:
        sent = str(row["sentence"])
        c0, c1 = int(row["char_offset_start"]), int(row["char_offset_end"])
        hit = row_taboo_hit(model, tokenizer, taboo_vocab, sent, c0, c1, top_k, device)
        results.append(
            {
                "canonical_phrase": row["canonical_phrase"],
                "period": row["period"],
                "taboo_in_topk": hit,
                "primary_category": row.get("primary_category", ""),
            }
        )
    return results

def load_second_pass(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # Handle formats:
            # 1. second_pass: sentence, canonical_phrase, char_offset_start/end
            # 2. collect_instances JSONL: context, word (and/or text), timestamp, category
            if "sentence" in o and "canonical_phrase" in o:
                row = dict(o)
                row.setdefault("timestamp", "")
                rows.append(row)
            elif "context" in o:
                phrase = (o.get("text") or o.get("word") or "")
                phrase = str(phrase).strip()
                if not phrase:
                    continue
                sentence = str(o["context"])
                start_idx = sentence.lower().find(phrase.lower())
                if start_idx == -1:
                    continue
                end_idx = start_idx + len(phrase)
                rows.append(
                    {
                        "sentence": sentence,
                        "canonical_phrase": phrase,
                        "char_offset_start": start_idx,
                        "char_offset_end": end_idx,
                        "timestamp": o.get("timestamp", ""),
                        "primary_category": o.get("category", ""),
                    }
                )
            else:
                continue

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Masked LM taboo top-k shift (second-pass JSONL)")
    p.add_argument(
        "--second-pass",
        type=Path,
        default=root / "second_pass_ccnews_top300.jsonl",
        help="Second-pass JSONL path",
    )
    p.add_argument(
        "--taboo",
        type=Path,
        default=root / "taboo_words_refined.py",
        help="taboo_words_refined.py path",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=root / "masked_prediction_outputs",
        help="Directory for CSV, JSON, PNG",
    )
    p.add_argument("--no-save-artifacts", action="store_true", help="Print only; do not write files")
    p.add_argument("--model", default="bert-base-uncased")
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--max-instances", type=int, default=2000, help="Subsample cap; 0 = no cap")
    p.add_argument("--max-per-group", type=int, default=50, help="Per (phrase, period); 0 = no cap")
    p.add_argument(
        "--time-bucket",
        choices=("year", "halfyear", "decade"),
        default="year",
        help="halfyear: Jan–Jun vs Jul–Dec per year (labels YYYY-H1 / YYYY-H2)",
    )
    p.add_argument("--log-every", type=int, default=100, help="Progress every N rows; 0 = quiet")
    p.add_argument("--min-n-per-period", type=int, default=3, help="Plot: min rows per period to include phrase")
    p.add_argument("--no-plot", action="store_true", help="Skip matplotlib figure")
    p.add_argument("--task-id", type=int, default=0, help="(Array job) Task ID for this job")
    p.add_argument("--num-tasks", type=int, default=1, help="(Array job) Total number of parallel tasks")
    p.add_argument("--enable-multiprocessing", action="store_true", help="Use multiprocessing for row batches")
    p.add_argument("--num-workers", type=int, default=4, help="Number of worker threads (if multiprocessing enabled)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    second_pass_jsonl: Path = args.second_pass
    taboo_words_py: Path = args.taboo
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_artifacts = not args.no_save_artifacts

    model_name = args.model
    top_k = args.top_k
    max_instances = args.max_instances if args.max_instances > 0 else None
    max_per_group = args.max_per_group if args.max_per_group > 0 else None
    time_bucket = args.time_bucket
    log_every = args.log_every if args.log_every > 0 else None
    
    # Array job sharding
    task_id = args.task_id
    num_tasks = args.num_tasks
    enable_multiproc = args.enable_multiprocessing
    num_workers = args.num_workers

    taboo_vocab = load_taboo_vocab(taboo_words_py)
    print(f"Loaded {len(taboo_vocab)} taboo-related vocabulary items (with word parts).")

    if not second_pass_jsonl.is_file():
        print(f"error: missing second-pass file: {second_pass_jsonl}", file=sys.stderr)
        return 1

    df_raw = load_second_pass(second_pass_jsonl)
    if "timestamp" not in df_raw.columns:
        df_raw["timestamp"] = ""
    print("=== Input file ===")
    print(f"  Path: {second_pass_jsonl}")
    print(f"  Rows (JSONL lines): {len(df_raw)}")
    if len(df_raw) == 0:
        print(
            "error: parsed 0 rows. JSONL must be second-pass "
            "(sentence + canonical_phrase + char offsets) or collect_instances "
            "(context + word or text, optional timestamp).",
            file=sys.stderr,
        )
        return 1

    # Shard rows across array tasks
    if num_tasks > 1:
        df_raw = df_raw[df_raw.index % num_tasks == task_id].copy()
        print(f"  Task {task_id}/{num_tasks}: processing {len(df_raw)} rows (sharded)")
        if len(df_raw) == 0:
            print("  No rows assigned to this task; exiting without loading the model.")
            return 0

    df_raw["_dt"] = df_raw["timestamp"].map(parse_timestamp)
    missing_ts = int(df_raw["_dt"].isna().sum())
    df = df_raw[df_raw["_dt"].notna()].copy()
    df["period"] = df["_dt"].map(lambda d: bucket_label(d, time_bucket))

    print("\n=== After timestamp parsing ===")
    print(f"  Rows with missing/unparseable timestamp (dropped): {missing_ts}")
    print(f"  Rows retained: {len(df)}")
    if len(df):
        print(f"  Period range: {df['period'].min()} – {df['period'].max()}")
        print(f"  Unique canonical_phrase: {df['canonical_phrase'].nunique()}")

    n_after_ts = len(df)
    if max_instances is not None and len(df) > max_instances:
        df = df.sample(n=max_instances, random_state=42)
        print(f"\n  Subsampled to max_instances={max_instances} (was {n_after_ts})")
    n_after_subsample = len(df)
    if max_per_group is not None:
        df = df.groupby(["canonical_phrase", "period"], group_keys=False).head(max_per_group)
        print(
            f"  After max_per_group={max_per_group} per (phrase, period): "
            f"{len(df)} rows (was {n_after_subsample})"
        )

    if len(df) == 0:
        print("error: no rows left after filtering.", file=sys.stderr)
        return 1

    print("\n=== Dataframe used for masking (head) ===")
    print(df[["canonical_phrase", "period", "sentence"]].head(3).to_string())

    load_stats = {
        "run_stamp": run_stamp,
        "task_id": task_id,
        "num_tasks": num_tasks,
        "second_pass_jsonl": str(second_pass_jsonl),
        "time_bucket": time_bucket,
        "max_instances": max_instances,
        "max_per_group": max_per_group,
        "rows_loaded": int(len(df_raw)),
        "rows_missing_timestamp": missing_ts,
        "rows_after_time_filter": int(n_after_ts),
        "rows_after_subsample_cap": int(n_after_subsample),
        "rows_final": int(len(df)),
        "unique_canonical_phrases": int(df["canonical_phrase"].nunique()),
        "periods": {str(k): int(v) for k, v in df["period"].value_counts().items()},
    }
    if save_artifacts:
        lp = output_dir / f"masked_pred_load_stats_task_{task_id}_{run_stamp}.json"
        with open(lp, "w", encoding="utf-8") as f:
            json.dump(load_stats, f, indent=2)
        print(f"\nSaved load stats -> {lp}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n=== Model ===")
    print(f"  device: {device}")
    print(f"  model: {model_name}  top_k={top_k}")
    if enable_multiproc:
        print(f"  multiprocessing: {num_workers} workers")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    model.eval()

    records: list[dict] = []
    n_total = len(df)

    # Single-threaded (original)
    if not enable_multiproc:
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            if log_every and i % log_every == 0:
                print(f"  masking progress: {i }/{n_total}")
            sent = str(row["sentence"])
            c0, c1 = int(row["char_offset_start"]), int(row["char_offset_end"])
            hit = row_taboo_hit(
                model, tokenizer, taboo_vocab, sent, c0, c1, top_k, device
            )
            records.append(
                {
                    "canonical_phrase": row["canonical_phrase"],
                    "period": row["period"],
                    "taboo_in_topk": hit,
                    "primary_category": row.get("primary_category", ""),
                }
            )
    
    # Multithreaded (batches)
    else:
        batch_size = max(1, n_total // (num_workers * 4))
        batches = []
        for i in range(0, n_total, batch_size):
            batch_df = df.iloc[i : i + batch_size]
            batch = [(idx, row.to_dict()) for idx, (_, row) in enumerate(batch_df.iterrows())]
            batches.append(batch)
        
        print(f"  Processing {len(batches)} batches with {num_workers} workers...")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    process_row_batch, batch, model, tokenizer, taboo_vocab, top_k, device
                ): i
                for i, batch in enumerate(batches)
            }
            completed = 0
            for future in as_completed(futures):
                batch_results = future.result()
                records.extend(batch_results)
                completed += 1
                if log_every and completed % max(1, len(futures) // 10) == 0:
                    print(f"  batches completed: {completed}/{len(batches)}")

    print(f"  done: {n_total} sentences processed")

    res = pd.DataFrame(records)
    agg = (
        res.groupby(["canonical_phrase", "period"], as_index=False)
        .agg(taboo_rate=("taboo_in_topk", "mean"), n=("taboo_in_topk", "size"))
        .sort_values(["canonical_phrase", "period"])
    )

    overall_rate = float(res["taboo_in_topk"].mean()) if len(res) else 0.0
    print("\n=== Masked LM: taboo token in top-k (lexicon overlap) ===")
    print(f"  Overall fraction of contexts with ≥1 hit: {overall_rate:.4f}  (n={len(res)})")
    by_period = res.groupby("period", as_index=False).agg(
        hit_rate=("taboo_in_topk", "mean"),
        n=("taboo_in_topk", "size"),
    )
    print("\n  By period:")
    print(by_period.to_string(index=False))
    phrase_stats = (
        res.groupby("canonical_phrase")
        .agg(hit_rate=("taboo_in_topk", "mean"), n=("taboo_in_topk", "size"))
        .query("n >= 5")
        .sort_values("hit_rate", ascending=False)
        .head(15)
    )
    print("\n  Top phrases by mean hit rate (min n≥5):")
    if len(phrase_stats):
        print(phrase_stats.to_string())
    else:
        print("  (no phrase with n≥5 — lower threshold or use more data)")
    print("\n  agg (first 25 rows):")
    print(agg.head(25).to_string(index=False))

    run_summary = {
        "run_stamp": run_stamp,
        "task_id": task_id,
        "num_tasks": num_tasks,
        "model_name": model_name,
        "top_k": top_k,
        "device": str(device),
        "n_instances": int(len(res)),
        "overall_taboo_hit_rate": overall_rate,
        "by_period": by_period.to_dict(orient="records"),
    }
    if save_artifacts:
        res_path = output_dir / f"masked_pred_instances_task_{task_id}_{run_stamp}.csv"
        agg_path = output_dir / f"masked_pred_agg_task_{task_id}_{run_stamp}.csv"
        sum_path = output_dir / f"masked_pred_run_summary_task_{task_id}_{run_stamp}.json"
        res.to_csv(res_path, index=False)
        agg.to_csv(agg_path, index=False)
        with open(sum_path, "w", encoding="utf-8") as f:
            json.dump(run_summary, f, indent=2)
        print(f"\nSaved task outputs -> {res_path.name}, {agg_path.name}, {sum_path.name} under {output_dir}")

    # Only task 0 makes the plot (after merging, you can regenerate)
    if not args.no_plot and task_id == 0:
        min_n = args.min_n_per_period
        wide = agg.pivot(index="period", columns="canonical_phrase", values="taboo_rate")
        counts = res.groupby(["canonical_phrase", "period"]).size().unstack(fill_value=0)

        period_order = sorted({str(x) for x in wide.index}, key=period_sort_key)
        wide = wide.reindex(period_order)
        counts = counts.reindex(columns=period_order, fill_value=0)

        # Debug: print shapes and columns
        print(f"  wide shape: {wide.shape}, columns: {list(wide.columns)}")
        print(f"  counts shape: {counts.shape}, index: {list(counts.index)}, columns: {list(counts.columns)}")

        phrases_ok = []
        for c in wide.columns:
            try:
                if c in counts.index:
                    period_counts = counts.loc[c]
                    if (period_counts >= min_n).sum() >= 2:
                        phrases_ok.append(c)
            except (KeyError, IndexError):
                print(f"  Warning: Could not check counts for phrase '{c}'")
                continue

        if not phrases_ok:
            phrases_ok = list(wide.columns)[:8]

        print("\n=== Plot (task 0 only) ===")
        print(f"  min_n_per_period={min_n}  phrases (up to 12): {list(phrases_ok[:12])}")

        x_pos = {p: i for i, p in enumerate(period_order)}
        fig, ax = plt.subplots(figsize=(10, 5))
        for phrase in phrases_ok[:12]:
            ser = wide[phrase].reindex(period_order)
            y = ser.dropna()
            if y.empty:
                continue
            xs = [x_pos[str(i)] for i in y.index]
            ax.plot(xs, y.values, marker="o", label=phrase[:40])
        ax.set_xticks(range(len(period_order)))
        ax.set_xticklabels(period_order, rotation=45, ha="right")
        ax.set_xlabel("Period")
        ax.set_ylabel(f"Fraction of contexts with taboo token in top-{top_k}")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        ax.set_title("Masked LM: taboo-related predictions over time (by candidate)")
        plt.tight_layout()
        if save_artifacts:
            fig_path = output_dir / f"masked_pred_plot_{run_stamp}.png"
            plt.savefig(fig_path, dpi=150, bbox_inches="tight")
            print(f"  Saved figure -> {fig_path}")
        plt.close(fig)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
