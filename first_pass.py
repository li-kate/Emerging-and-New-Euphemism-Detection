"""
FIRST PASS: EUPHEMISM CANDIDATE DETECTION PIPELINE

Scans corpora for words/phrases semantically close to taboo anchors
but NOT the taboo words themselves. Outputs candidates with context,
scores, and timestamps for second-pass collection.

Depends on: data_sources.py, taboo_words_refined.py
"""

import torch
import faiss
import numpy as np
import json
import logging
import re
import os
import itertools
from dataclasses import dataclass, field, asdict
from typing import Iterator, Optional
from sentence_transformers import SentenceTransformer

from data_sources import TextRecord, SourceConfig, batch_records

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# 0. CONFIGURATION
# ============================================================================

@dataclass
class PipelineConfig:
    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    anchor_template: str = "none"           # "none" | "taboo" | "euphemism"
    context_window_size: Optional[int] = None  # None = full sentence, or ±N words
    window_prefilter_threshold: float = 0.45
    faiss_k: int = 5
    use_gpu_faiss: bool = True
    global_threshold: float = 0.65
    use_per_category_thresholds: bool = False
    phrase_method: str = "perturbation"     # "perturbation" | "contrastive_masking"
    max_ngram_size: int = 4
    min_drop_threshold: float = 0.05
    sentence_batch_size: int = 256
    output_path: str = "first_pass_results.jsonl"


CATEGORY_THRESHOLDS = {
    "drugs": 0.60, "alcohol": 0.60, "sex": 0.60, "death": 0.55,
    "health": 0.60, "menstruation": 0.55, "aging": 0.70,
    "body_functions": 0.55, "bodily_fluids": 0.65, "toilet": 0.55,
    "physical_injury": 0.60, "intelligence": 0.70, "emotions": 0.65,
    "sexual_orientation": 0.60, "identity": 0.65, "family": 0.60,
    "money_class": 0.65, "housing": 0.60, "employment": 0.60,
    "education": 0.65, "crime": 0.60, "politics": 0.65,
    "military": 0.60, "racism": 0.60, "migration": 0.60,
    "social_conflict": 0.60, "violence": 0.55, "technology": 0.65,
    "environment": 0.65, "morality": 0.65, "disabilities": 0.60,
    "mental_health": 0.60, "religion": 0.55, "weight_appearance": 0.65,
    "hygiene": 0.60, "profanity": 0.50, "deception": 0.60,
    "cosmetic": 0.60, "rejection": 0.65,
}


# ============================================================================
# 1. TABOO ANCHORS
# ============================================================================

def load_taboo_anchors(path: str = "taboo_words_refined.py") -> dict[str, list[str]]:
    namespace = {}
    with open(path, "r") as f:
        exec(f.read(), namespace)
    return namespace["TABOO_ANCHORS"]


@dataclass
class AnchorData:
    texts: list[str]
    labels: list[str]
    categories: list[str]
    embeddings: np.ndarray = field(default=None, repr=False)

    @property
    def size(self) -> int:
        return len(self.labels)


def prepare_anchors(taboo_dict: dict[str, list[str]], template: str = "none") -> AnchorData:
    texts, labels, categories = [], [], []
    for category, words in taboo_dict.items():
        for word in words:
            labels.append(word)
            categories.append(category)
            if template == "taboo":
                texts.append(f"the taboo topic of {word}")
            elif template == "euphemism":
                texts.append(f"a euphemism for {word}")
            else:
                texts.append(word)
    return AnchorData(texts=texts, labels=labels, categories=categories)


# ============================================================================
# 2. EXCLUSION LOGIC (multi-word anchor aware)
# ============================================================================

_exclusion_patterns: dict[str, re.Pattern] = {}

def _is_direct_reference(anchor_label: str, sentence: str) -> bool:
    if anchor_label not in _exclusion_patterns:
        words = anchor_label.lower().split()
        pattern_str = r"\b" + r"\s+".join(re.escape(w) for w in words)
        _exclusion_patterns[anchor_label] = re.compile(pattern_str, re.IGNORECASE)
    return bool(_exclusion_patterns[anchor_label].search(sentence))


# ============================================================================
# 3. FAISS INDEX
# ============================================================================

def build_faiss_index(embeddings: np.ndarray, use_gpu: bool = True) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    if use_gpu and torch.cuda.is_available():
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            logger.info("FAISS index on GPU")
        except Exception as e:
            logger.warning(f"GPU FAISS failed, CPU fallback: {e}")
    index.add(embeddings)
    logger.info(f"FAISS index: {embeddings.shape[0]} anchors, dim={dim}")
    return index


# ============================================================================
# 4. CONTEXT WINDOWS
# ============================================================================

@dataclass
class WindowedChunk:
    text: str
    center_word: str
    center_index: int
    full_sentence: str
    source_record: TextRecord


def generate_windows(sentence: str, record: TextRecord, window_size: int) -> list[WindowedChunk]:
    words = sentence.split()
    chunks = []
    for i, word in enumerate(words):
        start = max(0, i - window_size)
        end = min(len(words), i + window_size + 1)
        chunks.append(WindowedChunk(
            text=" ".join(words[start:end]),
            center_word=word, center_index=i,
            full_sentence=sentence, source_record=record,
        ))
    return chunks


# ============================================================================
# 5. PHRASE EXTRACTION
# ============================================================================

def extract_phrase_perturbation(sentence, anchor_vec, model, max_n=4, min_drop=0.05):
    words = sentence.split()
    if len(words) < 2:
        return None
    baseline_vec = model.encode(sentence, convert_to_numpy=True).reshape(1, -1)
    baseline_vec /= np.linalg.norm(baseline_vec) + 1e-10
    baseline_score = float(np.dot(baseline_vec, anchor_vec.T).squeeze())

    candidates = []
    for n in range(1, min(max_n + 1, len(words) + 1)):
        perturbed, ngrams = [], []
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i:i + n])
            p = " ".join(words[:i] + words[i + n:])
            if p.strip():
                ngrams.append(ngram)
                perturbed.append(p)
        if not perturbed:
            continue
        vecs = model.encode(perturbed, convert_to_numpy=True, batch_size=64)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        vecs /= norms
        scores = np.dot(vecs, anchor_vec.T).squeeze()
        if scores.ndim == 0:
            scores = np.array([float(scores)])
        for ngram, score in zip(ngrams, scores):
            drop = baseline_score - float(score)
            if drop > min_drop:
                candidates.append({"phrase": ngram, "drop": drop, "ngram_size": n})

    if not candidates:
        return None
    candidates.sort(key=lambda x: (x["drop"], x["ngram_size"]), reverse=True)
    best = candidates[0]
    for c in candidates[1:]:
        if (c["ngram_size"] > best["ngram_size"]
                and best["phrase"] in c["phrase"]
                and abs(c["drop"] - best["drop"]) < 0.05):
            best = c
            break
    return best


def extract_phrase_contrastive_masking(sentence, anchor_vec, model, max_n=4, min_drop=0.05, mask_token="[MASK]"):
    words = sentence.split()
    if len(words) < 2:
        return None
    baseline_vec = model.encode(sentence, convert_to_numpy=True).reshape(1, -1)
    baseline_vec /= np.linalg.norm(baseline_vec) + 1e-10
    baseline_score = float(np.dot(baseline_vec, anchor_vec.T).squeeze())

    candidates = []
    for n in range(1, min(max_n + 1, len(words) + 1)):
        masked, ngrams = [], []
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i:i + n])
            m = words[:i] + [mask_token] + words[i + n:]
            masked.append(" ".join(m))
            ngrams.append(ngram)
        if not masked:
            continue
        vecs = model.encode(masked, convert_to_numpy=True, batch_size=64)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        vecs /= norms
        scores = np.dot(vecs, anchor_vec.T).squeeze()
        if scores.ndim == 0:
            scores = np.array([float(scores)])
        for ngram, score in zip(ngrams, scores):
            drop = baseline_score - float(score)
            if drop > min_drop:
                candidates.append({"phrase": ngram, "drop": drop, "ngram_size": n})

    if not candidates:
        return None
    candidates.sort(key=lambda x: (x["drop"], x["ngram_size"]), reverse=True)
    best = candidates[0]
    for c in candidates[1:]:
        if (c["ngram_size"] > best["ngram_size"]
                and best["phrase"] in c["phrase"]
                and abs(c["drop"] - best["drop"]) < 0.05):
            best = c
            break
    return best


def extract_phrase(sentence, anchor_vec, model, config):
    if config.phrase_method == "contrastive_masking":
        return extract_phrase_contrastive_masking(
            sentence, anchor_vec, model,
            max_n=config.max_ngram_size, min_drop=config.min_drop_threshold)
    return extract_phrase_perturbation(
        sentence, anchor_vec, model,
        max_n=config.max_ngram_size, min_drop=config.min_drop_threshold)


# ============================================================================
# 6. OUTPUT STRUCTURE
# ============================================================================

@dataclass
class EuphemismCandidate:
    text: str
    context: str
    taboo_anchor: str
    taboo_category: str
    similarity_score: float
    phrase_extraction_method: str = ""
    phrase_drop_score: float = 0.0
    timestamp: str = ""
    source_url: str = ""
    source: str = ""
    context_window_size: Optional[int] = None
    all_anchor_matches: list = field(default_factory=list)


# ============================================================================
# 7. CORE FIRST PASS
# ============================================================================

def _get_threshold(category: str, config: PipelineConfig) -> float:
    if config.use_per_category_thresholds:
        return CATEGORY_THRESHOLDS.get(category, config.global_threshold)
    return config.global_threshold


def first_pass(data_stream, model, anchors, faiss_index, config):
    total_processed = 0
    total_hits = 0

    for batch in batch_records(data_stream, config.sentence_batch_size):
        sentences = [r.text for r in batch]
        batch_vecs = model.encode(
            sentences, convert_to_numpy=True, batch_size=config.sentence_batch_size,
        ).astype("float32")
        faiss.normalize_L2(batch_vecs)

        if config.context_window_size is None:
            scores, indices = faiss_index.search(batch_vecs, config.faiss_k)
            for i, record in enumerate(batch):
                yield from _process_hits(
                    record.text, record, scores[i], indices[i],
                    anchors, model, config)
        else:
            pre_scores, _ = faiss_index.search(batch_vecs, 1)
            for i, record in enumerate(batch):
                if pre_scores[i][0] < config.window_prefilter_threshold:
                    continue
                windows = generate_windows(record.text, record, config.context_window_size)
                win_texts = [w.text for w in windows]
                if not win_texts:
                    continue
                win_vecs = model.encode(win_texts, convert_to_numpy=True, batch_size=64).astype("float32")
                faiss.normalize_L2(win_vecs)
                win_scores, win_indices = faiss_index.search(win_vecs, config.faiss_k)
                seen = set()
                for j, window in enumerate(windows):
                    for rank in range(config.faiss_k):
                        score = float(win_scores[j][rank])
                        anchor_idx = int(win_indices[j][rank])
                        category = anchors.categories[anchor_idx]
                        label = anchors.labels[anchor_idx]
                        if score < _get_threshold(category, config):
                            continue
                        if _is_direct_reference(label, record.text):
                            continue
                        anchor_vec = anchors.embeddings[anchor_idx].reshape(1, -1)
                        phrase_result = extract_phrase(window.text, anchor_vec, model, config)
                        phrase_text = phrase_result["phrase"] if phrase_result else window.center_word
                        phrase_drop = phrase_result["drop"] if phrase_result else 0.0
                        dedup_key = (phrase_text.lower(), label)
                        if dedup_key in seen:
                            continue
                        seen.add(dedup_key)
                        total_hits += 1
                        yield EuphemismCandidate(
                            text=phrase_text, context=record.text,
                            taboo_anchor=label, taboo_category=category,
                            similarity_score=score,
                            phrase_extraction_method=config.phrase_method,
                            phrase_drop_score=phrase_drop,
                            timestamp=record.timestamp,
                            source_url=record.source_url,
                            source=record.source,
                            context_window_size=config.context_window_size,
                            all_anchor_matches=[])

        total_processed += len(batch)
        if total_processed % 10000 == 0:
            logger.info(f"Processed {total_processed} sentences, {total_hits} hits")

    logger.info(f"First pass complete: {total_processed} sentences, {total_hits} hits")


def _process_hits(sentence, record, scores, indices, anchors, model, config):
    valid = []
    for rank in range(config.faiss_k):
        score = float(scores[rank])
        anchor_idx = int(indices[rank])
        category = anchors.categories[anchor_idx]
        label = anchors.labels[anchor_idx]
        if score < _get_threshold(category, config):
            continue
        if _is_direct_reference(label, sentence):
            continue
        valid.append({"anchor": label, "category": category,
                       "score": score, "anchor_idx": anchor_idx})

    if not valid:
        return

    all_matches = [{"anchor": m["anchor"], "category": m["category"],
                     "score": m["score"]} for m in valid]

    for match in valid:
        anchor_vec = anchors.embeddings[match["anchor_idx"]].reshape(1, -1)
        phrase_result = extract_phrase(sentence, anchor_vec, model, config)
        phrase_text = phrase_result["phrase"] if phrase_result else sentence
        phrase_drop = phrase_result["drop"] if phrase_result else 0.0
        yield EuphemismCandidate(
            text=phrase_text, context=sentence,
            taboo_anchor=match["anchor"], taboo_category=match["category"],
            similarity_score=match["score"],
            phrase_extraction_method=config.phrase_method,
            phrase_drop_score=phrase_drop,
            timestamp=record.timestamp, source_url=record.source_url,
            source=record.source,
            context_window_size=config.context_window_size,
            all_anchor_matches=all_matches)


# ============================================================================
# 8. OUTPUT
# ============================================================================

def write_results(candidates, output_path):
    count = 0
    with open(output_path, "w") as f:
        for c in candidates:
            f.write(json.dumps(asdict(c)) + "\n")
            count += 1
            if count % 1000 == 0:
                logger.info(f"Written {count} candidates")
    logger.info(f"Output: {count} candidates -> {output_path}")
    return count


# ============================================================================
# 9. PIPELINE RUNNER
# ============================================================================

def run_pipeline(sources, config=PipelineConfig(), taboo_path="taboo_words_refined.py"):
    logger.info(f"Loading model: {config.model_name} on {config.device}")
    model = SentenceTransformer(config.model_name, device=config.device)

    logger.info("Loading taboo anchors...")
    taboo_dict = load_taboo_anchors(taboo_path)
    anchors = prepare_anchors(taboo_dict, template=config.anchor_template)
    logger.info(f"{anchors.size} anchors across {len(set(anchors.categories))} categories")

    logger.info("Encoding anchors...")
    anchor_embs = model.encode(anchors.texts, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(anchor_embs)
    anchors.embeddings = anchor_embs

    faiss_index = build_faiss_index(anchor_embs.copy(), use_gpu=config.use_gpu_faiss)
    combined_stream = itertools.chain(*sources)

    logger.info(f"Starting first pass (window={config.context_window_size}, "
                f"method={config.phrase_method}, k={config.faiss_k})")
    candidates = first_pass(combined_stream, model, anchors, faiss_index, config)
    count = write_results(candidates, config.output_path)

    config_path = config.output_path.replace(".jsonl", "_config.json")
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)
    logger.info(f"Config saved to {config_path}")
    return count


# ============================================================================
# 10. EXPERIMENT RUNNER
# ============================================================================

def get_experiment_configs():
    baseline = dict(
        anchor_template="taboo", context_window_size=None,
        phrase_method="perturbation", use_per_category_thresholds=True, faiss_k=5)

    experiments = []
    for tmpl in ["none", "taboo", "euphemism"]:
        experiments.append(PipelineConfig(**{**baseline, "anchor_template": tmpl},
                                          output_path=f"exp_template_{tmpl}.jsonl"))
    for win in [None, 3, 5, 10]:
        label = "full" if win is None else str(win)
        experiments.append(PipelineConfig(**{**baseline, "context_window_size": win},
                                          output_path=f"exp_window_{label}.jsonl"))
    for method in ["perturbation", "contrastive_masking"]:
        experiments.append(PipelineConfig(**{**baseline, "phrase_method": method},
                                          output_path=f"exp_phrase_{method}.jsonl"))
    for per_cat in [True, False]:
        label = "percat" if per_cat else "global"
        experiments.append(PipelineConfig(**{**baseline, "use_per_category_thresholds": per_cat},
                                          output_path=f"exp_threshold_{label}.jsonl"))
    for k in [1, 5]:
        experiments.append(PipelineConfig(**{**baseline, "faiss_k": k},
                                          output_path=f"exp_k{k}.jsonl"))

    seen, unique = set(), []
    for exp in experiments:
        if exp.output_path not in seen:
            seen.add(exp.output_path)
            unique.append(exp)
    return unique


def run_experiments(source_factory, taboo_path="taboo_words_refined.py",
                    task_id=0, num_tasks=1, output_dir="."):
    configs = get_experiment_configs()
    logger.info(f"Running {len(configs)} experiments (task {task_id}/{num_tasks})")
    for i, config in enumerate(configs):
        # Prepend output dir and append task ID for SLURM parallelism
        base, ext = os.path.splitext(config.output_path)
        if num_tasks > 1:
            config.output_path = os.path.join(output_dir, f"{base}_task{task_id}{ext}")
        else:
            config.output_path = os.path.join(output_dir, config.output_path)

        logger.info(f"\n{'='*60}")
        logger.info(f"EXPERIMENT {i+1}/{len(configs)}: {config.output_path}")
        logger.info(f"{'='*60}")
        sources = source_factory()
        run_pipeline(sources, config, taboo_path)


# ============================================================================
# 11. ENTRY POINT — ACTUALLY RUNS
# ============================================================================

if __name__ == "__main__":
    import argparse

    from data_sources import (
        SourceConfig,
        stream_text_file,
        stream_csv,
        stream_csv_directory,
        stream_common_crawl,
        stream_common_crawl_wet,
        stream_common_crawl_wet_list,
        stream_reddit_directory,
    )

    parser = argparse.ArgumentParser(description="First-pass euphemism detection")
    parser.add_argument("--source", default="common_crawl",
                        choices=["common_crawl", "csv", "text_file", "reddit"],
                        help="Data source to use")
    parser.add_argument("--crawl-id", default="CC-MAIN-2024-10",
                        help="Common Crawl crawl ID (e.g. CC-MAIN-2024-10)")
    parser.add_argument("--max-wet-files", type=int, default=1,
                        help="Number of WET files to process (1 for testing)")
    parser.add_argument("--csv-path", default=None,
                        help="Path to CSV file or directory")
    parser.add_argument("--text-path", default=None,
                        help="Path to plain text file")
    parser.add_argument("--reddit-dir", default=None,
                        help="Path to directory of Reddit .zst dumps")
    parser.add_argument("--text-column", default=None,
                        help="CSV text column name (auto-detected if omitted)")
    parser.add_argument("--timestamp-column", default=None,
                        help="CSV timestamp column name (auto-detected if omitted)")
    parser.add_argument("--template", default="taboo",
                        choices=["none", "taboo", "euphemism"],
                        help="Anchor template wrapping strategy")
    parser.add_argument("--threshold", type=float, default=0.65,
                        help="Global similarity threshold")
    parser.add_argument("--per-category-thresholds", action="store_true",
                        help="Use per-category thresholds instead of global")
    parser.add_argument("--window", type=int, default=None,
                        help="Context window size (None=full sentence)")
    parser.add_argument("--output", default="first_pass_results.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--experiments", action="store_true",
                        help="Run full experiment suite instead of single run")

    # --- SLURM array parallelization ---
    parser.add_argument("--task-id", type=int, default=0,
                        help="SLURM array task index ($SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--num-tasks", type=int, default=1,
                        help="Total SLURM array tasks ($SLURM_ARRAY_TASK_COUNT)")
    args = parser.parse_args()

    # --- Source config (shared across all sources) ---
    source_config = SourceConfig(
        target_languages={"en"},
        lang_confidence_threshold=0.7,
        start_year=2015,
        end_year=2026,
    )

    # --- Pipeline config ---
    pipeline_config = PipelineConfig(
        anchor_template=args.template,
        context_window_size=args.window,
        use_per_category_thresholds=args.per_category_thresholds,
        global_threshold=args.threshold,
        output_path=args.output,
    )

    # --- Build source list ---
    def make_sources():
        if args.source == "common_crawl":
            # When parallelized with SLURM arrays, each task gets a
            # disjoint slice of WET files. With --num-tasks=4 and
            # --max-wet-files=100:
            #   task 0: files 0-24
            #   task 1: files 25-49
            #   task 2: files 50-74
            #   task 3: files 75-99
            #
            # When --num-tasks=1 (no parallelization), task 0 gets all files.
            from data_sources import download_wet_paths, stream_common_crawl_wet

            paths_file = download_wet_paths(args.crawl_id)
            with open(paths_file, "r") as f:
                all_paths = [line.strip() for line in f if line.strip()]

            # Cap to requested number of files
            all_paths = all_paths[:args.max_wet_files]

            # Shard: each task takes every num_tasks-th file starting at task_id
            # This interleaves rather than chunking, which gives better load
            # balancing (early files aren't systematically different from late ones)
            my_paths = all_paths[args.task_id::args.num_tasks]

            logger.info(
                f"Task {args.task_id}/{args.num_tasks}: "
                f"processing {len(my_paths)}/{len(all_paths)} WET files"
            )

            base_url = "https://data.commoncrawl.org/"
            return [
                stream_common_crawl_wet(base_url + path, config=source_config)
                for path in my_paths
            ]

        elif args.source == "csv":
            if args.csv_path is None:
                parser.error("--csv-path required when --source=csv")
            if os.path.isdir(args.csv_path):
                return [stream_csv_directory(
                    args.csv_path,
                    text_column=args.text_column,
                    timestamp_column=args.timestamp_column,
                    config=source_config,
                )]
            else:
                return [stream_csv(
                    args.csv_path,
                    text_column=args.text_column,
                    timestamp_column=args.timestamp_column,
                    config=source_config,
                    task_id=args.task_id,
                    num_tasks=args.num_tasks,
                )]

        elif args.source == "text_file":
            if args.text_path is None:
                parser.error("--text-path required when --source=text_file")
            return [stream_text_file(args.text_path, config=source_config)]

        elif args.source == "reddit":
            if args.reddit_dir is None:
                parser.error("--reddit-dir required when --source=reddit")
            # For Reddit with SLURM arrays, shard by file
            import glob as globmod
            all_files = sorted(globmod.glob(os.path.join(args.reddit_dir, "RC_*.zst")))
            if not all_files:
                all_files = sorted(globmod.glob(os.path.join(args.reddit_dir, "RC_*.jsonl")))
            my_files = all_files[args.task_id::args.num_tasks]
            logger.info(
                f"Task {args.task_id}/{args.num_tasks}: "
                f"processing {len(my_files)}/{len(all_files)} Reddit files"
            )
            from data_sources import stream_reddit_dump
            return [
                stream_reddit_dump(f, config=source_config)
                for f in my_files
            ]

    # --- Append task ID to output path so parallel jobs don't collide ---
    if args.num_tasks > 1:
        base, ext = os.path.splitext(args.output)
        pipeline_config = PipelineConfig(
            anchor_template=args.template,
            context_window_size=args.window,
            use_per_category_thresholds=args.per_category_thresholds,
            global_threshold=args.threshold,
            output_path=f"{base}_task{args.task_id}{ext}",
        )
    else:
        pipeline_config = PipelineConfig(
            anchor_template=args.template,
            context_window_size=args.window,
            use_per_category_thresholds=args.per_category_thresholds,
            global_threshold=args.threshold,
            output_path=args.output,
        )
    # --- Run ---
    if args.experiments:
        logger.info(f"Running experiment suite (task {args.task_id}/{args.num_tasks})")
        output_dir = os.path.dirname(args.output) or "."
        run_experiments(make_sources, task_id=args.task_id,
                        num_tasks=args.num_tasks, output_dir=output_dir)
    else:
        logger.info(f"Task {args.task_id}/{args.num_tasks}, source={args.source}")
        sources = make_sources()
        run_pipeline(sources, pipeline_config)