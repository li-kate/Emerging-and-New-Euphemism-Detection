"""
============================================================================
FIRST PASS: EUPHEMISM CANDIDATE DETECTION PIPELINE
============================================================================

Goal: Scan large corpora for words/phrases that are semantically close to
taboo anchors but are NOT the taboo words themselves (i.e., potential
euphemisms). Output candidates with context, scores, and timestamps for
second-pass validation and temporal analysis.

Architecture:
    1. Load & prepare taboo anchors (with optional template wrapping)
    2. Build FAISS index over anchor embeddings
    3. Stream input text from data_sources.py
    4. Optionally apply sliding context windows
    5. Stage A: Sentence-level similarity as cheap filter
    6. Stage B: Phrase extraction to isolate the euphemistic term
    7. Output structured candidates to JSONL

Experimental axes (for the paper):
    - Template wrapping: "none" vs "taboo" vs "euphemism"
    - k=1 vs k=5 in FAISS search
    - Context window size: ±3, ±5, ±10, full sentence
    - Phrase extraction: perturbation vs contrastive masking
    - Per-category vs global similarity thresholds

Depends on:
    - data_sources.py (TextRecord, batch_records, stream_* functions)
    - taboo_words_refined.py (TABOO_ANCHORS dict)
============================================================================
"""

import torch
import faiss
import numpy as np
import json
import logging
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
    """
    All tunable parameters in one place.

    Keeping config as a dataclass means:
        - Every experiment is fully reproducible (serialize the config)
        - No hidden globals or magic numbers scattered through the code
        - Easy to log alongside results
    """

    # --- Model ---
    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Anchor preparation ---
    # EXPERIMENT: Template wrapping
    #
    # Why this matters:
    #   Sentence embedders were trained on sentences, not isolated words.
    #   Embedding the bare word "death" produces a less stable vector than
    #   embedding "the taboo topic of death" because the model has more
    #   context to work with. The template nudges the anchor vector into
    #   the right semantic region.
    #
    # Options:
    #   "none"      -> embed raw: "death"
    #   "taboo"     -> embed: "the taboo topic of death"
    #   "euphemism" -> embed: "a euphemism for death"
    #
    # Hypothesis: "taboo" should outperform "none" because it adds context.
    # "euphemism" might overshoot — it could pull the anchor toward the
    # euphemism region itself, making it harder to detect euphemisms that
    # are semantically distant (which are the interesting ones).
    anchor_template: str = "none"

    # --- Context window ---
    # EXPERIMENT: How much text to embed around each position
    #
    # Why this matters:
    #   Full sentences dilute the euphemistic signal. A sentence with 20
    #   words where 1 is a euphemism means that word contributes ~5% of
    #   the embedding. A ±5 word window (11 words) means it contributes ~9%.
    #   But too narrow a window loses discourse cues (hedging, topic framing).
    #
    # Options:
    #   None  -> embed the full sentence as-is (original behavior)
    #   3     -> embed ±3 words around each word (7-word windows)
    #   5     -> embed ±5 words (11-word windows)
    #   10    -> embed ±10 words (21-word windows)
    #
    # Implementation: For each sentence, generate overlapping windows
    # centered on each word. Each window is embedded separately. This is
    # more expensive (N windows per sentence vs 1 embedding) but gives
    # much better localization of the euphemistic term.
    #
    # Trade-off: cost scales linearly with sentence length.
    # Mitigation: Only run windowed mode on sentences that FIRST pass the
    # full-sentence filter (two-stage approach).
    context_window_size: Optional[int] = None  # None = full sentence

    # When using windows, we can skip the full-sentence pre-filter or use it.
    # Pre-filtering is faster but might miss euphemisms that only show up
    # in narrow context. Setting this to a lower threshold than the main
    # threshold gives a cheap "maybe" filter before the expensive windowed pass.
    window_prefilter_threshold: float = 0.45

    # --- FAISS search ---
    # EXPERIMENT: k=1 vs k=5
    #
    # Why k>1 matters:
    #   A euphemism might be equidistant between two taboo anchors.
    #   "escort" matches both "prostitution" and general "companion" concepts.
    #   k=5 costs almost nothing extra (FAISS already computed the distances)
    #   and gives richer output for analysis.
    #
    # For the paper: k=5 with threshold filtering is strictly better than k=1.
    # k=1 is only worth testing to show this empirically.
    faiss_k: int = 5
    use_gpu_faiss: bool = True

    # --- Thresholds ---
    # EXPERIMENT: Global vs per-category
    #
    # Why per-category matters:
    #   Different taboo domains have different euphemism "distances" in
    #   embedding space. Profanity minced oaths ("fudge" for "fuck") are
    #   quite distant. Death euphemisms ("passed away") are closer.
    #   A single global threshold either misses distant euphemisms (too high)
    #   or floods you with false positives (too low).
    global_threshold: float = 0.65
    use_per_category_thresholds: bool = False

    # --- Phrase extraction ---
    # METHOD CHOICE: perturbation vs contrastive masking
    #
    # See detailed analysis in the section 4 docstrings below.
    # Summary: perturbation is the recommended default. Contrastive masking
    # is included as a variant for experimental comparison.
    #
    # "perturbation"        -> remove n-grams, measure similarity drop
    # "contrastive_masking" -> replace n-grams with [MASK], measure drop
    phrase_method: str = "perturbation"
    max_ngram_size: int = 4
    min_drop_threshold: float = 0.05

    # --- Batching ---
    sentence_batch_size: int = 256
    source_batch_size: int = 1000

    # --- Output ---
    output_path: str = "first_pass_results.jsonl"


# Per-category thresholds. These are starting points — tune on a labeled dev set.
# The values reflect how "far" euphemisms typically land from their anchors
# in embedding space.
CATEGORY_THRESHOLDS = {
    "drugs":              0.60,
    "alcohol":            0.60,
    "sex":                0.60,
    "death":              0.55,
    "health":             0.60,
    "menstruation":       0.55,
    "aging":              0.70,
    "body_functions":     0.55,
    "bodily_fluids":      0.65,
    "toilet":             0.55,
    "physical_injury":    0.60,
    "intelligence":       0.70,
    "emotions":           0.65,
    "sexual_orientation": 0.60,
    "identity":           0.65,
    "family":             0.60,
    "money_class":        0.65,
    "housing":            0.60,
    "employment":         0.60,
    "education":          0.65,
    "crime":              0.60,
    "politics":           0.65,
    "military":           0.60,
    "racism":             0.60,
    "migration":          0.60,
    "social_conflict":    0.60,
    "violence":           0.55,
    "technology":         0.65,
    "environment":        0.65,
    "morality":           0.65,
    "disabilities":       0.60,
    "mental_health":      0.60,
    "religion":           0.55,
    "weight_appearance":  0.65,
    "hygiene":            0.60,
    "profanity":          0.50,
    "deception":          0.60,
    "cosmetic":           0.60,
    "rejection":          0.65,
}


# ============================================================================
# 1. TABOO ANCHOR LOADING & PREPARATION
# ============================================================================

def load_taboo_anchors(path: str = "taboo_words_refined.py") -> dict[str, list[str]]:
    """
    Load the TABOO_ANCHORS dict from the refined list file.

    Using exec() is fine for research code where you control the file.
    For production, convert to JSON or YAML.
    """
    namespace = {}
    with open(path, "r") as f:
        exec(f.read(), namespace)
    return namespace["TABOO_ANCHORS"]


@dataclass
class AnchorData:
    """
    Bundled anchor data passed through the pipeline.
    Keeps the parallel arrays together so they can't get out of sync.
    """
    texts: list[str]          # What gets embedded (possibly template-wrapped)
    labels: list[str]         # Raw anchor words (for exclusion matching)
    categories: list[str]     # Category each anchor belongs to
    embeddings: np.ndarray = field(default=None, repr=False)  # Set after encoding

    @property
    def size(self) -> int:
        return len(self.labels)


def prepare_anchors(
    taboo_dict: dict[str, list[str]],
    template: str = "none",
) -> AnchorData:
    """
    Flatten the category dict into parallel lists and optionally apply
    template wrapping.

    Multi-word anchor handling:
        Anchors like "terminal illness" or "domestic violence" are embedded
        as-is (or template-wrapped as-is). The sentence embedder handles
        multi-word inputs natively. The key consideration is in the
        EXCLUSION logic (see _is_direct_reference below), not here.
    """
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
# 2. MULTI-WORD ANCHOR EXCLUSION LOGIC
# ============================================================================
#
# Why this needs special care:
#   The exclusion rule is: "if the taboo word itself appears in the sentence,
#   it's a direct reference, not a euphemism — skip it."
#
#   For single-word anchors like "death", a simple `"death" in sentence.lower()`
#   works fine. But for multi-word anchors:
#
#   Problem 1: "terminal illness" should match "terminal illness" but NOT
#              "terminal" alone (which might appear in "airport terminal").
#   Problem 2: We want word-boundary matching, not substring matching.
#              "ill" should NOT match inside "illustration".
#   Problem 3: Multi-word anchors might appear with different spacing or
#              punctuation: "terminal  illness" or "terminal-illness".
#
#   Solution: Use regex with word boundaries for robust matching.
# ============================================================================

import re

# Cache compiled regex patterns for performance (one per anchor)
_exclusion_patterns: dict[str, re.Pattern] = {}


def _is_direct_reference(anchor_label: str, sentence: str) -> bool:
    """
    Check if the sentence contains the taboo anchor as a direct reference
    (not a euphemism). Uses word-boundary regex for robust matching.

    Handles:
        - Single words: "death" matches "death" but not "deaths" ... wait,
          actually we DO want to match "deaths" as a direct reference.
          So we use a flexible boundary: word start + optional plural/suffix.
        - Multi-word: "terminal illness" matches with flexible spacing.
        - Case-insensitive.

    Returns True if the anchor appears directly (meaning: skip this sentence).
    """
    if anchor_label not in _exclusion_patterns:
        # For multi-word anchors, allow flexible whitespace between words
        # For all anchors, require word boundary at start, flexible at end
        # (to catch plurals, verb forms, etc.)
        words = anchor_label.lower().split()
        pattern_str = r"\b" + r"\s+".join(re.escape(w) for w in words)
        _exclusion_patterns[anchor_label] = re.compile(pattern_str, re.IGNORECASE)

    return bool(_exclusion_patterns[anchor_label].search(sentence))


# ============================================================================
# 3. FAISS INDEX
# ============================================================================

def build_faiss_index(
    embeddings: np.ndarray,
    use_gpu: bool = True,
) -> faiss.Index:
    """
    Build a normalized inner-product (cosine similarity) FAISS index.

    Why IndexFlatIP (not IVF, HNSW, etc.):
        We have ~75 anchors. For such a tiny index, exact search is
        instantaneous. Approximate methods only help with millions+ vectors.
        IndexFlatIP with L2-normalized vectors gives exact cosine similarity.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)

    if use_gpu and torch.cuda.is_available():
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            logger.info("FAISS index on GPU")
        except Exception as e:
            logger.warning(f"GPU FAISS failed, falling back to CPU: {e}")

    index.add(embeddings)
    logger.info(f"FAISS index built: {embeddings.shape[0]} anchors, dim={dim}")
    return index


# ============================================================================
# 4. CONTEXT WINDOW GENERATION
# ============================================================================
#
# Why context windows matter:
#
#   Consider the sentence: "After a long battle, grandma finally went to
#   a better place last Tuesday."
#
#   Full sentence embedding: The words "battle", "grandma", "finally",
#   "Tuesday" all contribute to the vector. The euphemism "went to a
#   better place" is diluted by non-euphemistic context.
#
#   ±5 window centered on "better": "finally went to a better place last"
#   Now the euphemism dominates the embedding and will score much higher
#   against the "death" anchor.
#
#   The cost: instead of 1 embedding per sentence, we need N embeddings
#   (one per word position). For a 20-word sentence, that's 20x more
#   compute. The two-stage approach mitigates this: only apply windowing
#   to sentences that passed a loose pre-filter.
#
# ============================================================================

@dataclass
class WindowedChunk:
    """A context window extracted from a sentence."""
    text: str              # The windowed text to embed
    center_word: str       # The word at the center of the window
    center_index: int      # Position of center word in original sentence
    full_sentence: str     # The original sentence (for context in output)
    source_record: TextRecord  # Original record metadata


def generate_windows(
    sentence: str,
    record: TextRecord,
    window_size: int,
) -> list[WindowedChunk]:
    """
    Generate overlapping context windows centered on each word.

    Args:
        sentence:    the full sentence text
        record:      the source TextRecord (for metadata)
        window_size: number of words on each side of center (±N)

    Returns:
        list of WindowedChunk objects, one per word position
    """
    words = sentence.split()
    chunks = []

    for i, word in enumerate(words):
        start = max(0, i - window_size)
        end = min(len(words), i + window_size + 1)
        window_text = " ".join(words[start:end])

        chunks.append(WindowedChunk(
            text=window_text,
            center_word=word,
            center_index=i,
            full_sentence=sentence,
            source_record=record,
        ))

    return chunks


# ============================================================================
# 5. PHRASE EXTRACTION METHODS
# ============================================================================

# ---------------------------------------------------------------------------
# Method A: Perturbation-Based (RECOMMENDED DEFAULT)
# ---------------------------------------------------------------------------
#
# How it works:
#   For each n-gram in the sentence, remove it, re-encode the remaining
#   sentence, and measure how much cosine similarity to the taboo anchor
#   drops. The n-gram whose removal causes the biggest drop is most
#   responsible for the euphemistic signal.
#
# Pros:
#   + Model-agnostic: works with ANY encoder as a pure black box
#   + Highly interpretable: "removing X dropped similarity by Y"
#   + Easy to explain in a paper — reviewers get it immediately
#   + Naturally produces n-gram candidates (not just tokens)
#   + All perturbations can be batched for efficient GPU encoding
#   + Handles multi-word euphemisms directly (just test larger n-grams)
#
# Cons:
#   - O(n × max_ngram_size) forward passes per sentence (mitigated by batching)
#   - Assumes independence: removing "passed" separately from "away" won't
#     show they form a unit. Mitigated by testing bigrams/trigrams.
#   - Deletion changes sentence structure, which may confuse the encoder.
#     In practice this effect is small with modern sentence transformers.
#
# When to use: Always. This is your primary method.
# ---------------------------------------------------------------------------

def extract_phrase_perturbation(
    sentence: str,
    anchor_vec: np.ndarray,
    model: SentenceTransformer,
    max_n: int = 4,
    min_drop: float = 0.05,
) -> Optional[dict]:
    """
    Find the n-gram most responsible for the sentence's similarity
    to the taboo anchor by measuring the impact of removing it.
    """
    words = sentence.split()
    if len(words) < 2:
        return None

    # Baseline: full sentence similarity
    baseline_vec = model.encode(sentence, convert_to_numpy=True).reshape(1, -1)
    baseline_vec /= np.linalg.norm(baseline_vec) + 1e-10
    baseline_score = float(np.dot(baseline_vec, anchor_vec.T).squeeze())

    candidates = []

    for n in range(1, min(max_n + 1, len(words) + 1)):
        perturbed_sentences = []
        ngrams = []

        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i:i + n])
            perturbed = " ".join(words[:i] + words[i + n:])
            if not perturbed.strip():
                continue
            ngrams.append(ngram)
            perturbed_sentences.append(perturbed)

        if not perturbed_sentences:
            continue

        # Batch encode all perturbations at once (key optimization!)
        perturbed_vecs = model.encode(
            perturbed_sentences, convert_to_numpy=True, batch_size=64
        )
        norms = np.linalg.norm(perturbed_vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        perturbed_vecs /= norms

        scores = np.dot(perturbed_vecs, anchor_vec.T).squeeze()
        if scores.ndim == 0:
            scores = np.array([float(scores)])

        for ngram, score in zip(ngrams, scores):
            drop = baseline_score - float(score)
            if drop > min_drop:
                candidates.append({
                    "phrase": ngram,
                    "drop": drop,
                    "ngram_size": n,
                })

    if not candidates:
        return None

    # Sort: highest drop first, break ties by preferring longer n-grams
    candidates.sort(key=lambda x: (x["drop"], x["ngram_size"]), reverse=True)

    # Overlap resolution: if a bigram contains the top unigram and has
    # a similar drop, prefer the bigram (more informative unit)
    best = candidates[0]
    for c in candidates[1:]:
        if (
            c["ngram_size"] > best["ngram_size"]
            and best["phrase"] in c["phrase"]
            and abs(c["drop"] - best["drop"]) < 0.05
        ):
            best = c
            break

    return best


# ---------------------------------------------------------------------------
# Method B: Contrastive Masking (variant for experimental comparison)
# ---------------------------------------------------------------------------
#
# How it works:
#   Same as perturbation, but instead of DELETING an n-gram, we REPLACE
#   it with a [MASK] token (or a neutral placeholder). This preserves
#   the sentence length and syntactic structure.
#
# Pros:
#   + Preserves sentence structure (more linguistically principled)
#   + The model sees a well-formed sentence instead of a truncated one
#   + Trivial to implement as a variant of perturbation (same code path)
#
# Cons:
#   - [MASK] is a BERT pretraining token; sentence transformers may handle
#     it differently than expected (it's not in their training distribution)
#   - Using a neutral word like "something" instead of [MASK] introduces
#     its own semantic content
#   - In practice, modern sentence transformers are robust enough that
#     the deletion vs replacement distinction rarely matters
#
# When to use: As an experimental comparison to perturbation. My prediction
#   is that results will be very similar. If they are, perturbation wins
#   by simplicity (fewer assumptions about the replacement token).
#
# Note on Integrated Gradients (not implemented):
#   IG is a gradient-based attribution method that computes how each input
#   token contributes to the output similarity score. It's theoretically
#   elegant (satisfies the completeness axiom) but:
#     1. Requires unwrapping the SentenceTransformer internals
#     2. Produces token-level (not n-gram-level) attributions
#     3. Needs O(n_steps) forward+backward passes per sentence
#     4. Is harder to explain to reviewers than perturbation
#     5. The similarity metric (cosine) is less natural for IG than
#        a classification logit
#   For this pipeline, perturbation provides the same practical information
#   with much less engineering complexity. If reviewers ask, you can cite
#   IG as future work or run it on a small validation sample.
# ---------------------------------------------------------------------------

def extract_phrase_contrastive_masking(
    sentence: str,
    anchor_vec: np.ndarray,
    model: SentenceTransformer,
    max_n: int = 4,
    min_drop: float = 0.05,
    mask_token: str = "[MASK]",
) -> Optional[dict]:
    """
    Find the n-gram most responsible for similarity by replacing it with
    a mask token and measuring the similarity drop.

    The mask_token argument lets you experiment:
        "[MASK]"    -> BERT-style mask (may behave unexpectedly)
        "something" -> neutral semantically-light word
        "___"       -> visual placeholder (model sees it as unknown token)
    """
    words = sentence.split()
    if len(words) < 2:
        return None

    baseline_vec = model.encode(sentence, convert_to_numpy=True).reshape(1, -1)
    baseline_vec /= np.linalg.norm(baseline_vec) + 1e-10
    baseline_score = float(np.dot(baseline_vec, anchor_vec.T).squeeze())

    candidates = []

    for n in range(1, min(max_n + 1, len(words) + 1)):
        masked_sentences = []
        ngrams = []

        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i:i + n])
            # Replace with mask instead of deleting
            masked = words[:i] + [mask_token] + words[i + n:]
            masked_sentences.append(" ".join(masked))
            ngrams.append(ngram)

        if not masked_sentences:
            continue

        masked_vecs = model.encode(
            masked_sentences, convert_to_numpy=True, batch_size=64
        )
        norms = np.linalg.norm(masked_vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        masked_vecs /= norms

        scores = np.dot(masked_vecs, anchor_vec.T).squeeze()
        if scores.ndim == 0:
            scores = np.array([float(scores)])

        for ngram, score in zip(ngrams, scores):
            drop = baseline_score - float(score)
            if drop > min_drop:
                candidates.append({
                    "phrase": ngram,
                    "drop": drop,
                    "ngram_size": n,
                })

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x["drop"], x["ngram_size"]), reverse=True)

    best = candidates[0]
    for c in candidates[1:]:
        if (
            c["ngram_size"] > best["ngram_size"]
            and best["phrase"] in c["phrase"]
            and abs(c["drop"] - best["drop"]) < 0.05
        ):
            best = c
            break

    return best


# Dispatcher: pick the right method based on config
def extract_phrase(
    sentence: str,
    anchor_vec: np.ndarray,
    model: SentenceTransformer,
    config: PipelineConfig,
) -> Optional[dict]:
    """Route to the configured phrase extraction method."""
    if config.phrase_method == "contrastive_masking":
        return extract_phrase_contrastive_masking(
            sentence, anchor_vec, model,
            max_n=config.max_ngram_size,
            min_drop=config.min_drop_threshold,
        )
    else:  # "perturbation" (default)
        return extract_phrase_perturbation(
            sentence, anchor_vec, model,
            max_n=config.max_ngram_size,
            min_drop=config.min_drop_threshold,
        )


# ============================================================================
# 6. STRUCTURED OUTPUT
# ============================================================================

@dataclass
class EuphemismCandidate:
    """Structured output for each detected candidate."""
    text: str                           # The extracted euphemistic phrase
    context: str                        # The full sentence
    taboo_anchor: str                   # Which anchor it matched
    taboo_category: str                 # Category of the anchor
    similarity_score: float             # Cosine similarity
    phrase_extraction_method: str = ""  # "perturbation" | "contrastive_masking"
    phrase_drop_score: float = 0.0      # Similarity drop when phrase is removed
    timestamp: str = ""
    source_url: str = ""
    source: str = ""
    context_window_size: Optional[int] = None  # None = full sentence
    all_anchor_matches: list = field(default_factory=list)


# ============================================================================
# 7. CORE FIRST PASS
# ============================================================================

def _get_threshold(category: str, config: PipelineConfig) -> float:
    """Get the similarity threshold for a given category."""
    if config.use_per_category_thresholds:
        return CATEGORY_THRESHOLDS.get(category, config.global_threshold)
    return config.global_threshold


def first_pass(
    data_stream: Iterator[TextRecord],
    model: SentenceTransformer,
    anchors: AnchorData,
    faiss_index: faiss.Index,
    config: PipelineConfig,
) -> Iterator[EuphemismCandidate]:
    """
    Main detection loop.

    Two modes depending on config.context_window_size:

    Mode 1 (context_window_size=None): Full-sentence embedding
        - Embed each sentence once
        - Search against FAISS index
        - Run phrase extraction on hits
        Fast, good for initial exploration.

    Mode 2 (context_window_size=N): Two-stage windowed approach
        - Stage 0: Embed full sentence, apply loose pre-filter threshold
        - Stage 1: For sentences passing pre-filter, generate ±N word windows
        - Stage 2: Embed each window, search against FAISS
        - Run phrase extraction on window-level hits
        More expensive but better at localizing euphemisms.
    """
    total_processed = 0
    total_hits = 0

    for batch in batch_records(data_stream, config.sentence_batch_size):
        sentences = [r.text for r in batch]

        # --- Encode full sentences ---
        batch_vecs = model.encode(
            sentences, convert_to_numpy=True, batch_size=config.sentence_batch_size,
        ).astype("float32")
        faiss.normalize_L2(batch_vecs)

        if config.context_window_size is None:
            # ============================================================
            # MODE 1: Full-sentence matching
            # ============================================================
            scores, indices = faiss_index.search(batch_vecs, config.faiss_k)

            for i, record in enumerate(batch):
                yield from _process_hits(
                    sentence=record.text,
                    record=record,
                    scores=scores[i],
                    indices=indices[i],
                    anchors=anchors,
                    model=model,
                    config=config,
                )
                total_hits += 1  # approximate; actual count inside _process_hits

        else:
            # ============================================================
            # MODE 2: Windowed matching (two-stage)
            # ============================================================

            # Stage 0: Loose pre-filter on full sentences
            pre_scores, _ = faiss_index.search(batch_vecs, 1)

            for i, record in enumerate(batch):
                if pre_scores[i][0] < config.window_prefilter_threshold:
                    continue  # Sentence is too far from all anchors, skip

                # Stage 1: Generate windows
                windows = generate_windows(
                    record.text, record, config.context_window_size
                )
                window_texts = [w.text for w in windows]

                if not window_texts:
                    continue

                # Stage 2: Embed and search windows
                win_vecs = model.encode(
                    window_texts, convert_to_numpy=True, batch_size=64,
                ).astype("float32")
                faiss.normalize_L2(win_vecs)

                win_scores, win_indices = faiss_index.search(win_vecs, config.faiss_k)

                # Deduplicate: a euphemism might be found in multiple
                # overlapping windows. Track (phrase, anchor) pairs we've
                # already yielded for this sentence.
                seen = set()

                for j, window in enumerate(windows):
                    for rank in range(config.faiss_k):
                        score = float(win_scores[j][rank])
                        anchor_idx = int(win_indices[j][rank])
                        category = anchors.categories[anchor_idx]
                        label = anchors.labels[anchor_idx]
                        threshold = _get_threshold(category, config)

                        if score < threshold:
                            continue

                        if _is_direct_reference(label, record.text):
                            continue

                        # Phrase extraction on the window
                        anchor_vec = anchors.embeddings[anchor_idx].reshape(1, -1)
                        phrase_result = extract_phrase(
                            window.text, anchor_vec, model, config
                        )

                        phrase_text = phrase_result["phrase"] if phrase_result else window.center_word
                        phrase_drop = phrase_result["drop"] if phrase_result else 0.0

                        dedup_key = (phrase_text.lower(), label)
                        if dedup_key in seen:
                            continue
                        seen.add(dedup_key)

                        yield EuphemismCandidate(
                            text=phrase_text,
                            context=record.text,
                            taboo_anchor=label,
                            taboo_category=category,
                            similarity_score=score,
                            phrase_extraction_method=config.phrase_method,
                            phrase_drop_score=phrase_drop,
                            timestamp=record.timestamp,
                            source_url=record.source_url,
                            source=record.source,
                            context_window_size=config.context_window_size,
                            all_anchor_matches=[],
                        )
                        total_hits += 1

        total_processed += len(batch)
        if total_processed % 10000 == 0:
            logger.info(f"Processed {total_processed} sentences, {total_hits} hits")

    logger.info(f"First pass complete: {total_processed} sentences, {total_hits} hits")


def _process_hits(
    sentence: str,
    record: TextRecord,
    scores: np.ndarray,
    indices: np.ndarray,
    anchors: AnchorData,
    model: SentenceTransformer,
    config: PipelineConfig,
) -> Iterator[EuphemismCandidate]:
    """
    Process FAISS results for a single sentence (Mode 1).
    Yields one candidate per unique (sentence, anchor) pair above threshold.
    """
    # First collect all valid anchor matches for this sentence
    valid_matches = []
    for rank in range(config.faiss_k):
        score = float(scores[rank])
        anchor_idx = int(indices[rank])
        category = anchors.categories[anchor_idx]
        label = anchors.labels[anchor_idx]
        threshold = _get_threshold(category, config)

        if score < threshold:
            continue
        if _is_direct_reference(label, sentence):
            continue

        valid_matches.append({
            "anchor": label,
            "category": category,
            "score": score,
            "anchor_idx": anchor_idx,
        })

    if not valid_matches:
        return

    # Build the all_matches list (for multi-category analysis)
    all_matches_summary = [
        {"anchor": m["anchor"], "category": m["category"], "score": m["score"]}
        for m in valid_matches
    ]

    # Yield one candidate per unique anchor match
    for match in valid_matches:
        anchor_vec = anchors.embeddings[match["anchor_idx"]].reshape(1, -1)
        phrase_result = extract_phrase(sentence, anchor_vec, model, config)

        phrase_text = phrase_result["phrase"] if phrase_result else sentence
        phrase_drop = phrase_result["drop"] if phrase_result else 0.0

        yield EuphemismCandidate(
            text=phrase_text,
            context=sentence,
            taboo_anchor=match["anchor"],
            taboo_category=match["category"],
            similarity_score=match["score"],
            phrase_extraction_method=config.phrase_method,
            phrase_drop_score=phrase_drop,
            timestamp=record.timestamp,
            source_url=record.source_url,
            source=record.source,
            context_window_size=config.context_window_size,
            all_anchor_matches=all_matches_summary,
        )


# ============================================================================
# 8. OUTPUT
# ============================================================================

def write_results(
    candidates: Iterator[EuphemismCandidate],
    output_path: str,
) -> int:
    """Write candidates to JSONL. Returns count."""
    count = 0
    with open(output_path, "w") as f:
        for candidate in candidates:
            f.write(json.dumps(asdict(candidate)) + "\n")
            count += 1
            if count % 1000 == 0:
                logger.info(f"Written {count} candidates")
    logger.info(f"Output: {count} candidates -> {output_path}")
    return count


# ============================================================================
# 9. PIPELINE RUNNER
# ============================================================================

def run_pipeline(
    sources: list[Iterator[TextRecord]],
    config: PipelineConfig = PipelineConfig(),
    taboo_path: str = "taboo_words_refined.py",
):
    """Full first-pass pipeline."""

    # 1. Load model
    logger.info(f"Loading model: {config.model_name} on {config.device}")
    model = SentenceTransformer(config.model_name, device=config.device)

    # 2. Load and prepare anchors
    logger.info("Loading taboo anchors...")
    taboo_dict = load_taboo_anchors(taboo_path)
    anchors = prepare_anchors(taboo_dict, template=config.anchor_template)
    logger.info(f"{anchors.size} anchors across {len(set(anchors.categories))} categories")

    # 3. Encode anchors
    logger.info("Encoding anchors...")
    anchor_embs = model.encode(
        anchors.texts, convert_to_numpy=True
    ).astype("float32")
    faiss.normalize_L2(anchor_embs)
    anchors.embeddings = anchor_embs

    # 4. Build FAISS index
    faiss_index = build_faiss_index(anchor_embs.copy(), use_gpu=config.use_gpu_faiss)

    # 5. Chain all sources
    combined_stream = itertools.chain(*sources)

    # 6. Run first pass
    logger.info(f"Starting first pass (window={config.context_window_size}, "
                f"method={config.phrase_method}, k={config.faiss_k})")
    candidates = first_pass(combined_stream, model, anchors, faiss_index, config)

    # 7. Write output
    count = write_results(candidates, config.output_path)

    # 8. Save config alongside results for reproducibility
    config_path = config.output_path.replace(".jsonl", "_config.json")
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)
    logger.info(f"Config saved to {config_path}")

    return count


# ============================================================================
# 10. EXPERIMENT RUNNER
# ============================================================================

def get_experiment_configs() -> list[PipelineConfig]:
    """
    Define all experimental conditions for the paper.

    Axes:
        1. Template wrapping: none / taboo / euphemism
        2. Context window: None / 5 / 10
        3. Phrase method: perturbation / contrastive_masking
        4. Thresholds: global / per-category
        5. FAISS k: 1 / 5

    We don't run the full Cartesian product (that's 2×3×2×2×2 = 48 configs).
    Instead we vary one axis at a time from a sensible baseline.
    """
    baseline = dict(
        anchor_template="taboo",
        context_window_size=None,
        phrase_method="perturbation",
        use_per_category_thresholds=True,
        faiss_k=5,
    )

    experiments = []

    # --- Axis 1: Template wrapping ---
    for tmpl in ["none", "taboo", "euphemism"]:
        experiments.append(PipelineConfig(
            **{**baseline, "anchor_template": tmpl},
            output_path=f"exp_template_{tmpl}.jsonl",
        ))

    # --- Axis 2: Context window size ---
    for win in [None, 3, 5, 10]:
        label = "full" if win is None else str(win)
        experiments.append(PipelineConfig(
            **{**baseline, "context_window_size": win},
            output_path=f"exp_window_{label}.jsonl",
        ))

    # --- Axis 3: Phrase extraction method ---
    for method in ["perturbation", "contrastive_masking"]:
        experiments.append(PipelineConfig(
            **{**baseline, "phrase_method": method},
            output_path=f"exp_phrase_{method}.jsonl",
        ))

    # --- Axis 4: Global vs per-category thresholds ---
    for per_cat in [True, False]:
        label = "percat" if per_cat else "global"
        experiments.append(PipelineConfig(
            **{**baseline, "use_per_category_thresholds": per_cat},
            output_path=f"exp_threshold_{label}.jsonl",
        ))

    # --- Axis 5: FAISS k ---
    for k in [1, 5]:
        experiments.append(PipelineConfig(
            **{**baseline, "faiss_k": k},
            output_path=f"exp_k{k}.jsonl",
        ))

    # Deduplicate (baseline appears in multiple axes)
    seen_paths = set()
    unique = []
    for exp in experiments:
        if exp.output_path not in seen_paths:
            seen_paths.add(exp.output_path)
            unique.append(exp)

    return unique


def run_experiments(
    source_factory,
    taboo_path: str = "taboo_words_refined.py",
):
    """
    Run all experimental conditions.

    Args:
        source_factory: A CALLABLE that returns a fresh list of source
                        iterators each time it's called. This is necessary
                        because Python iterators are consumed after one pass.

                        Example:
                            def make_sources():
                                return [stream_text_file("corpus.txt")]
                            run_experiments(make_sources)
    """
    configs = get_experiment_configs()
    logger.info(f"Running {len(configs)} experiments")

    for i, config in enumerate(configs):
        logger.info(f"\n{'='*60}")
        logger.info(f"EXPERIMENT {i+1}/{len(configs)}: {config.output_path}")
        logger.info(f"  template={config.anchor_template}, "
                     f"window={config.context_window_size}, "
                     f"method={config.phrase_method}, "
                     f"per_cat={config.use_per_category_thresholds}, "
                     f"k={config.faiss_k}")
        logger.info(f"{'='*60}")

        # Fresh source iterators for each experiment
        sources = source_factory()
        run_pipeline(sources, config, taboo_path)


# ============================================================================
# 11. ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    from data_sources import (
        SourceConfig,
        stream_text_file,
        stream_common_crawl_wet,
        stream_reddit_directory,
        stream_common_crawl_wet_list,
        # stream_wikipedia_dump,
        # stream_reddit_dump,
        # stream_twitter_dump,
        # stream_news_dump,
    )

    # ------------------------------------------------------------------
    # SOURCE FILTERING CONFIG (shared across all sources)
    # ------------------------------------------------------------------
    source_config = SourceConfig(
        target_languages={"en"},       # English only
        lang_confidence_threshold=0.7, # Conservative
        start_year=2015,               # Study period start
        end_year=2026,                 # Study period end
        min_words=5,
        max_words=200,
    )

    # ------------------------------------------------------------------
    # SINGLE RUN (quick test)
    # ------------------------------------------------------------------
    # pipeline_config = PipelineConfig(
    #     anchor_template="taboo",
    #     context_window_size=5,
    #     use_per_category_thresholds=True,
    #     phrase_method="perturbation",
    #     faiss_k=5,
    #     output_path="first_pass_results.jsonl",
    # )
    # sources = [stream_text_file("test_corpus.txt", config=source_config)]
    # run_pipeline(sources, pipeline_config)

    # ------------------------------------------------------------------
    # MULTI-SOURCE RUN (realistic)
    # ------------------------------------------------------------------
    # sources = [
    #     stream_reddit_directory("./reddit_dumps/", config=source_config),
    #     stream_common_crawl_wet_list(
    #         "./cc_wet_paths.txt", config=source_config, max_files=10,
    #     ),
    #     stream_text_file("local_corpus.txt", config=source_config),
    # ]
    # run_pipeline(sources, pipeline_config)

    # ------------------------------------------------------------------
    # FULL EXPERIMENT SUITE (for the paper)
    # ------------------------------------------------------------------
    # def make_sources():
    #     return [
    #         stream_text_file("test_corpus.txt", config=source_config),
    #         stream_reddit_directory("./reddit_dumps/", config=source_config),
    #     ]
    # run_experiments(make_sources)

    print("Uncomment a configuration above and run.")