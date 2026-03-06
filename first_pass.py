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
    3. Stream input text from multiple sources
    4. Stage A: Sentence-level similarity as cheap filter
    5. Stage B: Phrase extraction to isolate the euphemistic term
    6. Output structured candidates

Experimental axes (for the paper):
    - Template wrapping vs bare anchors
    - k=1 vs k=5 in FAISS search
    - Perturbation vs Integrated Gradients for phrase extraction
    - Per-category vs global similarity thresholds
============================================================================
"""

import torch
import faiss
import numpy as np
import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Iterator, Optional
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# 0. CONFIGURATION
# ============================================================================

@dataclass
class PipelineConfig:
    """All tunable parameters in one place for reproducibility."""

    # --- Model ---
    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Anchor preparation ---
    # EXPERIMENT: Does template wrapping improve results?
    #   "none"     -> embed the raw anchor word: "death"
    #   "taboo"    -> embed "the taboo topic of death"
    #   "euphemism"-> embed "a euphemism for death"
    anchor_template: str = "none"  # "none" | "taboo" | "euphemism"

    # --- FAISS search ---
    # EXPERIMENT: k=1 vs k=5
    #   k=1: fastest, but misses multi-category euphemisms
    #   k=5: slightly slower, captures ambiguity, richer output for analysis
    faiss_k: int = 5
    use_gpu_faiss: bool = True

    # --- Thresholds ---
    # EXPERIMENT: global vs per-category thresholds
    # If per-category, these are overridden by CATEGORY_THRESHOLDS below
    global_threshold: float = 0.65
    use_per_category_thresholds: bool = False

    # --- Phrase extraction ---
    # EXPERIMENT: which method(s) to run
    #   "perturbation" -> drop-one-n-gram method (model agnostic)
    #   "ig"           -> integrated gradients (more principled, slower)
    #   "both"         -> run both, store both results for comparison
    phrase_method: str = "perturbation"  # "perturbation" | "ig" | "both"
    max_ngram_size: int = 4  # max n-gram to test in perturbation
    min_drop_threshold: float = 0.05  # min similarity drop to count as a hit

    # --- Batching ---
    sentence_batch_size: int = 256  # sentences per encode() call
    source_batch_size: int = 1000  # sentences to accumulate before processing

    # --- Output ---
    output_path: str = "first_pass_results.jsonl"


# Per-category thresholds (used if config.use_per_category_thresholds=True)
# High-polysemy anchors need stricter thresholds to reduce false positives.
# Tune these empirically on a labeled dev set.
CATEGORY_THRESHOLDS = {
    "drugs":              0.60,
    "alcohol":            0.60,
    "sex":                0.60,
    "death":              0.55,  # "death" is unambiguous, euphemisms may be distant
    "health":             0.60,
    "menstruation":       0.55,
    "aging":              0.70,  # "elderly" is polysemous-adjacent
    "body_functions":     0.55,
    "bodily_fluids":      0.65,
    "toilet":             0.55,
    "physical_injury":    0.60,
    "intelligence":       0.70,  # "stupid" is used very loosely
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
    "religion":           0.55,  # Minced oaths are often quite distant
    "weight_appearance":  0.65,
    "hygiene":            0.60,
    "profanity":          0.50,  # Minced oaths can be very distant ("fudge"<->"fuck")
    "deception":          0.60,
    "cosmetic":           0.60,
    "rejection":          0.65,
}


# ============================================================================
# 1. TABOO ANCHOR LOADING
# ============================================================================

def load_taboo_anchors(path: str = "taboo_words_refined.py") -> dict[str, list[str]]:
    """
    Load the TABOO_ANCHORS dict from the refined list file.
    We exec() the file and extract the variable. In production you'd
    want this as JSON or YAML, but for research this is fine.
    """
    namespace = {}
    with open(path, "r") as f:
        exec(f.read(), namespace)
    return namespace["TABOO_ANCHORS"]


def prepare_anchors(
    taboo_dict: dict[str, list[str]],
    template: str = "none",
) -> tuple[list[str], list[str], list[str]]:
    """
    Flatten the category dict into parallel lists and optionally apply
    template wrapping for embedding.

    Returns:
        anchor_texts:   what gets embedded (possibly template-wrapped)
        anchor_labels:  the raw anchor word (for matching/exclusion logic)
        anchor_categories: the category each anchor belongs to
    """
    anchor_texts = []
    anchor_labels = []
    anchor_categories = []

    for category, words in taboo_dict.items():
        for word in words:
            anchor_labels.append(word)
            anchor_categories.append(category)

            if template == "taboo":
                anchor_texts.append(f"the taboo topic of {word}")
            elif template == "euphemism":
                anchor_texts.append(f"a euphemism for {word}")
            else:
                anchor_texts.append(word)

    return anchor_texts, anchor_labels, anchor_categories


# ============================================================================
# 2. FAISS INDEX
# ============================================================================

def build_faiss_index(
    embeddings: np.ndarray,
    use_gpu: bool = True,
) -> faiss.Index:
    """
    Build a normalized inner-product (cosine similarity) FAISS index.
    GPU-accelerated if available and requested.
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

    # Normalize before adding (so inner product = cosine similarity)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    logger.info(f"FAISS index built with {embeddings.shape[0]} anchors, dim={dim}")
    return index


# ============================================================================
# 3. DATA SOURCES — Streaming Iterators
# ============================================================================
# Each source yields dicts: {"text": str, "timestamp": str, "source_url": str, "source": str}
# The pipeline consumes these uniformly.

@dataclass
class TextRecord:
    """Standardized record from any source."""
    text: str
    timestamp: str
    source_url: str = ""
    source: str = ""


def clean_text(text: str) -> str:
    """Minimal cleaning. Preserve casing and punctuation for the embedder."""
    text = re.sub(r"https?://\S+", "", text)          # strip URLs
    text = re.sub(r"@\w+", "", text)                   # strip @handles
    text = re.sub(r"<[^>]+>", "", text)                 # strip residual HTML
    text = re.sub(r"\s+", " ", text).strip()            # normalize whitespace
    return text


# --- Common Crawl WET ---

def stream_common_crawl_wet(wet_url: str) -> Iterator[TextRecord]:
    """
    Stream pre-extracted text from a Common Crawl WET file.
    WET files are already plaintext, saving HTML parsing.
    """
    import requests
    from warcio.archiveiterator import ArchiveIterator

    response = requests.get(wet_url, stream=True)
    if response.status_code != 200:
        logger.error(f"Failed to fetch {wet_url}: {response.status_code}")
        return

    for record in ArchiveIterator(response.raw):
        if record.rec_type == "conversion":
            uri = record.rec_headers.get_header("WARC-Target-URI")
            timestamp = record.rec_headers.get_header("WARC-Date")
            content = record.content_stream().read().decode("utf-8", errors="ignore")

            if not content:
                continue

            for line in content.split("\n"):
                line = clean_text(line)
                # Heuristic: skip very short lines (headers, nav elements)
                # and very long lines (likely boilerplate/repeated content)
                if 5 < len(line.split()) < 200:
                    yield TextRecord(
                        text=line,
                        timestamp=timestamp or "",
                        source_url=uri or "",
                        source="Common Crawl",
                    )


# --- Wikipedia dump ---

def stream_wikipedia_dump(dump_path: str) -> Iterator[TextRecord]:
    """
    Stream sentences from a Wikipedia XML dump.
    Use mwparserfromhell to strip wikitext markup.

    Requires: pip install mwxml mwparserfromhell
    Download: https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
    """
    import mwxml
    import mwparserfromhell
    import bz2

    dump = mwxml.Dump.from_file(bz2.open(dump_path, "rt", encoding="utf-8"))

    for page in dump:
        for revision in page:
            if revision.text is None:
                continue
            wikicode = mwparserfromhell.parse(revision.text)
            plaintext = wikicode.strip_code()

            for line in plaintext.split("\n"):
                line = clean_text(line)
                if 5 < len(line.split()) < 200:
                    yield TextRecord(
                        text=line,
                        timestamp=str(revision.timestamp) if revision.timestamp else "",
                        source_url=f"https://en.wikipedia.org/wiki/{page.title}",
                        source="Wikipedia",
                    )


# --- Reddit (Pushshift/Academic Torrents dumps) ---

def stream_reddit_dump(dump_path: str) -> Iterator[TextRecord]:
    """
    Stream from a Reddit comment dump (JSONL, one JSON per line).
    Available from Academic Torrents or Pushshift archives.
    Handles both .jsonl and .zst compressed files.
    """
    import zstandard as zstd

    if dump_path.endswith(".zst"):
        dctx = zstd.ZstdDecompressor()
        fh = open(dump_path, "rb")
        reader = dctx.stream_reader(fh)
        text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")
    else:
        text_stream = open(dump_path, "r", encoding="utf-8", errors="ignore")

    try:
        for line_str in text_stream:
            try:
                obj = json.loads(line_str)
            except json.JSONDecodeError:
                continue

            body = obj.get("body", "")
            if not body or body in ("[deleted]", "[removed]"):
                continue

            body = clean_text(body)
            # Reddit comments can be multi-sentence; split them
            for sentence in body.split(". "):
                sentence = sentence.strip()
                if 5 < len(sentence.split()) < 200:
                    yield TextRecord(
                        text=sentence,
                        timestamp=str(obj.get("created_utc", "")),
                        source_url=f"https://reddit.com{obj.get('permalink', '')}",
                        source="Reddit",
                    )
    finally:
        if dump_path.endswith(".zst"):
            fh.close()


# --- Twitter / X (Academic API JSONL exports) ---

def stream_twitter_dump(dump_path: str) -> Iterator[TextRecord]:
    """
    Stream from Twitter/X academic API export (JSONL).
    Each line is a tweet JSON object.
    """
    with open(dump_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_str in f:
            try:
                obj = json.loads(line_str)
            except json.JSONDecodeError:
                continue

            text = obj.get("text", "")
            if not text:
                continue

            text = clean_text(text)
            if 3 < len(text.split()) < 200:  # tweets are shorter, lower bound
                yield TextRecord(
                    text=text,
                    timestamp=obj.get("created_at", ""),
                    source_url=f"https://twitter.com/i/status/{obj.get('id', '')}",
                    source="Twitter",
                )


# --- Generic file (for testing with local text files) ---

def stream_text_file(path: str, source_name: str = "local") -> Iterator[TextRecord]:
    """Simple line-by-line streaming for local test files."""
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = clean_text(line)
            if 5 < len(line.split()) < 200:
                yield TextRecord(
                    text=line,
                    timestamp="",
                    source_url=path,
                    source=source_name,
                )


def batch_records(
    stream: Iterator[TextRecord], batch_size: int
) -> Iterator[list[TextRecord]]:
    """Collect records into batches for efficient GPU encoding."""
    batch = []
    for record in stream:
        batch.append(record)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# ============================================================================
# 4. PHRASE EXTRACTION METHODS
# ============================================================================

# ---------------------------------------------------------------------------
# Method A: Perturbation-Based (model-agnostic, interpretable)
# ---------------------------------------------------------------------------
#
# How it works:
#   For each n-gram in the sentence, remove it, re-encode the sentence,
#   and measure how much the cosine similarity to the taboo anchor drops.
#   The n-gram whose removal causes the biggest drop is most responsible
#   for the euphemistic signal.
#
# Pros:
#   - Works with ANY encoder (black box)
#   - Highly interpretable: "removing X caused similarity to drop by Y"
#   - Easy to explain in a paper
#   - Naturally produces n-gram candidates (not just tokens)
#
# Cons:
#   - O(n * max_ngram_size) forward passes per sentence — expensive
#   - Assumes independence: removing "passed" and "away" separately
#     won't show that "passed away" is the unit unless you test bigrams
#   - Deletion changes sentence structure, which may confuse the encoder
#
# When to use: Default choice. Best for interpretability and simplicity.
# ---------------------------------------------------------------------------

def extract_phrase_perturbation(
    sentence: str,
    anchor_vec: np.ndarray,
    model: SentenceTransformer,
    max_n: int = 4,
    min_drop: float = 0.05,
) -> Optional[dict]:
    """
    Perturbation-based phrase extraction.

    Args:
        sentence:   the flagged sentence
        anchor_vec: embedding of the matched taboo anchor (1, dim), normalized
        model:      the sentence transformer
        max_n:      largest n-gram to test
        min_drop:   minimum similarity drop to consider meaningful

    Returns:
        dict with keys: phrase, drop, ngram_size — or None if nothing found
    """
    words = sentence.split()
    if len(words) < 2:
        return None

    # Baseline: full sentence similarity
    baseline_vec = model.encode(sentence, convert_to_numpy=True).reshape(1, -1)
    baseline_vec = baseline_vec / np.linalg.norm(baseline_vec)
    baseline_score = float(np.dot(baseline_vec, anchor_vec.T).squeeze())

    candidates = []

    for n in range(1, min(max_n + 1, len(words) + 1)):
        # Batch all perturbations for this n-gram size
        perturbed_sentences = []
        ngrams = []

        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i : i + n])
            perturbed = " ".join(words[:i] + words[i + n :])
            if not perturbed.strip():
                continue
            ngrams.append(ngram)
            perturbed_sentences.append(perturbed)

        if not perturbed_sentences:
            continue

        # Encode all perturbations in one batch (efficient!)
        perturbed_vecs = model.encode(
            perturbed_sentences, convert_to_numpy=True, batch_size=64
        )
        norms = np.linalg.norm(perturbed_vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # avoid division by zero
        perturbed_vecs = perturbed_vecs / norms

        scores = np.dot(perturbed_vecs, anchor_vec.T).squeeze()
        if scores.ndim == 0:
            scores = np.array([float(scores)])

        for j, (ngram, score) in enumerate(zip(ngrams, scores)):
            drop = baseline_score - float(score)
            if drop > min_drop:
                candidates.append({
                    "phrase": ngram,
                    "drop": drop,
                    "ngram_size": n,
                })

    if not candidates:
        return None

    # Sort by drop descending, break ties by preferring longer n-grams
    # (a bigram that drops as much as a unigram is more informative)
    candidates.sort(key=lambda x: (x["drop"], x["ngram_size"]), reverse=True)

    # Check for overlap: if top bigram contains top unigram with similar
    # score, prefer the bigram
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
# Method B: Integrated Gradients (gradient-based, more principled)
# ---------------------------------------------------------------------------
#
# How it works:
#   Compute gradients of the cosine similarity w.r.t. the input token
#   embeddings, accumulated along a straight-line path from a zero baseline
#   to the actual embedding. This gives per-token attribution scores.
#
# Pros:
#   - Theoretically grounded (satisfies completeness axiom)
#   - Captures token interactions the perturbation method misses
#   - Single forward+backward pass (plus integration steps)
#
# Cons:
#   - Requires access to model internals (not a black box)
#   - SentenceTransformer wraps HuggingFace models, so we need to unwrap
#   - Produces TOKEN-level attributions, not n-gram-level — need
#     post-processing to aggregate into phrases
#   - Harder to explain in a paper (reviewers may question IG on
#     a similarity metric rather than a classification logit)
#   - Integration steps (n_steps) trade accuracy for speed
#
# When to use: When you want a more rigorous attribution and are willing
#   to handle the engineering complexity. Good as a comparison/validation
#   of the perturbation method.
# ---------------------------------------------------------------------------

def extract_phrase_ig(
    sentence: str,
    anchor_vec_tensor: torch.Tensor,
    model: SentenceTransformer,
    n_steps: int = 50,
    max_ngram_size: int = 4,
) -> Optional[dict]:
    """
    Integrated Gradients phrase extraction.

    This unwraps the SentenceTransformer to access the underlying
    HuggingFace transformer, computes IG attributions at the token level,
    then aggregates into n-gram scores.

    Args:
        sentence:          the flagged sentence
        anchor_vec_tensor: embedding of the taboo anchor as a torch tensor (1, dim)
        model:             the SentenceTransformer
        n_steps:           number of integration steps (higher = more accurate)
        max_ngram_size:    max n-gram window for aggregation

    Returns:
        dict with keys: phrase, attribution_score, ngram_size — or None
    """
    device = next(model.parameters()).device

    # Unwrap the SentenceTransformer:
    #   model[0] = the HuggingFace Transformer wrapper
    #   model[0].auto_model = the actual transformer (BertModel, etc.)
    #   model[1] = the pooling layer
    transformer = model[0].auto_model
    tokenizer = model[0].tokenizer
    pooling = model[1]

    # Tokenize
    inputs = tokenizer(
        sentence, return_tensors="pt", padding=True, truncation=True
    ).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Get the embedding layer
    embedding_layer = transformer.embeddings.word_embeddings
    actual_embeddings = embedding_layer(input_ids).detach()  # (1, seq_len, hidden_dim)

    # Baseline: zero embeddings (standard IG baseline)
    baseline_embeddings = torch.zeros_like(actual_embeddings)

    # Anchor vector on same device
    anchor_vec_tensor = anchor_vec_tensor.to(device)

    # Forward function: embeddings -> cosine similarity with anchor
    def forward_from_embeddings(embeddings):
        # Pass through transformer (bypassing the embedding layer)
        # We need to add position embeddings and layer norm manually
        position_ids = torch.arange(
            embeddings.shape[1], device=device
        ).unsqueeze(0)
        position_embeddings = transformer.embeddings.position_embeddings(position_ids)
        token_type_ids = torch.zeros_like(input_ids)
        token_type_embeddings = transformer.embeddings.token_type_embeddings(
            token_type_ids
        )

        combined = embeddings + position_embeddings + token_type_embeddings
        combined = transformer.embeddings.LayerNorm(combined)
        combined = transformer.embeddings.dropout(combined)

        # Run through encoder layers
        encoder_output = transformer.encoder(
            combined, attention_mask=attention_mask.unsqueeze(1).unsqueeze(2).float()
        )
        hidden_state = encoder_output.last_hidden_state  # (1, seq_len, hidden_dim)

        # Pool (mean pooling, matching SentenceTransformer behavior)
        mask_expanded = attention_mask.unsqueeze(-1).float()
        summed = (hidden_state * mask_expanded).sum(dim=1)
        count = mask_expanded.sum(dim=1)
        pooled = summed / count  # (1, hidden_dim)

        # Normalize
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

        # Cosine similarity with anchor
        similarity = torch.nn.functional.cosine_similarity(
            pooled, anchor_vec_tensor.reshape(1, -1)
        )
        return similarity

    # Integrated Gradients: accumulate gradients along the path
    # from baseline to actual embeddings
    all_grads = []
    for step in range(n_steps + 1):
        alpha = step / n_steps
        interpolated = baseline_embeddings + alpha * (
            actual_embeddings - baseline_embeddings
        )
        interpolated.requires_grad_(True)

        similarity = forward_from_embeddings(interpolated)
        similarity.backward()

        all_grads.append(interpolated.grad.detach().clone())
        interpolated.grad = None

    # Average gradients * (actual - baseline)
    avg_grads = torch.stack(all_grads).mean(dim=0)
    integrated_grads = (actual_embeddings - baseline_embeddings) * avg_grads

    # Sum across embedding dim to get per-token attribution
    token_attributions = integrated_grads.sum(dim=-1).squeeze(0)  # (seq_len,)

    # Map tokens back to words and aggregate
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
    # Skip [CLS] and [SEP]
    word_scores = []
    current_word = ""
    current_score = 0.0

    for i, token in enumerate(tokens):
        if token in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        if token.startswith("##"):
            # Subword continuation
            current_word += token[2:]
            current_score += token_attributions[i].item()
        else:
            if current_word:
                word_scores.append((current_word, current_score))
            current_word = token
            current_score = token_attributions[i].item()
    if current_word:
        word_scores.append((current_word, current_score))

    if not word_scores:
        return None

    # Find best n-gram by summing attribution scores
    candidates = []
    for n in range(1, min(max_ngram_size + 1, len(word_scores) + 1)):
        for i in range(len(word_scores) - n + 1):
            phrase = " ".join(w for w, _ in word_scores[i : i + n])
            score = sum(s for _, s in word_scores[i : i + n])
            candidates.append({
                "phrase": phrase,
                "attribution_score": score,
                "ngram_size": n,
            })

    candidates.sort(key=lambda x: x["attribution_score"], reverse=True)

    if candidates and candidates[0]["attribution_score"] > 0:
        return candidates[0]

    return None


# ============================================================================
# 5. CORE FIRST PASS
# ============================================================================

@dataclass
class EuphemismCandidate:
    """Structured output for each detected candidate."""
    text: str                          # The extracted euphemistic phrase
    context: str                       # The full sentence
    taboo_anchor: str                  # Which anchor it matched
    taboo_category: str                # Category of the anchor
    similarity_score: float            # Cosine similarity (sentence-level)
    phrase_extraction_method: str = ""  # "perturbation" | "ig"
    phrase_drop_score: float = 0.0     # How much removing the phrase dropped similarity
    timestamp: str = ""
    source_url: str = ""
    source: str = ""
    all_anchor_matches: list = field(default_factory=list)  # Top-k anchor matches


def first_pass(
    data_stream: Iterator[TextRecord],
    model: SentenceTransformer,
    anchor_texts: list[str],
    anchor_labels: list[str],
    anchor_categories: list[str],
    faiss_index: faiss.Index,
    anchor_embeddings: np.ndarray,
    config: PipelineConfig,
) -> Iterator[EuphemismCandidate]:
    """
    Stage A: Sentence-level filtering via FAISS.
    Stage B: Phrase extraction on hits.
    Yields EuphemismCandidate objects.
    """

    # Precompute anchor vectors as torch tensors for IG method
    anchor_tensors = None
    if config.phrase_method in ("ig", "both"):
        anchor_tensors = torch.from_numpy(anchor_embeddings).to(config.device)

    total_processed = 0
    total_hits = 0

    for batch in batch_records(data_stream, config.sentence_batch_size):
        sentences = [r.text for r in batch]

        # --- Stage A: Sentence-level similarity ---
        batch_embeddings = model.encode(
            sentences, convert_to_numpy=True, batch_size=config.sentence_batch_size
        ).astype("float32")
        faiss.normalize_L2(batch_embeddings)

        scores, indices = faiss_index.search(batch_embeddings, config.faiss_k)

        total_processed += len(batch)
        if total_processed % 10000 == 0:
            logger.info(
                f"Processed {total_processed} sentences, {total_hits} hits so far"
            )

        for i, record in enumerate(batch):
            # Check each of the k nearest anchors
            for rank in range(config.faiss_k):
                score = float(scores[i][rank])
                anchor_idx = int(indices[i][rank])
                category = anchor_categories[anchor_idx]
                label = anchor_labels[anchor_idx]

                # Apply threshold (global or per-category)
                if config.use_per_category_thresholds:
                    threshold = CATEGORY_THRESHOLDS.get(
                        category, config.global_threshold
                    )
                else:
                    threshold = config.global_threshold

                if score < threshold:
                    continue  # Below threshold, skip

                # Exclusion: if the taboo word itself appears in the sentence,
                # it's not a euphemism — it's a direct reference
                if label.lower() in record.text.lower():
                    continue

                # --- Stage B: Phrase extraction ---
                anchor_vec = anchor_embeddings[anchor_idx].reshape(1, -1)

                phrase_result_perturbation = None
                phrase_result_ig = None

                if config.phrase_method in ("perturbation", "both"):
                    phrase_result_perturbation = extract_phrase_perturbation(
                        record.text,
                        anchor_vec,
                        model,
                        max_n=config.max_ngram_size,
                        min_drop=config.min_drop_threshold,
                    )

                if config.phrase_method in ("ig", "both") and anchor_tensors is not None:
                    phrase_result_ig = extract_phrase_ig(
                        record.text,
                        anchor_tensors[anchor_idx].unsqueeze(0),
                        model,
                        max_ngram_size=config.max_ngram_size,
                    )

                # Decide which phrase to use
                phrase_text = record.text  # fallback: whole sentence
                phrase_method = "none"
                phrase_score = 0.0

                if config.phrase_method == "perturbation" and phrase_result_perturbation:
                    phrase_text = phrase_result_perturbation["phrase"]
                    phrase_method = "perturbation"
                    phrase_score = phrase_result_perturbation["drop"]
                elif config.phrase_method == "ig" and phrase_result_ig:
                    phrase_text = phrase_result_ig["phrase"]
                    phrase_method = "ig"
                    phrase_score = phrase_result_ig["attribution_score"]
                elif config.phrase_method == "both":
                    # For "both", store the perturbation result as primary
                    # (more interpretable) but include IG in metadata
                    if phrase_result_perturbation:
                        phrase_text = phrase_result_perturbation["phrase"]
                        phrase_method = "perturbation"
                        phrase_score = phrase_result_perturbation["drop"]

                # Collect all anchor matches for this sentence (for analysis)
                all_matches = []
                for rank2 in range(config.faiss_k):
                    s = float(scores[i][rank2])
                    idx = int(indices[i][rank2])
                    cat = anchor_categories[idx]
                    thresh = (
                        CATEGORY_THRESHOLDS.get(cat, config.global_threshold)
                        if config.use_per_category_thresholds
                        else config.global_threshold
                    )
                    if s >= thresh:
                        all_matches.append({
                            "anchor": anchor_labels[idx],
                            "category": anchor_categories[idx],
                            "score": s,
                        })

                total_hits += 1

                candidate = EuphemismCandidate(
                    text=phrase_text,
                    context=record.text,
                    taboo_anchor=label,
                    taboo_category=category,
                    similarity_score=score,
                    phrase_extraction_method=phrase_method,
                    phrase_drop_score=phrase_score,
                    timestamp=record.timestamp,
                    source_url=record.source_url,
                    source=record.source,
                    all_anchor_matches=all_matches,
                )

                yield candidate

                # Only yield once per sentence per anchor rank
                # (avoid duplicates from the k loop)
                break  # Remove this break if you want multi-category output

    logger.info(f"First pass complete: {total_processed} sentences, {total_hits} hits")


# ============================================================================
# 6. OUTPUT
# ============================================================================

def write_results(
    candidates: Iterator[EuphemismCandidate],
    output_path: str,
):
    """Write candidates to JSONL for the second pass."""
    count = 0
    with open(output_path, "w") as f:
        for candidate in candidates:
            f.write(json.dumps(asdict(candidate)) + "\n")
            count += 1
            if count % 1000 == 0:
                logger.info(f"Written {count} candidates")
    logger.info(f"Output complete: {count} candidates written to {output_path}")


# ============================================================================
# 7. MAIN — RUNNABLE ENTRY POINT
# ============================================================================

def run_pipeline(
    sources: list[Iterator[TextRecord]],
    config: PipelineConfig = PipelineConfig(),
    taboo_path: str = "taboo_words_refined.py",
):
    """
    Full first-pass pipeline.

    Args:
        sources:    list of TextRecord iterators (one per data source)
        config:     pipeline configuration
        taboo_path: path to the taboo anchors Python file
    """
    import itertools

    # 1. Load model
    logger.info(f"Loading model: {config.model_name} on {config.device}")
    model = SentenceTransformer(config.model_name, device=config.device)

    # 2. Load and prepare anchors
    logger.info("Loading taboo anchors...")
    taboo_dict = load_taboo_anchors(taboo_path)
    anchor_texts, anchor_labels, anchor_categories = prepare_anchors(
        taboo_dict, template=config.anchor_template
    )
    logger.info(
        f"Prepared {len(anchor_texts)} anchors across "
        f"{len(set(anchor_categories))} categories"
    )

    # 3. Encode anchors
    logger.info("Encoding anchors...")
    anchor_embeddings = model.encode(
        anchor_texts, convert_to_numpy=True
    ).astype("float32")
    faiss.normalize_L2(anchor_embeddings)

    # 4. Build FAISS index
    faiss_index = build_faiss_index(
        anchor_embeddings.copy(),  # copy because normalize_L2 is in-place
        use_gpu=config.use_gpu_faiss,
    )

    # 5. Chain all sources into one stream
    combined_stream = itertools.chain(*sources)

    # 6. Run first pass
    logger.info("Starting first pass...")
    candidates = first_pass(
        data_stream=combined_stream,
        model=model,
        anchor_texts=anchor_texts,
        anchor_labels=anchor_labels,
        anchor_categories=anchor_categories,
        faiss_index=faiss_index,
        anchor_embeddings=anchor_embeddings,
        config=config,
    )

    # 7. Write output
    write_results(candidates, config.output_path)


# ============================================================================
# 8. EXPERIMENT RUNNER — For the paper
# ============================================================================

def run_experiments(
    sources: list[Iterator[TextRecord]],
    taboo_path: str = "taboo_words_refined.py",
):
    """
    Run multiple configurations and save results separately
    for comparison in the paper.
    """
    experiments = [
        # Experiment 1: Bare anchors, global threshold, perturbation
        PipelineConfig(
            anchor_template="none",
            use_per_category_thresholds=False,
            phrase_method="perturbation",
            faiss_k=5,
            output_path="results_bare_global_perturbation.jsonl",
        ),
        # Experiment 2: Template-wrapped anchors ("taboo"), global threshold
        PipelineConfig(
            anchor_template="taboo",
            use_per_category_thresholds=False,
            phrase_method="perturbation",
            faiss_k=5,
            output_path="results_taboo_global_perturbation.jsonl",
        ),
        # Experiment 3: Template-wrapped ("euphemism"), global threshold
        PipelineConfig(
            anchor_template="euphemism",
            use_per_category_thresholds=False,
            phrase_method="perturbation",
            faiss_k=5,
            output_path="results_euphemism_global_perturbation.jsonl",
        ),
        # Experiment 4: Bare anchors, per-category thresholds
        PipelineConfig(
            anchor_template="none",
            use_per_category_thresholds=True,
            phrase_method="perturbation",
            faiss_k=5,
            output_path="results_bare_percat_perturbation.jsonl",
        ),
        # Experiment 5: Best template + IG comparison
        PipelineConfig(
            anchor_template="taboo",
            use_per_category_thresholds=True,
            phrase_method="both",
            faiss_k=5,
            output_path="results_taboo_percat_both.jsonl",
        ),
        # Experiment 6: k=1 vs k=5 comparison
        PipelineConfig(
            anchor_template="taboo",
            use_per_category_thresholds=True,
            phrase_method="perturbation",
            faiss_k=1,
            output_path="results_taboo_percat_k1.jsonl",
        ),
    ]

    for i, config in enumerate(experiments):
        logger.info(f"\n{'='*60}")
        logger.info(f"EXPERIMENT {i+1}: {config.output_path}")
        logger.info(f"  template={config.anchor_template}")
        logger.info(f"  per_cat_thresh={config.use_per_category_thresholds}")
        logger.info(f"  phrase_method={config.phrase_method}")
        logger.info(f"  faiss_k={config.faiss_k}")
        logger.info(f"{'='*60}")

        # NOTE: You'll need to re-create source iterators for each experiment
        # since Python iterators are consumed after one pass. Either:
        #   a) re-instantiate the source functions here, or
        #   b) cache the data to disk first, then stream from cache each time.
        # For large corpora, (b) is recommended.
        run_pipeline(sources, config, taboo_path)


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # --- Single run with local test file ---
    config = PipelineConfig(
        anchor_template="taboo",
        use_per_category_thresholds=True,
        phrase_method="perturbation",
        faiss_k=5,
        output_path="first_pass_results.jsonl",
    )

    # Option A: Test with a local text file
    # sources = [stream_text_file("test_corpus.txt", source_name="test")]

    # Option B: Common Crawl
    # wet_url = "https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-10/segments/..."
    # sources = [stream_common_crawl_wet(wet_url)]

    # Option C: Multiple sources
    # sources = [
    #     stream_common_crawl_wet(wet_url),
    #     stream_wikipedia_dump("enwiki-latest-pages-articles.xml.bz2"),
    #     stream_reddit_dump("RC_2024-01.zst"),
    #     stream_twitter_dump("tweets_2024.jsonl"),
    # ]

    # run_pipeline(sources, config)

    # --- Or run all experiments ---
    # run_experiments(sources)

    print("Uncomment a source and run_pipeline() call above to start.")