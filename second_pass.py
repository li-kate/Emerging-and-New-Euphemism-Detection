"""
============================================================================
SECOND PASS: EXHAUSTIVE EUPHEMISM INSTANCE COLLECTION
============================================================================

Goal: Given the candidate euphemisms discovered in the first pass, scan the
full corpus again to find EVERY instance of those phrases with rich context.

Architecture:
    1. Load first-pass candidates from JSONL
    2. Build a multi-pattern string matcher (Aho-Corasick)
    3. Optionally expand candidates with morphological variants
    4. Stream the full corpus again, matching all patterns simultaneously
    5. For each match, extract configurable context (±N sentences)
    6. Output structured records to JSONL for the learning stage

Depends on:
    - data_sources.py (TextRecord, SourceConfig, stream_* functions)
    - First pass output (JSONL from first_pass.py)

Required: pip install ahocorasick-rs   (Rust-backed, fastest option)
     or:  pip install pyahocorasick    (C-backed, also fast)
============================================================================
"""

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Iterator, Optional

from data_sources import TextRecord, SourceConfig, batch_records

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# 0. CONFIGURATION
# ============================================================================

@dataclass
class SecondPassConfig:
    """
    Configuration for the second pass.

    Separate from PipelineConfig because the concerns are different:
    first pass = similarity search parameters, second pass = string
    matching and context extraction parameters.
    """

    # --- Input ---
    first_pass_results_path: str = "first_pass_results.jsonl"

    # --- Candidate filtering ---
    # Minimum similarity score from the first pass to include a candidate.
    # Raising this focuses the second pass on higher-confidence candidates.
    # Lowering it captures more candidates but increases false positives.
    min_first_pass_score: float = 0.55

    # Minimum phrase drop score (from perturbation extraction).
    # Phrases with low drop scores may not actually be the euphemistic
    # component of their sentence. 0.0 = include all, 0.05 = conservative.
    min_phrase_drop_score: float = 0.0

    # Deduplicate candidate phrases that appear with multiple anchors?
    # If True, "passed away" matched to both "death" and "funeral" is
    # stored once with both anchor associations. If False, separate entries.
    # True is recommended — the learning stage can handle multi-anchor phrases.
    deduplicate_candidates: bool = True

    # --- Matching mode ---
    # "exact":  Match the exact phrase as discovered in the first pass.
    #           Fast, precise, but misses morphological variants.
    #           "passed away" won't match "passing away".
    #
    # "lemma":  Expand each candidate phrase into morphological variants
    #           using lemmatization, then match all variants.
    #           Slower setup (needs spaCy), but catches inflected forms.
    #           "passed away" also matches "passing away", "passes away",
    #           "pass away".
    #
    # "stem":   Like lemma but uses Porter stemming instead of spaCy.
    #           Faster, cruder, more aggressive (may over-generate variants).
    #           "passed" stems to "pass", matching "passing", "passage", etc.
    #
    # Recommendation: Start with "exact" to establish a baseline, then
    # run "lemma" to measure how many additional instances you recover.
    # Report both in the paper.
    match_mode: str = "exact"  # "exact" | "lemma" | "stem"

    # --- Case sensitivity ---
    # False (case-insensitive) is almost always correct for euphemism
    # detection. Euphemisms don't change meaning with capitalization.
    # Exception: acronyms where case matters (rare for euphemisms).
    case_sensitive: bool = False

    # --- Context extraction ---
    # How many sentences of context to include around each match.
    # The "sentence" here means the text record from the source stream
    # (typically one line or one natural sentence).
    #
    # 0 = only the sentence containing the match
    # 1 = ±1 sentences (3 total: before, match, after)
    # 2 = ±2 sentences (5 total)
    #
    # More context helps the learning stage understand usage patterns.
    # But too much context dilutes the signal and increases storage.
    context_sentences: int = 1

    # --- Output ---
    output_path: str = "second_pass_results.jsonl"


# ============================================================================
# 1. LOAD FIRST-PASS CANDIDATES
# ============================================================================

@dataclass
class CandidatePhrase:
    """
    A euphemism candidate from the first pass, ready for second-pass matching.
    Groups all anchor associations for a single phrase.
    """
    phrase: str                         # The phrase to search for
    variants: list[str] = field(default_factory=list)  # Morphological variants
    anchors: list[dict] = field(default_factory=list)   # [{anchor, category, score}]
    max_score: float = 0.0             # Highest first-pass similarity score
    max_drop: float = 0.0              # Highest phrase drop score
    first_seen_timestamp: str = ""     # Earliest timestamp from first pass
    first_seen_source: str = ""        # Source of earliest occurrence


def load_first_pass_candidates(
    path: str,
    config: SecondPassConfig,
) -> dict[str, CandidatePhrase]:
    """
    Load and deduplicate first-pass results into CandidatePhrase objects.

    Returns:
        dict mapping normalized phrase -> CandidatePhrase
    """
    candidates: dict[str, CandidatePhrase] = {}
    total_loaded = 0
    total_filtered = 0

    with open(path, "r") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            total_loaded += 1

            # Apply score filters
            if record.get("similarity_score", 0) < config.min_first_pass_score:
                total_filtered += 1
                continue

            if record.get("phrase_drop_score", 0) < config.min_phrase_drop_score:
                total_filtered += 1
                continue

            phrase = record.get("text", "").strip()
            if not phrase:
                total_filtered += 1
                continue

            # Normalize for deduplication
            key = phrase.lower() if not config.case_sensitive else phrase

            if key not in candidates:
                candidates[key] = CandidatePhrase(
                    phrase=phrase,
                    max_score=record.get("similarity_score", 0),
                    max_drop=record.get("phrase_drop_score", 0),
                    first_seen_timestamp=record.get("timestamp", ""),
                    first_seen_source=record.get("source", ""),
                )

            cp = candidates[key]

            # Track the best scores across all occurrences
            score = record.get("similarity_score", 0)
            drop = record.get("phrase_drop_score", 0)
            if score > cp.max_score:
                cp.max_score = score
            if drop > cp.max_drop:
                cp.max_drop = drop

            # Aggregate anchor associations
            anchor_info = {
                "anchor": record.get("taboo_anchor", ""),
                "category": record.get("taboo_category", ""),
                "score": score,
            }
            # Avoid duplicate anchor entries
            if not any(
                a["anchor"] == anchor_info["anchor"] for a in cp.anchors
            ):
                cp.anchors.append(anchor_info)

    logger.info(
        f"Loaded {total_loaded} first-pass records, "
        f"filtered {total_filtered}, "
        f"yielded {len(candidates)} unique candidate phrases"
    )
    return candidates


# ============================================================================
# 2. MORPHOLOGICAL VARIANT EXPANSION
# ============================================================================
#
# Why this matters:
#   The first pass might find "passed away" but the corpus also contains
#   "passing away", "passes away", and "pass away". These are all the same
#   euphemism in different inflected forms. Without variant expansion,
#   the second pass would miss ~40-60% of occurrences (rough estimate
#   based on English verb morphology alone).
#
# Two approaches:
#
# A. Lemma-based (spaCy):
#    Lemmatize the candidate phrase, then generate likely inflections.
#    More linguistically principled, handles irregular forms.
#    Requires spaCy + English model (~15MB for en_core_web_sm).
#    Speed: ~100K words/sec (fast enough for candidate expansion,
#    which is only run once on a small set of candidates, not the corpus).
#
# B. Stem-based (Porter stemmer):
#    Stem each word, then at match time, stem the corpus word and compare
#    stems. Faster, no model download, but cruder:
#    - "passed" -> "pass", matches "passage", "passenger" (false positives)
#    - Doesn't handle irregular forms well
#    Stems are compared at search time, not expansion time, so this
#    changes the matching logic rather than expanding the pattern list.
#
# Recommendation: Lemma for precision, stem for recall. Report both.
# ============================================================================

def expand_variants_lemma(
    candidates: dict[str, CandidatePhrase],
) -> dict[str, CandidatePhrase]:
    """
    Expand candidate phrases with morphological variants using spaCy.

    For each candidate phrase:
        1. Lemmatize it to get the base form
        2. Generate common English inflections of each word
        3. Store all variants for Aho-Corasick matching

    Requires: pip install spacy && python -m spacy download en_core_web_sm
    """
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except (ImportError, OSError):
        logger.error(
            "spaCy not available for lemma expansion. "
            "Install: pip install spacy && python -m spacy download en_core_web_sm. "
            "Falling back to exact matching."
        )
        return candidates

    # English inflection patterns for common POS
    # These are heuristic — they won't cover every irregular form,
    # but they capture the most common euphemism-relevant inflections.
    def inflect_verb(lemma: str) -> set[str]:
        """Generate common verb inflections from a lemma."""
        forms = {lemma}
        # Regular patterns
        if lemma.endswith("e"):
            forms.add(lemma + "d")      # "use" -> "used"
            forms.add(lemma + "s")      # "use" -> "uses"
            forms.add(lemma[:-1] + "ing")  # "use" -> "using"
        elif lemma.endswith("y") and len(lemma) > 2 and lemma[-2] not in "aeiou":
            forms.add(lemma[:-1] + "ied")   # "carry" -> "carried"
            forms.add(lemma[:-1] + "ies")   # "carry" -> "carries"
            forms.add(lemma + "ing")         # "carry" -> "carrying"
        else:
            forms.add(lemma + "ed")     # "pass" -> "passed"
            forms.add(lemma + "s")      # "pass" -> "passes"
            forms.add(lemma + "es")     # "pass" -> "passes"
            forms.add(lemma + "ing")    # "pass" -> "passing"
        return forms

    def inflect_noun(lemma: str) -> set[str]:
        """Generate common noun inflections."""
        forms = {lemma}
        if lemma.endswith("s") or lemma.endswith("x") or lemma.endswith("z"):
            forms.add(lemma + "es")
        elif lemma.endswith("y") and len(lemma) > 2 and lemma[-2] not in "aeiou":
            forms.add(lemma[:-1] + "ies")
        else:
            forms.add(lemma + "s")
        return forms

    def inflect_adj(lemma: str) -> set[str]:
        """Generate common adjective inflections."""
        forms = {lemma}
        if lemma.endswith("e"):
            forms.add(lemma + "r")
            forms.add(lemma + "st")
        elif lemma.endswith("y") and len(lemma) > 2:
            forms.add(lemma[:-1] + "ier")
            forms.add(lemma[:-1] + "iest")
        else:
            forms.add(lemma + "er")
            forms.add(lemma + "est")
        return forms

    inflectors = {
        "VERB": inflect_verb,
        "NOUN": inflect_noun,
        "ADJ": inflect_adj,
    }

    expanded_count = 0

    for key, candidate in candidates.items():
        doc = nlp(candidate.phrase)
        # For multi-word phrases, generate the Cartesian product of
        # inflections for each word. For "passed away":
        #   "passed" (VERB) -> {pass, passed, passes, passing}
        #   "away" (ADV) -> {away}  (adverbs don't inflect)
        # Result: "pass away", "passed away", "passes away", "passing away"

        word_variants = []
        for token in doc:
            lemma = token.lemma_.lower()
            pos = token.pos_

            if pos in inflectors:
                forms = inflectors[pos](lemma)
                # Also include the original surface form
                forms.add(token.text.lower())
                word_variants.append(list(forms))
            else:
                # Non-inflecting word (adverb, preposition, etc.)
                word_variants.append([token.text.lower()])

        # Generate Cartesian product of all word variants
        # For efficiency, cap at reasonable number (avoid combinatorial explosion
        # with long phrases where every word inflects)
        from itertools import product as cartesian_product

        all_combos = list(cartesian_product(*word_variants))
        # Safety cap: if a phrase has 4 inflecting words with 5 forms each,
        # that's 625 variants. Cap at 100 to avoid bloating the matcher.
        if len(all_combos) > 100:
            logger.warning(
                f"Phrase '{candidate.phrase}' generated {len(all_combos)} variants, "
                f"capping at 100"
            )
            all_combos = all_combos[:100]

        variants = [" ".join(combo) for combo in all_combos]
        # Remove the original phrase (it's already the primary)
        variants = [v for v in variants if v != key]

        if variants:
            candidate.variants = variants
            expanded_count += len(variants)

    logger.info(
        f"Lemma expansion: {expanded_count} variants generated "
        f"for {len(candidates)} candidates"
    )
    return candidates


def expand_variants_stem(
    candidates: dict[str, CandidatePhrase],
) -> dict[str, CandidatePhrase]:
    """
    For stem mode, we don't expand patterns — instead we store the stemmed
    form alongside each candidate. Matching happens by stemming corpus
    n-grams at search time and comparing stems.

    This is a different paradigm: instead of expanding the pattern set,
    we generalize the matching function.

    Uses NLTK Porter stemmer (fast, no model download).
    Requires: pip install nltk
    """
    try:
        from nltk.stem import PorterStemmer
    except ImportError:
        logger.error("pip install nltk for stem mode")
        return candidates

    stemmer = PorterStemmer()

    for key, candidate in candidates.items():
        words = candidate.phrase.lower().split()
        stemmed = " ".join(stemmer.stem(w) for w in words)
        # Store the stem as the single "variant" — matching logic
        # in the search phase will stem corpus text and compare
        candidate.variants = [f"__STEM__:{stemmed}"]

    logger.info(f"Stem forms computed for {len(candidates)} candidates")
    return candidates


# ============================================================================
# 3. MULTI-PATTERN MATCHER (AHO-CORASICK)
# ============================================================================
#
# Why Aho-Corasick:
#   Naive approach: for each sentence, check if any of N patterns appear.
#   Cost: O(sentence_length × N × avg_pattern_length).
#   With 500 candidate phrases and billions of sentences, this is too slow.
#
#   Aho-Corasick builds a finite automaton from all patterns, then scans
#   the text ONCE, finding all matches simultaneously.
#   Cost: O(sentence_length + num_matches). Independent of pattern count.
#
#   For 500 patterns across a billion sentences, Aho-Corasick is ~500x faster.
#
# Library choice:
#   ahocorasick-rs: Rust-backed, fastest. pip install ahocorasick-rs
#   pyahocorasick:  C-backed, also fast. pip install pyahocorasick
#   We try ahocorasick-rs first, fall back to pyahocorasick.
# ============================================================================

class PatternMatcher:
    """
    Wrapper around Aho-Corasick that maps match positions back to
    candidate phrases and their metadata.
    """

    def __init__(
        self,
        candidates: dict[str, CandidatePhrase],
        case_sensitive: bool = False,
    ):
        self.case_sensitive = case_sensitive
        # Map: pattern_string -> CandidatePhrase key
        # Multiple patterns can map to the same candidate (variants)
        self.pattern_to_key: dict[str, str] = {}
        self.candidates = candidates

        patterns = []
        for key, cp in candidates.items():
            # Primary phrase
            p = cp.phrase if case_sensitive else cp.phrase.lower()
            self.pattern_to_key[p] = key
            patterns.append(p)

            # Variants (from lemma expansion)
            for variant in cp.variants:
                if variant.startswith("__STEM__:"):
                    continue  # Stem variants use different matching logic
                v = variant if case_sensitive else variant.lower()
                if v not in self.pattern_to_key:
                    self.pattern_to_key[v] = key
                    patterns.append(v)

        self.patterns = patterns
        self._automaton = None
        self._backend = None
        self._build_automaton()

        logger.info(
            f"Pattern matcher built: {len(candidates)} candidates, "
            f"{len(patterns)} total patterns (including variants)"
        )

    def _build_automaton(self):
        """Try Rust backend first, fall back to C backend."""
        # Try ahocorasick-rs (Rust)
        try:
            import ahocorasick_rs
            self._automaton = ahocorasick_rs.AhoCorasick(
                self.patterns,
                match_kind=ahocorasick_rs.MatchKind.LeftmostLongest,
            )
            self._backend = "rust"
            logger.info("Using ahocorasick-rs (Rust) backend")
            return
        except ImportError:
            pass

        # Try pyahocorasick (C)
        try:
            import ahocorasick
            A = ahocorasick.Automaton()
            for idx, pattern in enumerate(self.patterns):
                A.add_word(pattern, (idx, pattern))
            A.make_automaton()
            self._automaton = A
            self._backend = "c"
            logger.info("Using pyahocorasick (C) backend")
            return
        except ImportError:
            pass

        # Fallback: pure Python (slow but works)
        logger.warning(
            "No Aho-Corasick library found. Using naive matching (SLOW). "
            "Install: pip install ahocorasick-rs  OR  pip install pyahocorasick"
        )
        self._backend = "naive"

    def find_matches(self, text: str) -> list[dict]:
        """
        Find all candidate phrase occurrences in text.

        Returns list of dicts:
            {
                "phrase": str,        # The matched phrase
                "candidate_key": str, # Key into self.candidates
                "start": int,         # Character offset start
                "end": int,           # Character offset end
            }
        """
        search_text = text if self.case_sensitive else text.lower()
        matches = []

        if self._backend == "rust":
            for match in self._automaton.find_overlapping(search_text):
                pattern_idx = match.pattern()
                start = match.start()
                end = match.end()
                pattern = self.patterns[pattern_idx]
                key = self.pattern_to_key[pattern]
                # Word boundary check: ensure we're not matching inside a word
                if _is_word_boundary(search_text, start, end):
                    matches.append({
                        "phrase": text[start:end],  # Preserve original casing
                        "candidate_key": key,
                        "start": start,
                        "end": end,
                    })

        elif self._backend == "c":
            for end_idx, (pattern_idx, pattern) in self._automaton.iter(search_text):
                start = end_idx - len(pattern) + 1
                end = end_idx + 1
                key = self.pattern_to_key[pattern]
                if _is_word_boundary(search_text, start, end):
                    matches.append({
                        "phrase": text[start:end],
                        "candidate_key": key,
                        "start": start,
                        "end": end,
                    })

        else:  # naive fallback
            for pattern in self.patterns:
                idx = 0
                while True:
                    pos = search_text.find(pattern, idx)
                    if pos == -1:
                        break
                    end = pos + len(pattern)
                    if _is_word_boundary(search_text, pos, end):
                        key = self.pattern_to_key[pattern]
                        matches.append({
                            "phrase": text[pos:end],
                            "candidate_key": key,
                            "start": pos,
                            "end": end,
                        })
                    idx = pos + 1

        return matches


class StemMatcher:
    """
    Alternative matcher for stem mode.

    Instead of matching exact strings, this stems both the patterns and
    the corpus text, then matches stemmed forms.

    Slower than Aho-Corasick on exact strings because we have to stem
    every word in the corpus at search time. But catches more variants
    than lemma expansion.
    """

    def __init__(
        self,
        candidates: dict[str, CandidatePhrase],
    ):
        from nltk.stem import PorterStemmer
        self.stemmer = PorterStemmer()
        self.candidates = candidates

        # Build stem -> candidate key mapping
        self.stem_to_key: dict[str, str] = {}
        self.stem_to_ngram_size: dict[str, int] = {}

        for key, cp in candidates.items():
            for variant in cp.variants:
                if variant.startswith("__STEM__:"):
                    stemmed = variant.replace("__STEM__:", "")
                    self.stem_to_key[stemmed] = key
                    self.stem_to_ngram_size[stemmed] = len(stemmed.split())

        logger.info(f"Stem matcher built: {len(self.stem_to_key)} stem patterns")

    def find_matches(self, text: str) -> list[dict]:
        """Find matches by stemming corpus text and comparing."""
        words = text.lower().split()
        stemmed_words = [self.stemmer.stem(w) for w in words]
        matches = []

        for stem_pattern, key in self.stem_to_key.items():
            n = self.stem_to_ngram_size[stem_pattern]
            stem_parts = stem_pattern.split()

            for i in range(len(stemmed_words) - n + 1):
                window_stems = stemmed_words[i:i + n]
                if window_stems == stem_parts:
                    # Found a match — recover the original surface form
                    original_phrase = " ".join(words[i:i + n])
                    # Approximate character offsets
                    start = len(" ".join(words[:i])) + (1 if i > 0 else 0)
                    end = start + len(original_phrase)

                    matches.append({
                        "phrase": original_phrase,
                        "candidate_key": key,
                        "start": start,
                        "end": end,
                    })

        return matches


def _is_word_boundary(text: str, start: int, end: int) -> bool:
    """
    Check that a match is at word boundaries (not inside a larger word).

    "pass" should match in "pass away" but NOT in "compass" or "passenger".

    We check the character immediately before start and after end.
    A word boundary is: start of string, end of string, or a non-alphanumeric char.
    """
    if start > 0 and text[start - 1].isalnum():
        return False
    if end < len(text) and text[end].isalnum():
        return False
    return True


# ============================================================================
# 4. CONTEXT EXTRACTION
# ============================================================================

class ContextBuffer:
    """
    Maintains a sliding window of recent sentences for context extraction.

    Why a buffer instead of random access:
        We're streaming the corpus — we can't go back to arbitrary positions.
        The buffer keeps the last N sentences in memory so we can provide
        "before" context. "After" context requires a small lookahead, which
        we handle by delaying output by context_size sentences.
    """

    def __init__(self, context_size: int = 1):
        self.context_size = context_size
        # Buffer holds (TextRecord, matches) tuples
        self.buffer: list[tuple[TextRecord, list[dict]]] = []
        # Track the current document to reset context at document boundaries
        self.current_doc_url: str = ""

    def add(
        self,
        record: TextRecord,
        matches: list[dict],
    ) -> list[tuple[TextRecord, list[dict], list[str], list[str]]]:
        """
        Add a record to the buffer and return any records ready for output.

        Returns list of:
            (record, matches, before_context, after_context)

        Records are output when we have enough "after" context, or when
        a document boundary is crossed.
        """
        # Detect document boundary (new source URL = new document)
        if record.source_url != self.current_doc_url:
            # Flush remaining buffered records from the previous document
            flushed = self._flush()
            self.current_doc_url = record.source_url
            self.buffer.clear()
            self.buffer.append((record, matches))
            return flushed

        self.buffer.append((record, matches))

        # Check if the oldest buffered record has enough after-context
        ready = []
        while len(self.buffer) > 2 * self.context_size + 1:
            ready.append(self._emit_oldest())

        return ready

    def flush(self) -> list[tuple[TextRecord, list[dict], list[str], list[str]]]:
        """Flush all remaining buffered records (end of stream)."""
        return self._flush()

    def _flush(self) -> list[tuple[TextRecord, list[dict], list[str], list[str]]]:
        """Emit all remaining records with whatever context is available."""
        results = []
        while self.buffer:
            results.append(self._emit_oldest())
        return results

    def _emit_oldest(self) -> tuple[TextRecord, list[dict], list[str], list[str]]:
        """Emit the oldest record with its surrounding context."""
        if not self.buffer:
            raise IndexError("Buffer is empty")

        idx = 0
        record, matches = self.buffer.pop(0)

        # Before context: records that were before this one
        # (they've already been popped, so we look at what's left... no)
        # Actually, we need to rethink: the buffer contains records IN ORDER.
        # The oldest (index 0) is the one to emit. Records before it have
        # already been emitted. So "before context" must be stored separately.

        # Simpler approach: just store text, not try to reconstruct from buffer
        # Let's use a different strategy entirely.
        return (record, matches, [], [])


class SimpleContextExtractor:
    """
    Simpler approach: accumulate sentences per document, then extract
    context for matched sentences in a single pass.

    This uses more memory (holds one document at a time) but is much
    simpler to implement correctly than a streaming buffer.

    Memory usage: O(sentences_per_document). Typical web pages have
    10-200 sentences, so this is negligible.
    """

    def __init__(self, context_size: int = 1):
        self.context_size = context_size

    def extract_contexts(
        self,
        doc_records: list[TextRecord],
        doc_matches: list[list[dict]],
    ) -> Iterator[dict]:
        """
        Given all sentences and matches for one document, yield output
        records with surrounding context.

        Args:
            doc_records: ordered list of TextRecords for one document
            doc_matches: parallel list of match results for each record
        """
        for i, (record, matches) in enumerate(zip(doc_records, doc_matches)):
            if not matches:
                continue

            # Extract context window
            start = max(0, i - self.context_size)
            end = min(len(doc_records), i + self.context_size + 1)

            before = [doc_records[j].text for j in range(start, i)]
            after = [doc_records[j].text for j in range(i + 1, end)]

            for match in matches:
                yield {
                    "record": record,
                    "match": match,
                    "before_context": before,
                    "after_context": after,
                }


# ============================================================================
# 5. STRUCTURED OUTPUT
# ============================================================================

@dataclass
class EuphemismInstance:
    """
    A single occurrence of a candidate euphemism in the corpus.
    This is the primary data structure fed to the learning stage.
    """
    # --- The match ---
    phrase: str                         # Matched phrase (surface form in text)
    canonical_phrase: str               # Canonical form from first pass

    # --- Context ---
    sentence: str                       # Sentence containing the match
    before_context: list[str]           # Preceding sentences
    after_context: list[str]            # Following sentences
    char_offset_start: int              # Character position in sentence
    char_offset_end: int

    # --- Taboo associations (from first pass) ---
    taboo_anchors: list[dict]           # [{anchor, category, score}]
    primary_category: str               # Highest-scoring category

    # --- First-pass scores ---
    first_pass_similarity: float        # Best similarity from first pass
    first_pass_phrase_drop: float       # Best phrase drop from first pass

    # --- Metadata ---
    timestamp: str
    source_url: str
    source: str

    # --- Matching metadata ---
    match_mode: str                     # "exact" | "lemma" | "stem"
    is_variant: bool                    # True if matched via a variant, not the canonical form


# ============================================================================
# 6. CORE SECOND PASS
# ============================================================================

def second_pass(
    data_stream: Iterator[TextRecord],
    candidates: dict[str, CandidatePhrase],
    config: SecondPassConfig,
) -> Iterator[EuphemismInstance]:
    """
    Scan the corpus for all instances of candidate euphemisms.

    Processes one document at a time (grouped by source_url) to enable
    context extraction across sentence boundaries.
    """
    # Build the appropriate matcher
    use_stem = config.match_mode == "stem" and any(
        any(v.startswith("__STEM__:") for v in cp.variants)
        for cp in candidates.values()
    )

    if use_stem:
        matcher = StemMatcher(candidates)
    else:
        matcher = PatternMatcher(candidates, case_sensitive=config.case_sensitive)

    context_extractor = SimpleContextExtractor(context_size=config.context_sentences)

    # Process documents (groups of records with the same source_url)
    current_doc_url = None
    doc_records: list[TextRecord] = []
    doc_matches: list[list[dict]] = []

    total_sentences = 0
    total_matches = 0
    total_instances = 0

    def process_document():
        nonlocal total_instances
        if not doc_records:
            return
        for ctx in context_extractor.extract_contexts(doc_records, doc_matches):
            record = ctx["record"]
            match = ctx["match"]
            key = match["candidate_key"]
            cp = candidates[key]

            # Determine primary category (highest-scoring anchor)
            primary = max(cp.anchors, key=lambda a: a["score"]) if cp.anchors else {}

            instance = EuphemismInstance(
                phrase=match["phrase"],
                canonical_phrase=cp.phrase,
                sentence=record.text,
                before_context=ctx["before_context"],
                after_context=ctx["after_context"],
                char_offset_start=match["start"],
                char_offset_end=match["end"],
                taboo_anchors=cp.anchors,
                primary_category=primary.get("category", ""),
                first_pass_similarity=cp.max_score,
                first_pass_phrase_drop=cp.max_drop,
                timestamp=record.timestamp,
                source_url=record.source_url,
                source=record.source,
                match_mode=config.match_mode,
                is_variant=(match["phrase"].lower() != cp.phrase.lower()),
            )
            total_instances += 1
            yield instance

    for record in data_stream:
        total_sentences += 1

        # Document boundary detection
        if record.source_url != current_doc_url:
            # Process the completed document
            yield from process_document()
            # Reset for new document
            current_doc_url = record.source_url
            doc_records = []
            doc_matches = []

        # Match against all patterns
        matches = matcher.find_matches(record.text)
        total_matches += len(matches)

        doc_records.append(record)
        doc_matches.append(matches)

        if total_sentences % 100000 == 0:
            logger.info(
                f"Second pass: {total_sentences} sentences, "
                f"{total_matches} matches, {total_instances} instances"
            )

    # Flush last document
    yield from process_document()

    logger.info(
        f"Second pass complete: {total_sentences} sentences scanned, "
        f"{total_matches} pattern matches, {total_instances} instances output"
    )


# ============================================================================
# 7. OUTPUT
# ============================================================================

def write_results(
    instances: Iterator[EuphemismInstance],
    output_path: str,
) -> int:
    """Write instances to JSONL."""
    count = 0
    with open(output_path, "w") as f:
        for instance in instances:
            f.write(json.dumps(asdict(instance)) + "\n")
            count += 1
            if count % 10000 == 0:
                logger.info(f"Written {count} instances")
    logger.info(f"Output: {count} instances -> {output_path}")
    return count


# ============================================================================
# 8. PIPELINE RUNNER
# ============================================================================

def run_second_pass(
    sources: list[Iterator[TextRecord]],
    config: SecondPassConfig = SecondPassConfig(),
):
    """
    Full second-pass pipeline.

    Args:
        sources: List of TextRecord iterators (same sources as first pass,
                 or a subset for targeted collection)
        config:  Second pass configuration
    """
    import itertools

    # 1. Load first-pass candidates
    logger.info(f"Loading candidates from {config.first_pass_results_path}...")
    candidates = load_first_pass_candidates(config.first_pass_results_path, config)

    if not candidates:
        logger.error("No candidates loaded. Check first pass output path and filters.")
        return 0

    # 2. Expand variants if requested
    if config.match_mode == "lemma":
        logger.info("Expanding morphological variants (lemma mode)...")
        candidates = expand_variants_lemma(candidates)
    elif config.match_mode == "stem":
        logger.info("Computing stem forms...")
        candidates = expand_variants_stem(candidates)

    # 3. Stream and match
    logger.info(f"Starting second pass (mode={config.match_mode})...")
    combined_stream = itertools.chain(*sources)

    instances = second_pass(combined_stream, candidates, config)

    # 4. Write output
    count = write_results(instances, config.output_path)

    # 5. Save config
    config_path = config.output_path.replace(".jsonl", "_config.json")
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)
    logger.info(f"Config saved to {config_path}")

    return count


# ============================================================================
# 9. EXPERIMENT RUNNER
# ============================================================================

def run_second_pass_experiments(
    source_factory,
    first_pass_path: str = "first_pass_results.jsonl",
):
    """
    Run second pass with different matching modes for comparison.

    Args:
        source_factory: Callable returning fresh source iterator lists
        first_pass_path: Path to first-pass JSONL output
    """
    experiments = [
        SecondPassConfig(
            first_pass_results_path=first_pass_path,
            match_mode="exact",
            context_sentences=1,
            output_path="second_pass_exact.jsonl",
        ),
        SecondPassConfig(
            first_pass_results_path=first_pass_path,
            match_mode="lemma",
            context_sentences=1,
            output_path="second_pass_lemma.jsonl",
        ),
        SecondPassConfig(
            first_pass_results_path=first_pass_path,
            match_mode="stem",
            context_sentences=1,
            output_path="second_pass_stem.jsonl",
        ),
        # Wider context for learning stage analysis
        SecondPassConfig(
            first_pass_results_path=first_pass_path,
            match_mode="lemma",
            context_sentences=3,
            output_path="second_pass_lemma_ctx3.jsonl",
        ),
    ]

    for i, config in enumerate(experiments):
        logger.info(f"\n{'='*60}")
        logger.info(f"SECOND PASS EXPERIMENT {i+1}/{len(experiments)}: {config.output_path}")
        logger.info(f"  mode={config.match_mode}, context={config.context_sentences}")
        logger.info(f"{'='*60}")

        sources = source_factory()
        run_second_pass(sources, config)


# ============================================================================
# 10. ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    from data_sources import (
        SourceConfig,
        stream_text_file,
        stream_reddit_directory,
        # stream_common_crawl_wet,
        # stream_wikipedia_dump,
    )

    # Use the SAME source config as the first pass for consistency
    source_config = SourceConfig(
        target_languages={"en"},
        lang_confidence_threshold=0.7,
        start_year=2015,
        end_year=2026,
    )

    # ------------------------------------------------------------------
    # SINGLE RUN
    # ------------------------------------------------------------------
    # config = SecondPassConfig(
    #     first_pass_results_path="first_pass_results.jsonl",
    #     match_mode="lemma",
    #     context_sentences=1,
    #     min_first_pass_score=0.60,
    #     output_path="second_pass_results.jsonl",
    # )
    # sources = [stream_text_file("test_corpus.txt", config=source_config)]
    # run_second_pass(sources, config)

    # ------------------------------------------------------------------
    # EXPERIMENTS
    # ------------------------------------------------------------------
    # def make_sources():
    #     return [
    #         stream_reddit_directory("./reddit_dumps/", config=source_config),
    #         stream_text_file("test_corpus.txt", config=source_config),
    #     ]
    # run_second_pass_experiments(make_sources, "first_pass_results.jsonl")

    print("Uncomment a configuration above and run.")