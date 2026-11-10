"""
Shared sentence segmentation.

A single lightweight sentence splitter used by every pipeline that needs to
break a Reddit record into sentences, so they all segment text the same way.
Consumers:

  * ``hit_matrix_core.build_context`` — sentence-level context construction.
  * ``zhu_temporal/zhu_detection_month.py`` — optional ``--split-sentences``
    mode for the Zhu detection stage.

It depends only on the standard library, so importing it never pulls in
numpy/torch/nltk. The splitter is deliberately simple (no model, no new
dependency): Reddit text is messy, and downstream length rules (min-extend /
max-truncate, or a character cap) handle the fallout from run-ons and missing
punctuation.
"""

__all__ = ["split_sentences", "iter_sentences"]


def split_sentences(text):
    """
    Split ``text`` into sentence spans, returning ``[(start, end), ...]`` char
    offsets into the original string.

    Rule: break after a run of ``.!?`` that is followed by whitespace (or the
    end of the string). Consecutive terminators (``?!``, ``...``) count as one
    break. Leading whitespace after a break is skipped so spans start on the
    first non-space character of the next sentence. A trailing fragment with no
    terminator becomes its own span.

    Offsets index the original ``text`` (not lowercased/normalized), so
    ``text[start:end]`` returns the sentence's surface form.
    """
    spans = []
    start = 0
    n = len(text)
    i = 0
    while i < n:
        if text[i] in ".!?":
            j = i + 1
            while j < n and text[j] in ".!?":
                j += 1
            if j >= n or text[j].isspace():
                spans.append((start, j))
                while j < n and text[j].isspace():
                    j += 1
                start = j
                i = j
                continue
        i += 1
    if start < n:
        spans.append((start, n))
    return spans


def iter_sentences(text):
    """
    Yield each sentence's ``(sentence_text, start, end)`` for ``text``.

    Convenience wrapper over ``split_sentences`` for consumers that want the
    substring plus its offsets rather than offsets alone.
    """
    for start, end in split_sentences(text):
        yield text[start:end], start, end
