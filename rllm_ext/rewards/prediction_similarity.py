"""N-gram similarity reward for tool output predictions.

This module provides lightweight n-gram similarity computation (BLEU-style)
for rewarding accurate predictions of tool outputs.
"""

from collections import Counter
from dataclasses import dataclass

import math


@dataclass(frozen=True)
class SimilarityConfig:
    """Configuration for n-gram similarity computation.

    Attributes:
        enabled: Whether to compute similarity reward.
        weight: Maximum reward value (multiplied with similarity score in [0,1]).
        n: Maximum n-gram size (BLEU-4 uses n=4, computing 1-4 gram precisions).
        min_length: Minimum text length (in words) to compute similarity.
            Skips very short outputs like "4", "True".
        smoothing: Add +1 smoothing to avoid zero counts.
    """

    enabled: bool = True
    weight: float = 0.1
    n: int = 4
    min_length: int = 4
    smoothing: bool = True


def _get_ngrams(tokens: list[str], n: int) -> list[str]:
    """Extract n-grams from token list.

    Args:
        tokens: List of tokens (words).
        n: N-gram size.

    Returns:
        List of n-gram strings.
    """
    if len(tokens) < n:
        # For short texts, return the full text as a single n-gram
        return [" ".join(tokens)] if tokens else []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def compute_ngram_overlap(text1: str, text2: str, n: int = 4, smoothing: bool = True) -> float:
    """Compute n-gram overlap precision (BLEU-style).

    Args:
        text1: Prediction text.
        text2: Reference text (actual tool output).
        n: N-gram size.
        smoothing: Add +1 smoothing to avoid zero counts.

    Returns:
        Float in [0, 1] where 1.0 = perfect match, 0.0 = no overlap.
    """
    if not text1 or not text2:
        return 0.0

    # Tokenize (simple whitespace split - works well for code/math)
    tokens1 = text1.split()
    tokens2 = text2.split()

    if len(tokens1) < n or len(tokens2) < n:
        # For short texts, use word-level unigram
        n = 1

    # Extract n-grams
    ngrams1 = Counter(_get_ngrams(tokens1, n))
    ngrams2 = Counter(_get_ngrams(tokens2, n))

    # Compute overlap (clipped count - standard BLEU)
    overlap = sum(min(ngrams1[ngram], ngrams2[ngram]) for ngram in ngrams1)
    total = sum(ngrams1.values())

    if total == 0:
        return 0.0

    if smoothing:
        # Add 1 to numerator and denominator for smoothing
        return (overlap + 1) / (total + 1)

    return overlap / total


def compute_bleu_style_score(prediction: str, reference: str, config: SimilarityConfig) -> float:
    """Compute BLEU-style geometric mean of n-gram precisions.

    Args:
        prediction: Predicted text.
        reference: Reference (actual) text.
        config: Similarity configuration.

    Returns:
        Float in [0, 1] representing BLEU-style similarity score.
    """
    if not prediction or not reference:
        return 0.0

    pred_len = len(prediction.split())
    ref_len = len(reference.split())

    # Brevity penalty (punish short predictions)
    if pred_len < ref_len:
        bp = math.exp(1 - ref_len / pred_len) if pred_len > 0 else 0.0
    else:
        bp = 1.0

    # Compute n-gram precisions for n=1,2,3,4 (or up to config.n)
    precisions = []
    for n in range(1, min(config.n + 1, 5)):
        prec = compute_ngram_overlap(prediction, reference, n, config.smoothing)
        if prec > 0:
            precisions.append(prec)
        else:
            precisions.append(1e-10)  # Avoid log(0)

    # Geometric mean
    if precisions:
        log_prec = sum(math.log(p) for p in precisions) / len(precisions)
        score = bp * math.exp(log_prec)
    else:
        score = 0.0

    return min(score, 1.0)


def compute_prediction_similarity_reward(
    prediction: str, actual_output: str, config: SimilarityConfig
) -> float:
    """Compute reward based on prediction-actual similarity.

    Args:
        prediction: Predicted tool output.
        actual_output: Actual tool output.
        config: Similarity configuration.

    Returns:
        Float: Reward value in [0, config.weight].
    """
    if not config.enabled:
        return 0.0

    if not prediction or not actual_output:
        return 0.0

    # Skip very short outputs (e.g., "4", "True", "Success")
    if len(actual_output.split()) < config.min_length:
        return 0.0

    # Compute BLEU-style similarity
    similarity = compute_bleu_style_score(prediction, actual_output, config)

    return similarity * config.weight


__all__ = [
    "SimilarityConfig",
    "compute_ngram_overlap",
    "compute_bleu_style_score",
    "compute_prediction_similarity_reward",
]
