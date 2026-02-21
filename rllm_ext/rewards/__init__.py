"""rLLM extensions for reward functions."""

from rllm_ext.rewards.prediction_similarity import (
    SimilarityConfig,
    compute_bleu_style_score,
    compute_ngram_overlap,
    compute_prediction_similarity_reward,
)

__all__ = [
    "SimilarityConfig",
    "compute_ngram_overlap",
    "compute_bleu_style_score",
    "compute_prediction_similarity_reward",
]
