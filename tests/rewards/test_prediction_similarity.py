"""Tests for prediction similarity reward computation."""

import math
import pytest

from rllm_ext.rewards.prediction_similarity import (
    SimilarityConfig,
    compute_ngram_overlap,
    compute_bleu_style_score,
    compute_prediction_similarity_reward,
)


class TestSimilarityConfig:
    """Test SimilarityConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SimilarityConfig()
        assert config.enabled is True
        assert config.weight == 0.1
        assert config.n == 4
        assert config.min_length == 4
        assert config.smoothing is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = SimilarityConfig(
            enabled=False,
            weight=0.5,
            n=2,
            min_length=10,
            smoothing=False,
        )
        assert config.enabled is False
        assert config.weight == 0.5
        assert config.n == 2
        assert config.min_length == 10
        assert config.smoothing is False

    def test_from_dict(self):
        """Test creating config from dict."""
        config_dict = {
            "enabled": True,
            "weight": 0.2,
            "n": 3,
            "min_length": 5,
            "smoothing": True,
        }
        config = SimilarityConfig(**config_dict)
        assert config.enabled is True
        assert config.weight == 0.2
        assert config.n == 3
        assert config.min_length == 5
        assert config.smoothing is True

    def test_frozen_immutable(self):
        """Test that config is frozen (immutable)."""
        config = SimilarityConfig()
        with pytest.raises(TypeError):  # frozen dataclass raises TypeError
            config.enabled = False


class TestComputeNgramOverlap:
    """Test n-gram overlap computation."""

    def test_perfect_match_unigram(self):
        """Test perfect match with unigrams (n=1)."""
        text1 = "the result is 42"
        text2 = "the result is 42"
        overlap = compute_ngram_overlap(text1, text2, n=1, smoothing=False)
        assert overlap == pytest.approx(1.0, abs=0.01)

    def test_perfect_match_bigram(self):
        """Test perfect match with bigrams (n=2)."""
        text1 = "the result is 42"
        text2 = "the result is 42"
        overlap = compute_ngram_overlap(text1, text2, n=2, smoothing=False)
        assert overlap == pytest.approx(1.0, abs=0.01)

    def test_perfect_match_4gram(self):
        """Test perfect match with 4-grams."""
        text1 = "the result of calculation is 42"
        text2 = "the result of calculation is 42"
        overlap = compute_ngram_overlap(text1, text2, n=4, smoothing=False)
        assert overlap == pytest.approx(1.0, abs=0.01)

    def test_no_overlap(self):
        """Test completely different texts."""
        text1 = "python calculator result"
        text2 = "weather forecast tomorrow"
        overlap = compute_ngram_overlap(text1, text2, n=2, smoothing=False)
        assert overlap == 0.0

    def test_partial_overlap(self):
        """Test partial n-gram overlap."""
        text1 = "the result of calculation is 42"
        text2 = "the result of computation is 42"
        overlap = compute_ngram_overlap(text1, text2, n=2, smoothing=False)
        # Should have some overlap (the result, is 42)
        assert 0 < overlap < 1.0

    def test_short_text_fallback_to_unigram(self):
        """Test that short texts fall back to unigram."""
        text1 = "hello"
        text2 = "hello world"
        # Even with n=4, short texts should use unigram
        overlap = compute_ngram_overlap(text1, text2, n=4, smoothing=False)
        assert overlap > 0

    def test_empty_text(self):
        """Test handling of empty texts."""
        assert compute_ngram_overlap("", "text", n=1) == 0.0
        assert compute_ngram_overlap("text", "", n=1) == 0.0
        assert compute_ngram_overlap("", "", n=1) == 0.0
        assert compute_ngram_overlap(None, "text", n=1) == 0.0
        assert compute_ngram_overlap("text", None, n=1) == 0.0

    def test_smoothing(self):
        """Test +1 smoothing for zero counts."""
        text1 = "completely different words here"
        text2 = "other words entirely different"
        # Without smoothing
        overlap_no_smooth = compute_ngram_overlap(text1, text2, n=1, smoothing=False)
        # With smoothing should be higher
        overlap_smooth = compute_ngram_overlap(text1, text2, n=1, smoothing=True)
        assert overlap_smooth > overlap_no_smooth

    def test_code_like_text(self):
        """Test with code-like text (whitespace tokenization)."""
        text1 = "x = 5\ny = 10\nresult = x + y"
        text2 = "x = 5\ny = 10\nresult = 15"
        overlap = compute_ngram_overlap(text1, text2, n=1, smoothing=False)
        # Should have overlap for "x = 5", "y = 10", "result"
        assert overlap > 0

    def test_case_sensitivity(self):
        """Test that tokenization is case-sensitive."""
        text1 = "The Result Is 42"
        text2 = "the result is 42"
        overlap = compute_ngram_overlap(text1, text2, n=1, smoothing=False)
        # Different case = different tokens
        assert overlap < 1.0

    def test_single_character_tokens(self):
        """Test with single character tokens."""
        text1 = "a b c d"
        text2 = "a b c e"
        overlap = compute_ngram_overlap(text1, text2, n=1, smoothing=False)
        # 3 out of 4 match
        assert overlap == pytest.approx(0.75, abs=0.01)


class TestComputeBleuStyleScore:
    """Test BLEU-style scoring."""

    def test_exact_match(self):
        """Test BLEU score with exact match."""
        config = SimilarityConfig(n=4, smoothing=False)
        score = compute_bleu_style_score(
            "the cat sat on the mat",
            "the cat sat on the mat",
            config,
        )
        assert score == pytest.approx(1.0, abs=0.01)

    def test_no_match(self):
        """Test BLEU score with no overlap."""
        config = SimilarityConfig(n=4, smoothing=False)
        score = compute_bleu_style_score(
            "python code execution",
            "weather forecast today",
            config,
        )
        assert score == 0.0

    def test_partial_match(self):
        """Test BLEU score with partial overlap."""
        config = SimilarityConfig(n=4, smoothing=False)
        score = compute_bleu_style_score(
            "the result of calculation is 42",
            "the result of computation is 42",
            config,
        )
        assert 0 < score < 1.0

    def test_brevity_penalty_short_prediction(self):
        """Test brevity penalty for short predictions."""
        config = SimilarityConfig(n=4, smoothing=False)
        # Shorter prediction than reference
        score = compute_bleu_style_score(
            "the cat",  # Short
            "the cat sat on the mat",  # Long
            config,
        )
        # Should be penalized
        assert score < 1.0

    def test_no_brevity_penalty_longer_prediction(self):
        """Test no brevity penalty for longer predictions."""
        config = SimilarityConfig(n=4, smoothing=False)
        # Longer prediction than reference
        score = compute_bleu_style_score(
            "the cat sat on the mat",  # Long
            "the cat",  # Short
            config,
        )
        # Should have penalty = 1.0 (no penalty)
        assert score > 0.5

    def test_empty_text(self):
        """Test BLEU score with empty texts."""
        config = SimilarityConfig(n=4)
        assert compute_bleu_style_score("", "text", config) == 0.0
        assert compute_bleu_style_score("text", "", config) == 0.0
        assert compute_bleu_style_score("", "", config) == 0.0
        assert compute_bleu_style_score(None, "text", config) == 0.0

    def test_different_n_values(self):
        """Test BLEU score with different n values."""
        text1 = "the cat sat on the mat"
        text2 = "the cat sat on the mat"

        for n in [1, 2, 3, 4]:
            config = SimilarityConfig(n=n, smoothing=False)
            score = compute_bleu_style_score(text1, text2, config)
            assert score == pytest.approx(1.0, abs=0.01)

    def test_geometric_mean(self):
        """Test that BLEU uses geometric mean of n-gram precisions."""
        config = SimilarityConfig(n=4, smoothing=True)
        # This text will have different unigram, bigram, trigram, 4-gram scores
        # Geometric mean should be between min and max
        score = compute_bleu_style_score(
            "the cat sat on the mat",
            "the dog sat on the mat",
            config,
        )
        assert 0 < score < 1.0

    def test_single_word_texts(self):
        """Test with single word texts."""
        config = SimilarityConfig(n=4, min_length=1)
        score = compute_bleu_style_score("hello", "hello", config)
        # Should handle gracefully
        assert isinstance(score, float)

    def test_smoothing_affects_score(self):
        """Test that smoothing affects the score."""
        text1 = "some words here"
        text2 = "different words there"

        config_no_smooth = SimilarityConfig(n=2, smoothing=False)
        config_smooth = SimilarityConfig(n=2, smoothing=True)

        score_no_smooth = compute_bleu_style_score(text1, text2, config_no_smooth)
        score_smooth = compute_bleu_style_score(text1, text2, config_smooth)

        # Smoothed score should be different (typically higher for low overlaps)
        assert score_smooth != score_no_smooth


class TestComputePredictionSimilarityReward:
    """Test prediction similarity reward computation."""

    def test_perfect_match_max_reward(self):
        """Test perfect match returns max reward."""
        config = SimilarityConfig(enabled=True, weight=0.1, min_length=1)
        reward = compute_prediction_similarity_reward(
            "the result is 42",
            "the result is 42",
            config,
        )
        assert reward == pytest.approx(0.1, abs=0.01)

    def test_no_match_zero_reward(self):
        """Test no overlap returns zero reward."""
        config = SimilarityConfig(enabled=True, weight=0.1, min_length=1)
        reward = compute_prediction_similarity_reward(
            "python calculator",
            "weather forecast",
            config,
        )
        assert reward == 0.0

    def test_partial_match_partial_reward(self):
        """Test partial overlap returns partial reward."""
        config = SimilarityConfig(enabled=True, weight=0.1, min_length=1)
        reward = compute_prediction_similarity_reward(
            "the result is 42",
            "the result is 42.0",
            config,
        )
        # Should have some overlap but not perfect
        assert 0 < reward < 0.1

    def test_disabled_returns_zero(self):
        """Test that disabled config returns zero."""
        config = SimilarityConfig(enabled=False, weight=0.1)
        reward = compute_prediction_similarity_reward(
            "the result is 42",
            "the result is 42",
            config,
        )
        assert reward == 0.0

    def test_short_text_skipped(self):
        """Test that very short texts are skipped."""
        config = SimilarityConfig(enabled=True, weight=0.1, min_length=4)
        # Shorter than min_length
        reward = compute_prediction_similarity_reward(
            "42",
            "42",
            config,
        )
        assert reward == 0.0

    def test_min_length_boundary(self):
        """Test behavior at min_length boundary."""
        config = SimilarityConfig(enabled=True, weight=0.1, min_length=4)

        # Exactly min_length - should be skipped
        reward1 = compute_prediction_similarity_reward(
            "one two three",
            "one two three",
            config,
        )
        assert reward1 == 0.0

        # Just above min_length - should compute
        reward2 = compute_prediction_similarity_reward(
            "one two three four",
            "one two three four",
            config,
        )
        assert reward2 > 0

    def test_empty_text_returns_zero(self):
        """Test handling of empty/None texts."""
        config = SimilarityConfig(enabled=True, weight=0.1)

        assert compute_prediction_similarity_reward("", "text", config) == 0.0
        assert compute_prediction_similarity_reward("text", "", config) == 0.0
        assert compute_prediction_similarity_reward(None, "text", config) == 0.0
        assert compute_prediction_similarity_reward("text", None, config) == 0.0

    def test_weight_scaling(self):
        """Test that weight scales the reward correctly."""
        for weight in [0.05, 0.1, 0.2, 0.5]:
            config = SimilarityConfig(enabled=True, weight=weight, min_length=1)
            reward = compute_prediction_similarity_reward(
                "the result is 42",
                "the result is 42",
                config,
            )
            assert reward == pytest.approx(weight, abs=0.01)

    def test_n_value_affects_reward(self):
        """Test that n value affects the reward."""
        text1 = "the cat sat on the mat"
        text2 = "the dog sat on the mat"

        # Different n values should give different rewards
        rewards = []
        for n in [1, 2, 3, 4]:
            config = SimilarityConfig(enabled=True, weight=0.1, n=n, min_length=1)
            reward = compute_prediction_similarity_reward(text1, text2, config)
            rewards.append(reward)

        # Rewards should vary with n
        assert len(set(rewards)) > 1

    def test_code_output_similarity(self):
        """Test similarity with typical code tool outputs."""
        config = SimilarityConfig(enabled=True, weight=0.1, min_length=4)

        actual = "x = 5\ny = 10\nresult = x + y\nprint(result)"
        prediction = "x = 5\ny = 10\nresult = 15\nprint(result)"

        reward = compute_prediction_similarity_reward(prediction, actual, config)
        # Should have some overlap (x=5, y=10, parts)
        assert reward > 0
        assert reward <= 0.1

    def test_math_output_similarity(self):
        """Test similarity with math tool outputs."""
        config = SimilarityConfig(enabled=True, weight=0.1, min_length=4)

        actual = "Calculating: 2 + 2 = 4\nThe result is 4"
        prediction = "Calculating: 2 + 2 = 4\nThe result is 4.0"

        reward = compute_prediction_similarity_reward(prediction, actual, config)
        # Should have high overlap
        assert reward > 0.05

    def test_smoothing_affects_reward(self):
        """Test that smoothing affects the reward."""
        text1 = "some words"
        text2 = "other words"

        config_no_smooth = SimilarityConfig(
            enabled=True, weight=0.1, smoothing=False, min_length=1
        )
        config_smooth = SimilarityConfig(
            enabled=True, weight=0.1, smoothing=True, min_length=1
        )

        reward_no_smooth = compute_prediction_similarity_reward(
            text1, text2, config_no_smooth
        )
        reward_smooth = compute_prediction_similarity_reward(
            text1, text2, config_smooth
        )

        # Should be different
        assert reward_smooth != reward_no_smooth


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_unicode_text(self):
        """Test with unicode characters."""
        config = SimilarityConfig(enabled=True, weight=0.1, min_length=1)
        reward = compute_prediction_similarity_reward(
            "résultat est 42",
            "résultat est 42",
            config,
        )
        assert reward == pytest.approx(0.1, abs=0.01)

    def test_repeated_words(self):
        """Test with repeated words."""
        config = SimilarityConfig(enabled=True, weight=0.1, min_length=1)
        reward = compute_prediction_similarity_reward(
            "the the the the",
            "the the the the",
            config,
        )
        assert reward == pytest.approx(0.1, abs=0.01)

    def test_punctuation_handling(self):
        """Test handling of punctuation (treated as part of tokens)."""
        config = SimilarityConfig(enabled=True, weight=0.1, min_length=1)
        # Whitespace tokenization treats "word," as different from "word"
        reward = compute_prediction_similarity_reward(
            "the result is 42",
            "the result is 42.",
            config,
        )
        # Should have high but not perfect match due to period
        assert reward > 0.08

    def test_very_long_text(self):
        """Test with very long text."""
        config = SimilarityConfig(enabled=True, weight=0.1, min_length=1)
        long_text = "word " * 100
        reward = compute_prediction_similarity_reward(
            long_text.strip(),
            long_text.strip(),
            config,
        )
        assert reward == pytest.approx(0.1, abs=0.01)

    def test_single_character_difference(self):
        """Test with single character difference."""
        config = SimilarityConfig(enabled=True, weight=0.1, min_length=1)
        reward = compute_prediction_similarity_reward(
            "the result is 42",
            "the result is 43",
            config,
        )
        # Should have high similarity (only one token different)
        assert reward > 0.05

    def test_mixed_order_words(self):
        """Test with words in different order."""
        config = SimilarityConfig(enabled=True, weight=0.1, min_length=1)
        reward = compute_prediction_similarity_reward(
            "the cat sat on mat",
            "mat on sat cat the",
            config,
        )
        # Should have some overlap (unigrams match)
        assert reward > 0
        assert reward < 0.1  # But not perfect due to order

    def test_numbers_vs_words(self):
        """Test with numbers vs word representations."""
        config = SimilarityConfig(enabled=True, weight=0.1, min_length=1)
        reward = compute_prediction_similarity_reward(
            "result is 42",
            "result is forty two",
            config,
        )
        # Should have partial overlap (result, is)
        assert 0 < reward < 0.1


class TestConfigFromDict:
    """Test creating config from dict (common usage pattern)."""

    def test_empty_dict_uses_defaults(self):
        """Test that empty dict uses default values."""
        config = SimilarityConfig(**{})
        assert config.enabled is True
        assert config.weight == 0.1
        assert config.n == 4
        assert config.min_length == 4

    def test_partial_dict_uses_defaults_for_rest(self):
        """Test that partial dict uses defaults for unspecified values."""
        config = SimilarityConfig(**{"weight": 0.5})
        assert config.enabled is True  # default
        assert config.weight == 0.5  # specified
        assert config.n == 4  # default

    def test_all_values_specified(self):
        """Test with all values specified."""
        config = SimilarityConfig(
            **{
                "enabled": False,
                "weight": 0.3,
                "n": 2,
                "min_length": 10,
                "smoothing": False,
            }
        )
        assert config.enabled is False
        assert config.weight == 0.3
        assert config.n == 2
        assert config.min_length == 10
        assert config.smoothing is False
