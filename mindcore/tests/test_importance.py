"""
Tests for importance scoring algorithms.

Tests all importance algorithms to ensure correct scoring behavior.
"""

import pytest
from mindcore.importance import (
    ImportanceAlgorithm,
    LLMBasedImportance,
    KeywordImportance,
    LengthBasedImportance,
    SentimentBasedImportance,
    CompositeImportance,
    get_importance_algorithm
)


class TestImportanceAlgorithmBase:
    """Test base ImportanceAlgorithm class."""

    def test_abstract_class_cannot_instantiate(self):
        """ImportanceAlgorithm is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            ImportanceAlgorithm()

    def test_custom_algorithm_implementation(self):
        """Custom algorithm can be created by subclassing."""

        class CustomImportance(ImportanceAlgorithm):
            def calculate(self, text, metadata=None, **kwargs):
                return 0.75

        algorithm = CustomImportance()
        score = algorithm.calculate("test")
        assert score == 0.75


class TestLLMBasedImportance:
    """Test LLM-based importance scoring."""

    def test_extract_from_metadata(self):
        """Test extracting importance from metadata."""
        algorithm = LLMBasedImportance()

        metadata = {"importance": 0.8}
        score = algorithm.calculate("test text", metadata=metadata)

        assert score == 0.8

    def test_default_when_no_metadata(self):
        """Test default score when no metadata."""
        algorithm = LLMBasedImportance()

        score = algorithm.calculate("test text", metadata=None)
        assert score == 0.5

    def test_default_when_no_importance_in_metadata(self):
        """Test default score when importance not in metadata."""
        algorithm = LLMBasedImportance()

        metadata = {"topics": ["test"], "sentiment": "positive"}
        score = algorithm.calculate("test text", metadata=metadata)

        assert score == 0.5

    def test_various_importance_values(self):
        """Test various importance values from metadata."""
        algorithm = LLMBasedImportance()

        test_cases = [
            (0.0, 0.0),
            (0.25, 0.25),
            (0.5, 0.5),
            (0.75, 0.75),
            (1.0, 1.0)
        ]

        for metadata_value, expected in test_cases:
            metadata = {"importance": metadata_value}
            score = algorithm.calculate("text", metadata=metadata)
            assert score == expected


class TestKeywordImportance:
    """Test keyword-based importance scoring."""

    def test_default_keywords(self):
        """Test with default keywords."""
        algorithm = KeywordImportance()

        # High importance keywords
        score = algorithm.calculate("This is urgent and critical")
        assert score > 0.5

        # Low importance keywords
        score = algorithm.calculate("Maybe we can discuss this casually")
        assert score < 0.5

        # Neutral text
        score = algorithm.calculate("Hello, how are you?")
        assert score == 0.5

    def test_custom_keywords(self):
        """Test with custom keywords."""
        algorithm = KeywordImportance(
            high_importance_keywords=["important", "priority"],
            low_importance_keywords=["optional", "later"]
        )

        # High importance
        score_high = algorithm.calculate("This is important priority task")
        assert score_high > 0.5

        # Low importance
        score_low = algorithm.calculate("This is optional, can do later")
        assert score_low < 0.5

    def test_multiple_high_keywords(self):
        """Test multiple high importance keywords."""
        algorithm = KeywordImportance()

        score = algorithm.calculate("urgent critical important deadline asap")
        assert score > 0.8

    def test_multiple_low_keywords(self):
        """Test multiple low importance keywords."""
        algorithm = KeywordImportance()

        score = algorithm.calculate("maybe perhaps casual fyi just wondering")
        assert score < 0.3

    def test_case_insensitive(self):
        """Test keyword matching is case insensitive."""
        algorithm = KeywordImportance()

        score1 = algorithm.calculate("URGENT")
        score2 = algorithm.calculate("urgent")
        score3 = algorithm.calculate("Urgent")

        assert score1 == score2 == score3
        assert score1 > 0.5

    def test_score_bounds(self):
        """Test score is always between 0 and 1."""
        algorithm = KeywordImportance()

        # Extreme case with many keywords
        text_high = " ".join(["urgent critical important"] * 20)
        score_high = algorithm.calculate(text_high)
        assert 0.0 <= score_high <= 1.0

        text_low = " ".join(["maybe casual fyi"] * 20)
        score_low = algorithm.calculate(text_low)
        assert 0.0 <= score_low <= 1.0


class TestLengthBasedImportance:
    """Test length-based importance scoring."""

    def test_default_parameters(self):
        """Test with default min and optimal lengths."""
        algorithm = LengthBasedImportance()

        # Very short message
        score = algorithm.calculate("Hi")
        assert score == 0.2

        # Short message
        score = algorithm.calculate("Hello, how are you?")
        assert 0.2 < score < 0.8

        # Medium message (around optimal 200 chars)
        message = "This is a medium-length message. " * 6  # ~200 chars
        score = algorithm.calculate(message)
        assert 0.6 < score <= 0.8

        # Long message
        message = "This is a very long message. " * 20  # ~600 chars
        score = algorithm.calculate(message)
        assert 0.5 <= score < 0.8

    def test_custom_parameters(self):
        """Test with custom min and optimal lengths."""
        algorithm = LengthBasedImportance(min_length=20, optimal_length=100)

        # Below min
        score = algorithm.calculate("Short")
        assert score == 0.2

        # At optimal
        message = "x" * 100
        score = algorithm.calculate(message)
        assert 0.6 < score <= 0.8

    def test_very_short_messages(self):
        """Test very short messages (greetings)."""
        algorithm = LengthBasedImportance()

        short_messages = ["Hi", "Hello", "OK", "Yes", "No"]

        for msg in short_messages:
            score = algorithm.calculate(msg)
            assert score == 0.2

    def test_increasing_length_increases_score(self):
        """Test that score generally increases with length (up to optimal)."""
        algorithm = LengthBasedImportance(min_length=10, optimal_length=200)

        # Test messages of increasing length
        lengths = [10, 50, 100, 150, 200]
        scores = []

        for length in lengths:
            message = "x" * length
            score = algorithm.calculate(message)
            scores.append(score)

        # Scores should generally increase
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1]


class TestSentimentBasedImportance:
    """Test sentiment-based importance scoring."""

    def test_no_metadata(self):
        """Test default score when no metadata."""
        algorithm = SentimentBasedImportance()

        score = algorithm.calculate("test text", metadata=None)
        assert score == 0.5

    def test_no_sentiment_in_metadata(self):
        """Test default score when no sentiment in metadata."""
        algorithm = SentimentBasedImportance()

        metadata = {"topics": ["test"], "importance": 0.7}
        score = algorithm.calculate("test", metadata=metadata)

        assert score == 0.5

    def test_neutral_sentiment(self):
        """Test neutral sentiment gives lower importance."""
        algorithm = SentimentBasedImportance()

        metadata = {
            "sentiment": {
                "overall": "neutral",
                "score": 0.5
            }
        }

        score = algorithm.calculate("test", metadata=metadata)
        assert score == 0.4

    def test_positive_sentiment(self):
        """Test positive sentiment gives higher importance."""
        algorithm = SentimentBasedImportance()

        metadata = {
            "sentiment": {
                "overall": "positive",
                "score": 0.9  # Very positive
            }
        }

        score = algorithm.calculate("test", metadata=metadata)
        assert score > 0.5

    def test_negative_sentiment(self):
        """Test negative sentiment gives higher importance."""
        algorithm = SentimentBasedImportance()

        metadata = {
            "sentiment": {
                "overall": "negative",
                "score": 0.1  # Very negative
            }
        }

        score = algorithm.calculate("test", metadata=metadata)
        assert score > 0.5

    def test_sentiment_intensity(self):
        """Test that stronger sentiment gives higher importance."""
        algorithm = SentimentBasedImportance()

        # Mild sentiment
        metadata_mild = {
            "sentiment": {
                "overall": "positive",
                "score": 0.6
            }
        }
        score_mild = algorithm.calculate("test", metadata=metadata_mild)

        # Strong sentiment
        metadata_strong = {
            "sentiment": {
                "overall": "positive",
                "score": 1.0
            }
        }
        score_strong = algorithm.calculate("test", metadata=metadata_strong)

        assert score_strong > score_mild


class TestCompositeImportance:
    """Test composite importance scoring."""

    def test_default_algorithms(self):
        """Test with default algorithm weights."""
        algorithm = CompositeImportance()

        # Should combine LLM (50%), Keyword (30%), Length (20%)
        metadata = {"importance": 0.8}
        text = "urgent critical important message with good length"

        score = algorithm.calculate(text, metadata=metadata)

        # Score should be weighted average
        assert 0.0 <= score <= 1.0

    def test_custom_algorithms(self):
        """Test with custom algorithms and weights."""
        algorithm = CompositeImportance(algorithms=[
            (KeywordImportance(), 0.5),
            (LengthBasedImportance(), 0.5)
        ])

        text = "urgent critical important"
        score = algorithm.calculate(text, metadata={})

        assert 0.0 <= score <= 1.0

    def test_single_algorithm_fails(self):
        """Test that composite handles failed algorithms gracefully."""

        class FailingAlgorithm(ImportanceAlgorithm):
            def calculate(self, text, metadata=None, **kwargs):
                raise Exception("Algorithm failed")

        algorithm = CompositeImportance(algorithms=[
            (FailingAlgorithm(), 0.5),
            (KeywordImportance(), 0.5)
        ])

        # Should not crash, should use only successful algorithm
        score = algorithm.calculate("urgent text", metadata={})
        assert 0.0 <= score <= 1.0

    def test_all_algorithms_fail(self):
        """Test default score when all algorithms fail."""

        class FailingAlgorithm(ImportanceAlgorithm):
            def calculate(self, text, metadata=None, **kwargs):
                raise Exception("Failed")

        algorithm = CompositeImportance(algorithms=[
            (FailingAlgorithm(), 0.5),
            (FailingAlgorithm(), 0.5)
        ])

        score = algorithm.calculate("test", metadata={})
        assert score == 0.5  # Default

    def test_weighted_average(self):
        """Test weighted average calculation."""
        # Create predictable algorithms
        class FixedAlgorithm(ImportanceAlgorithm):
            def __init__(self, fixed_score):
                self.fixed_score = fixed_score

            def calculate(self, text, metadata=None, **kwargs):
                return self.fixed_score

        algorithm = CompositeImportance(algorithms=[
            (FixedAlgorithm(0.8), 0.5),  # 50% weight, score 0.8
            (FixedAlgorithm(0.4), 0.5)   # 50% weight, score 0.4
        ])

        score = algorithm.calculate("test", metadata={})

        # Expected: (0.8 * 0.5 + 0.4 * 0.5) / (0.5 + 0.5) = 0.6
        assert score == pytest.approx(0.6, abs=0.01)


class TestGetImportanceAlgorithm:
    """Test get_importance_algorithm factory function."""

    def test_get_llm_algorithm(self):
        """Test getting LLM algorithm."""
        algorithm = get_importance_algorithm("llm")
        assert isinstance(algorithm, LLMBasedImportance)

    def test_get_keyword_algorithm(self):
        """Test getting keyword algorithm."""
        algorithm = get_importance_algorithm("keyword")
        assert isinstance(algorithm, KeywordImportance)

    def test_get_length_algorithm(self):
        """Test getting length algorithm."""
        algorithm = get_importance_algorithm("length")
        assert isinstance(algorithm, LengthBasedImportance)

    def test_get_sentiment_algorithm(self):
        """Test getting sentiment algorithm."""
        algorithm = get_importance_algorithm("sentiment")
        assert isinstance(algorithm, SentimentBasedImportance)

    def test_get_composite_algorithm(self):
        """Test getting composite algorithm."""
        algorithm = get_importance_algorithm("composite")
        assert isinstance(algorithm, CompositeImportance)

    def test_invalid_algorithm_name(self):
        """Test invalid algorithm name raises error."""
        with pytest.raises(ValueError, match="Unknown algorithm"):
            get_importance_algorithm("invalid")

    def test_default_algorithm(self):
        """Test default algorithm is LLM."""
        algorithm = get_importance_algorithm()
        assert isinstance(algorithm, LLMBasedImportance)


class TestIntegration:
    """Integration tests for importance algorithms."""

    def test_all_algorithms_produce_valid_scores(self):
        """Test all algorithms produce scores in [0, 1] range."""
        algorithms = [
            LLMBasedImportance(),
            KeywordImportance(),
            LengthBasedImportance(),
            SentimentBasedImportance(),
            CompositeImportance()
        ]

        test_texts = [
            "Hi",
            "This is urgent and critical",
            "Maybe we can discuss this later casually",
            "This is a medium-length message with some content",
            "x" * 1000  # Very long
        ]

        metadata = {
            "importance": 0.7,
            "sentiment": {
                "overall": "positive",
                "score": 0.8
            }
        }

        for algorithm in algorithms:
            for text in test_texts:
                score = algorithm.calculate(text, metadata=metadata)
                assert 0.0 <= score <= 1.0, (
                    f"{algorithm.__class__.__name__} produced invalid score "
                    f"{score} for text: {text[:50]}"
                )

    def test_realistic_scenario(self):
        """Test realistic message scoring scenario."""
        messages = [
            ("Hi", 0.2),  # Greeting - low
            ("urgent: critical bug in production", 0.8),  # Urgent - high
            ("maybe we can chat later", 0.3),  # Casual - low
            ("This is a detailed explanation of the architecture...", 0.6)  # Medium
        ]

        algorithm = KeywordImportance()

        for text, expected_range in messages:
            score = algorithm.calculate(text)
            # Allow some tolerance
            if expected_range < 0.5:
                assert score < 0.6, f"Expected low score for: {text}"
            elif expected_range > 0.7:
                assert score > 0.6, f"Expected high score for: {text}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
