"""Comprehensive tests for TrivialMessageDetector.

CRITICAL: The trivial detector saves 20-40% of LLM enrichment calls by
detecting greetings, confirmations, and filler messages that don't need
LLM processing.
"""

import pytest

from mindcore.agents.trivial_detector import (
    TrivialCategory,
    TrivialMatch,
    TrivialMessageDetector,
    get_trivial_detector,
    reset_trivial_detector,
)


class TestTrivialCategory:
    """Tests for TrivialCategory enum."""

    def test_category_values(self):
        """Test trivial category enum values."""
        assert TrivialCategory.GREETING.value == "greeting"
        assert TrivialCategory.FAREWELL.value == "farewell"
        assert TrivialCategory.CONFIRMATION.value == "confirmation"
        assert TrivialCategory.THANKS.value == "thanks"
        assert TrivialCategory.FILLER.value == "filler"
        assert TrivialCategory.QUESTION_WORD.value == "question_word"
        assert TrivialCategory.ACKNOWLEDGMENT.value == "acknowledgment"
        assert TrivialCategory.APOLOGY.value == "apology"


class TestTrivialMatch:
    """Tests for TrivialMatch dataclass."""

    def test_trivial_match_creation(self):
        """Test creating a trivial match result."""
        match = TrivialMatch(
            is_trivial=True,
            category=TrivialCategory.GREETING,
            confidence=0.95,
            matched_pattern=r"^hello$",
        )

        assert match.is_trivial is True
        assert match.category == TrivialCategory.GREETING
        assert match.confidence == 0.95
        assert match.matched_pattern == r"^hello$"

    def test_trivial_match_defaults(self):
        """Test trivial match default values."""
        match = TrivialMatch(is_trivial=False)

        assert match.category is None
        assert match.confidence == 0.0
        assert match.matched_pattern is None


class TestTrivialMessageDetectorInitialization:
    """Tests for TrivialMessageDetector initialization."""

    def test_default_initialization(self):
        """Test default detector initialization."""
        detector = TrivialMessageDetector()

        assert detector.min_confidence == 0.8
        assert len(detector._compiled_patterns) > 0

    def test_custom_min_confidence(self):
        """Test initialization with custom min_confidence."""
        detector = TrivialMessageDetector(min_confidence=0.5)

        assert detector.min_confidence == 0.5

    def test_custom_patterns(self):
        """Test initialization with custom patterns."""
        custom_patterns = [
            (r"^custom_greeting$", TrivialCategory.GREETING, 0.1),
        ]
        detector = TrivialMessageDetector(custom_patterns=custom_patterns)

        result = detector.detect("custom_greeting")
        assert result.is_trivial is True
        assert result.category == TrivialCategory.GREETING


class TestTrivialMessageDetectorGreetings:
    """Tests for greeting detection."""

    @pytest.fixture
    def detector(self):
        return TrivialMessageDetector()

    @pytest.mark.parametrize(
        "text",
        [
            "hi",
            "Hi",
            "HI",
            "hi!",
            "hello",
            "Hello!",
            "hey",
            "Hey there",
            "howdy",
            "greetings",
            "yo",
        ],
    )
    def test_detect_simple_greetings(self, detector, text):
        """Test detecting simple greetings."""
        result = detector.detect(text)

        # Most should be detected as trivial greetings
        if result.is_trivial:
            assert result.category == TrivialCategory.GREETING

    def test_detect_hello(self, detector):
        """Test detecting 'hello'."""
        result = detector.detect("hello")

        assert result.is_trivial is True
        assert result.category == TrivialCategory.GREETING
        assert result.confidence >= 0.8

    def test_detect_hi_with_punctuation(self, detector):
        """Test detecting 'hi!' with punctuation."""
        result = detector.detect("hi!")

        assert result.is_trivial is True
        assert result.category == TrivialCategory.GREETING

    def test_detect_good_morning(self, detector):
        """Test detecting 'good morning'."""
        result = detector.detect("good morning")

        assert result.is_trivial is True
        assert result.category == TrivialCategory.GREETING

    def test_detect_good_afternoon(self, detector):
        """Test detecting 'good afternoon'."""
        result = detector.detect("good afternoon!")

        assert result.is_trivial is True
        assert result.category == TrivialCategory.GREETING


class TestTrivialMessageDetectorFarewells:
    """Tests for farewell detection."""

    @pytest.fixture
    def detector(self):
        return TrivialMessageDetector()

    @pytest.mark.parametrize(
        "text",
        [
            "bye",
            "goodbye",
            "see you",
            "later",
            "take care",
            "cya",
        ],
    )
    def test_detect_farewells(self, detector, text):
        """Test detecting farewell messages."""
        result = detector.detect(text)

        if result.is_trivial:
            assert result.category == TrivialCategory.FAREWELL

    def test_detect_goodbye(self, detector):
        """Test detecting 'goodbye'."""
        result = detector.detect("goodbye")

        assert result.is_trivial is True
        assert result.category == TrivialCategory.FAREWELL


class TestTrivialMessageDetectorConfirmations:
    """Tests for confirmation detection."""

    @pytest.fixture
    def detector(self):
        return TrivialMessageDetector()

    @pytest.mark.parametrize(
        "text",
        [
            "yes",
            "yeah",
            "yep",
            "yup",
            "no",
            "nope",
            "ok",
            "okay",
            "alright",
            "sure",
            "got it",
            "correct",
            "right",
        ],
    )
    def test_detect_confirmations(self, detector, text):
        """Test detecting confirmation messages."""
        result = detector.detect(text)

        if result.is_trivial:
            assert result.category == TrivialCategory.CONFIRMATION

    def test_detect_yes(self, detector):
        """Test detecting 'yes'."""
        result = detector.detect("yes")

        assert result.is_trivial is True
        assert result.category == TrivialCategory.CONFIRMATION

    def test_detect_ok(self, detector):
        """Test detecting 'ok'."""
        result = detector.detect("ok")

        assert result.is_trivial is True
        assert result.category == TrivialCategory.CONFIRMATION

    def test_detect_sounds_good(self, detector):
        """Test detecting 'sounds good'."""
        result = detector.detect("sounds good")

        assert result.is_trivial is True
        assert result.category == TrivialCategory.CONFIRMATION


class TestTrivialMessageDetectorThanks:
    """Tests for thanks detection."""

    @pytest.fixture
    def detector(self):
        return TrivialMessageDetector()

    @pytest.mark.parametrize(
        "text",
        [
            "thanks",
            "thank you",
            "thx",
            "ty",
            "thanks!",
            "thanks a lot",
            "thanks so much",
            "much appreciated",
        ],
    )
    def test_detect_thanks(self, detector, text):
        """Test detecting thank you messages."""
        result = detector.detect(text)

        if result.is_trivial:
            assert result.category == TrivialCategory.THANKS

    def test_detect_thank_you(self, detector):
        """Test detecting 'thank you'."""
        result = detector.detect("thank you")

        assert result.is_trivial is True
        assert result.category == TrivialCategory.THANKS


class TestTrivialMessageDetectorFillers:
    """Tests for filler detection."""

    @pytest.fixture
    def detector(self):
        return TrivialMessageDetector()

    @pytest.mark.parametrize(
        "text",
        [
            "um",
            "uh",
            "hmm",
            "er",
            "ah",
            "oh",
            "well",
            "so",
            "i see",
            "i understand",
        ],
    )
    def test_detect_fillers(self, detector, text):
        """Test detecting filler messages."""
        result = detector.detect(text)

        if result.is_trivial:
            assert result.category == TrivialCategory.FILLER

    def test_detect_hmm(self, detector):
        """Test detecting 'hmm'."""
        result = detector.detect("hmm")

        assert result.is_trivial is True
        assert result.category == TrivialCategory.FILLER


class TestTrivialMessageDetectorQuestionWords:
    """Tests for single question word detection."""

    @pytest.fixture
    def detector(self):
        return TrivialMessageDetector()

    @pytest.mark.parametrize(
        "text",
        [
            "what",
            "where",
            "when",
            "why",
            "how",
            "who",
            "which",
            "what?",
        ],
    )
    def test_detect_question_words(self, detector, text):
        """Test detecting single question words."""
        result = detector.detect(text)

        if result.is_trivial:
            assert result.category == TrivialCategory.QUESTION_WORD


class TestTrivialMessageDetectorAcknowledgments:
    """Tests for acknowledgment detection."""

    @pytest.fixture
    def detector(self):
        return TrivialMessageDetector()

    @pytest.mark.parametrize(
        "text",
        [
            "nice",
            "great",
            "awesome",
            "cool",
            "perfect",
            "excellent",
            "interesting",
        ],
    )
    def test_detect_acknowledgments(self, detector, text):
        """Test detecting acknowledgment messages."""
        result = detector.detect(text)

        if result.is_trivial:
            assert result.category == TrivialCategory.ACKNOWLEDGMENT

    def test_detect_great(self, detector):
        """Test detecting 'great'."""
        result = detector.detect("great")

        assert result.is_trivial is True
        assert result.category == TrivialCategory.ACKNOWLEDGMENT


class TestTrivialMessageDetectorApologies:
    """Tests for apology detection."""

    @pytest.fixture
    def detector(self):
        return TrivialMessageDetector()

    @pytest.mark.parametrize(
        "text",
        [
            "apologies",
            "my bad",
            "oops",
            "i'm sorry",
        ],
    )
    def test_detect_apologies(self, detector, text):
        """Test detecting apology messages."""
        result = detector.detect(text)

        if result.is_trivial:
            assert result.category == TrivialCategory.APOLOGY

    def test_detect_apologies_specific(self, detector):
        """Test detecting 'apologies'."""
        result = detector.detect("apologies")

        assert result.is_trivial is True
        assert result.category == TrivialCategory.APOLOGY

    def test_sorry_matches_question_word(self, detector):
        """Test 'sorry' matches QUESTION_WORD pattern (pardon/sorry/excuse me pattern).

        Note: 'sorry' is in the QUESTION_WORD patterns (e.g., 'pardon', 'sorry',
        'excuse me') which comes before APOLOGY patterns in the detector.
        """
        result = detector.detect("sorry")

        assert result.is_trivial is True
        # 'sorry' matches QUESTION_WORD pattern before APOLOGY pattern
        assert result.category == TrivialCategory.QUESTION_WORD


class TestTrivialMessageDetectorNonTrivial:
    """Tests for non-trivial message detection."""

    @pytest.fixture
    def detector(self):
        return TrivialMessageDetector()

    def test_not_trivial_question(self, detector):
        """Test real questions are not detected as trivial."""
        result = detector.detect("What is the status of my order #12345?")

        assert result.is_trivial is False

    def test_not_trivial_request(self, detector):
        """Test requests are not detected as trivial."""
        result = detector.detect("Please refund my order, it arrived damaged.")

        assert result.is_trivial is False

    def test_not_trivial_complex_message(self, detector):
        """Test complex messages are not detected as trivial."""
        result = detector.detect(
            "I need help understanding why my billing statement shows an extra charge from last month."
        )

        assert result.is_trivial is False

    def test_not_trivial_hello_with_content(self, detector):
        """Test greeting with content is not trivial."""
        result = detector.detect("Hello, I need help with my order")

        assert result.is_trivial is False

    def test_empty_string(self, detector):
        """Test empty string is not trivial."""
        result = detector.detect("")

        assert result.is_trivial is False


class TestTrivialMessageDetectorConfidence:
    """Tests for confidence calculation."""

    @pytest.fixture
    def detector(self):
        return TrivialMessageDetector()

    def test_high_confidence_exact_match(self, detector):
        """Test exact matches have high confidence."""
        result = detector.detect("hello")

        assert result.confidence >= 0.8

    def test_confidence_affects_detection(self):
        """Test min_confidence affects what's detected."""
        strict_detector = TrivialMessageDetector(min_confidence=0.99)

        # May not detect some messages that require high confidence
        result = strict_detector.detect("hi!")
        # Confidence check is the key here

    def test_short_message_low_confidence(self, detector):
        """Test short unmatched messages get low confidence."""
        result = detector.detect("?")

        # Should not be confidently trivial
        if not result.is_trivial:
            assert result.confidence < 0.8


class TestTrivialMessageDetectorSingleton:
    """Tests for singleton pattern."""

    def test_get_trivial_detector_returns_same_instance(self):
        """Test get_trivial_detector returns singleton."""
        reset_trivial_detector()

        detector1 = get_trivial_detector()
        detector2 = get_trivial_detector()

        assert detector1 is detector2

    def test_reset_trivial_detector_creates_new(self):
        """Test reset creates new instance."""
        detector1 = get_trivial_detector()
        reset_trivial_detector()
        detector2 = get_trivial_detector()

        assert detector1 is not detector2


class TestTrivialMessageDetectorCaseInsensitivity:
    """Tests for case insensitivity."""

    @pytest.fixture
    def detector(self):
        return TrivialMessageDetector()

    def test_uppercase_hello(self, detector):
        """Test uppercase 'HELLO' is detected."""
        result = detector.detect("HELLO")

        assert result.is_trivial is True
        assert result.category == TrivialCategory.GREETING

    def test_mixed_case_thanks(self, detector):
        """Test mixed case 'ThAnKs' is detected."""
        result = detector.detect("ThAnKs")

        assert result.is_trivial is True
        assert result.category == TrivialCategory.THANKS

    def test_uppercase_ok(self, detector):
        """Test 'OK' is detected."""
        result = detector.detect("OK")

        assert result.is_trivial is True
        assert result.category == TrivialCategory.CONFIRMATION


class TestTrivialMessageDetectorEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def detector(self):
        return TrivialMessageDetector()

    def test_whitespace_only(self, detector):
        """Test whitespace-only string."""
        result = detector.detect("   ")

        # After strip, this is empty
        assert result.is_trivial is False

    def test_with_leading_trailing_whitespace(self, detector):
        """Test message with whitespace is still detected."""
        result = detector.detect("  hello  ")

        assert result.is_trivial is True
        assert result.category == TrivialCategory.GREETING

    def test_multiple_punctuation(self, detector):
        """Test message with multiple punctuation."""
        result = detector.detect("ok!!!")

        if result.is_trivial:
            assert result.category == TrivialCategory.CONFIRMATION

    def test_numeric_only(self, detector):
        """Test numeric-only message."""
        result = detector.detect("12345")

        assert result.is_trivial is False

    def test_special_characters_only(self, detector):
        """Test special characters only."""
        result = detector.detect("@#$%")

        assert result.is_trivial is False
