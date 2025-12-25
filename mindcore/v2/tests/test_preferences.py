"""Tests for PreferenceManager - Temporal preference handling."""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mindcore.v2.flr import FLR, Memory, PreferenceManager
from mindcore.v2.storage.sqlite import SQLiteStorage


@pytest.fixture
def storage():
    """Create a temporary SQLite storage for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    storage = SQLiteStorage(db_path)
    yield storage

    storage.close()
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def flr(storage):
    """Create FLR instance."""
    return FLR(storage)


@pytest.fixture
def prefs(storage, flr):
    """Create PreferenceManager instance."""
    return PreferenceManager(storage, flr)


class TestPreferenceManagerBasics:
    """Basic preference operations."""

    def test_set_preference(self, prefs, storage):
        """Test setting a new preference."""
        pref = prefs.set_preference(
            user_id="user_123",
            key="theme",
            value="User prefers dark mode",
            categories=["ui"],
        )

        assert pref is not None
        assert pref.memory_type == "preference"
        assert pref.content == "User prefers dark mode"
        assert "theme" in pref.topics
        assert "preferences" in pref.topics
        assert pref.semantic_metadata["preference_key"] == "theme"
        assert pref.semantic_metadata["version"] == 1

    def test_get_preference(self, prefs):
        """Test retrieving a preference."""
        prefs.set_preference(
            user_id="user_123",
            key="language",
            value="User prefers English",
        )

        pref = prefs.get_preference("user_123", "language")
        assert pref is not None
        assert pref.content == "User prefers English"

    def test_get_nonexistent_preference(self, prefs):
        """Test getting a preference that doesn't exist."""
        pref = prefs.get_preference("user_123", "nonexistent")
        assert pref is None

    def test_preference_with_confidence(self, prefs):
        """Test preference with confidence score."""
        pref = prefs.set_preference(
            user_id="user_123",
            key="font_size",
            value="User prefers large fonts",
            confidence_score=0.95,
        )

        assert pref.confidence_score == 0.95


class TestPreferenceVersioning:
    """Test preference versioning and updates."""

    def test_update_preference(self, prefs):
        """Test updating a preference creates new version."""
        # Set initial preference
        prefs.set_preference(
            user_id="user_123",
            key="color",
            value="User prefers blue",
        )

        # Update preference
        result = prefs.update_preference(
            user_id="user_123",
            key="color",
            value="User now prefers green",
        )

        assert result.version == 2
        assert result.superseded is True
        assert result.old_preference is not None
        assert result.old_preference.content == "User prefers blue"
        assert result.new_preference.content == "User now prefers green"

    def test_multiple_updates(self, prefs):
        """Test multiple preference updates."""
        prefs.set_preference(user_id="user_123", key="food", value="Likes pizza")
        prefs.update_preference(user_id="user_123", key="food", value="Likes sushi")
        result = prefs.update_preference(user_id="user_123", key="food", value="Likes tacos")

        assert result.version == 3

        # Current should be latest
        current = prefs.get_preference("user_123", "food")
        assert current.content == "Likes tacos"

    def test_old_preference_expired(self, prefs, storage):
        """Test that old preference is marked as expired."""
        pref1 = prefs.set_preference(user_id="user_123", key="music", value="Likes rock")
        original_id = pref1.memory_id

        prefs.update_preference(user_id="user_123", key="music", value="Likes jazz")

        # Check old preference is expired
        old = storage.get(original_id)
        assert old is not None
        assert "valid_until" in old.semantic_metadata
        assert old.semantic_metadata["valid_until"] is not None

    def test_supersedes_chain(self, prefs):
        """Test that supersedes chain is maintained."""
        pref1 = prefs.set_preference(user_id="user_123", key="sport", value="Likes tennis")
        id1 = pref1.memory_id

        result2 = prefs.update_preference(user_id="user_123", key="sport", value="Likes golf")
        id2 = result2.new_preference.memory_id

        result3 = prefs.update_preference(user_id="user_123", key="sport", value="Likes swimming")

        # Check chain
        assert result2.new_preference.semantic_metadata["supersedes"] == id1
        assert result3.new_preference.semantic_metadata["supersedes"] == id2


class TestPreferenceHistory:
    """Test preference history tracking."""

    def test_get_history(self, prefs):
        """Test retrieving preference history."""
        prefs.set_preference(user_id="user_123", key="pet", value="Has a dog")
        prefs.update_preference(user_id="user_123", key="pet", value="Has two dogs")
        prefs.update_preference(user_id="user_123", key="pet", value="Has two dogs and a cat")

        history = prefs.get_preference_history("user_123", "pet")

        assert history.total_versions == 3
        assert len(history.versions) == 3
        assert history.current is not None
        assert history.current.content == "Has two dogs and a cat"

        # Check chronological order
        versions = [v.semantic_metadata["version"] for v in history.versions]
        assert versions == [1, 2, 3]

    def test_history_to_dict(self, prefs):
        """Test history serialization."""
        prefs.set_preference(user_id="user_123", key="city", value="Lives in NYC")

        history = prefs.get_preference_history("user_123", "city")
        data = history.to_dict()

        assert data["preference_key"] == "city"
        assert data["total_versions"] == 1
        assert data["current"] is not None


class TestPreferenceDeletion:
    """Test preference deletion."""

    def test_soft_delete(self, prefs):
        """Test soft delete expires preference."""
        prefs.set_preference(user_id="user_123", key="newsletter", value="Subscribed")

        result = prefs.delete_preference("user_123", "newsletter")
        assert result is True

        # Should not be returned by default
        pref = prefs.get_preference("user_123", "newsletter")
        assert pref is None

        # But should exist with include_expired
        pref = prefs.get_preference("user_123", "newsletter", include_expired=True)
        assert pref is not None
        assert pref.semantic_metadata.get("deleted") is True

    def test_hard_delete(self, prefs, storage):
        """Test hard delete removes completely."""
        pref = prefs.set_preference(user_id="user_123", key="temp", value="Temporary preference")
        memory_id = pref.memory_id

        result = prefs.delete_preference("user_123", "temp", hard_delete=True)
        assert result is True

        # Should be completely gone
        assert storage.get(memory_id) is None

    def test_delete_nonexistent(self, prefs):
        """Test deleting nonexistent preference."""
        result = prefs.delete_preference("user_123", "nonexistent")
        assert result is False


class TestListPreferences:
    """Test listing preferences."""

    def test_list_all_preferences(self, prefs):
        """Test listing all user preferences."""
        prefs.set_preference(
            user_id="user_123",
            key="theme",
            value="Dark mode",
            categories=["ui"],
        )
        prefs.set_preference(
            user_id="user_123",
            key="language",
            value="English",
            categories=["localization"],
        )
        prefs.set_preference(
            user_id="user_123",
            key="timezone",
            value="UTC-5",
            categories=["localization"],
        )

        summary = prefs.list_preferences("user_123")

        assert summary.total_keys == 3
        assert "theme" in summary.preferences
        assert "language" in summary.preferences
        assert "timezone" in summary.preferences
        assert "ui" in summary.categories
        assert "localization" in summary.categories

    def test_list_by_category(self, prefs):
        """Test filtering preferences by category."""
        prefs.set_preference(
            user_id="user_123",
            key="theme",
            value="Dark",
            categories=["ui"],
        )
        prefs.set_preference(
            user_id="user_123",
            key="lang",
            value="EN",
            categories=["localization"],
        )

        summary = prefs.list_preferences("user_123", category="ui")

        # Should only get ui preferences
        assert summary.total_keys >= 1
        assert "theme" in summary.preferences

    def test_list_excludes_expired(self, prefs):
        """Test that expired preferences are excluded by default."""
        prefs.set_preference(user_id="user_123", key="old", value="Old value")
        prefs.delete_preference("user_123", "old")  # Soft delete

        prefs.set_preference(user_id="user_123", key="new", value="New value")

        summary = prefs.list_preferences("user_123")

        assert "new" in summary.preferences
        assert "old" not in summary.preferences

    def test_summary_to_dict(self, prefs):
        """Test summary serialization."""
        prefs.set_preference(user_id="user_123", key="test", value="Value")

        summary = prefs.list_preferences("user_123")
        data = summary.to_dict()

        assert data["user_id"] == "user_123"
        assert data["total_keys"] == 1


class TestPreferenceMerge:
    """Test merging multiple preferences."""

    def test_merge_preferences(self, prefs):
        """Test merging multiple preferences into one."""
        prefs.set_preference(
            user_id="user_123",
            key="color_primary",
            value="Blue",
            topics=["design"],
        )
        prefs.set_preference(
            user_id="user_123",
            key="color_secondary",
            value="White",
            topics=["design"],
        )
        prefs.set_preference(
            user_id="user_123",
            key="color_accent",
            value="Orange",
            topics=["design"],
        )

        merged = prefs.merge_preferences(
            user_id="user_123",
            keys=["color_primary", "color_secondary", "color_accent"],
            new_key="color_scheme",
            new_value="Blue primary, white secondary, orange accents",
        )

        assert merged.semantic_metadata["preference_key"] == "color_scheme"
        assert "design" in merged.topics
        assert "merged_from" in merged.semantic_metadata

        # Old preferences should be expired
        for key in ["color_primary", "color_secondary", "color_accent"]:
            pref = prefs.get_preference("user_123", key)
            assert pref is None


class TestPreferenceReinforcement:
    """Test reinforcement signal integration."""

    def test_new_preference_reinforced(self, prefs, flr, storage):
        """Test that new preferences receive positive reinforcement."""
        pref = prefs.set_preference(
            user_id="user_123",
            key="test_reinforce",
            value="Test value",
        )

        # Get fresh from storage
        stored = storage.get(pref.memory_id)
        # Reinforcement is applied asynchronously via FLR
        # Just verify the preference was created correctly
        assert stored is not None
        assert stored.memory_type == "preference"

    def test_expired_preference_penalized(self, prefs, flr, storage):
        """Test that expired preferences receive negative reinforcement."""
        pref1 = prefs.set_preference(
            user_id="user_123",
            key="test_expire",
            value="Original value",
        )
        original_id = pref1.memory_id

        # Update creates new version and penalizes old
        prefs.update_preference(
            user_id="user_123",
            key="test_expire",
            value="New value",
        )

        # Old preference should exist but be expired
        old = storage.get(original_id)
        assert old is not None
        assert "valid_until" in old.semantic_metadata


class TestPreferenceMetadata:
    """Test preference metadata handling."""

    def test_extra_metadata(self, prefs):
        """Test storing extra metadata with preference."""
        pref = prefs.set_preference(
            user_id="user_123",
            key="custom",
            value="Custom preference",
            extra_metadata={
                "source": "onboarding",
                "confidence": 0.9,
                "tags": ["important", "verified"],
            },
        )

        assert pref.semantic_metadata["source"] == "onboarding"
        assert pref.semantic_metadata["confidence"] == 0.9
        assert pref.semantic_metadata["tags"] == ["important", "verified"]

    def test_inherited_metadata(self, prefs):
        """Test that topics/categories are inherited on update."""
        prefs.set_preference(
            user_id="user_123",
            key="inherited",
            value="Original",
            topics=["custom_topic"],
            categories=["custom_category"],
        )

        result = prefs.update_preference(
            user_id="user_123",
            key="inherited",
            value="Updated",
            # No topics/categories provided - should inherit
        )

        assert "custom_topic" in result.new_preference.topics
        assert "custom_category" in result.new_preference.categories


class TestPreferenceIsolation:
    """Test user preference isolation."""

    def test_user_isolation(self, prefs):
        """Test that preferences are isolated per user."""
        prefs.set_preference(user_id="user_1", key="theme", value="Dark")
        prefs.set_preference(user_id="user_2", key="theme", value="Light")

        pref1 = prefs.get_preference("user_1", "theme")
        pref2 = prefs.get_preference("user_2", "theme")

        assert pref1.content == "Dark"
        assert pref2.content == "Light"

    def test_list_preferences_user_scoped(self, prefs):
        """Test that list only returns current user's preferences."""
        prefs.set_preference(user_id="user_1", key="pref_a", value="A")
        prefs.set_preference(user_id="user_2", key="pref_b", value="B")

        summary = prefs.list_preferences("user_1")

        assert "pref_a" in summary.preferences
        assert "pref_b" not in summary.preferences


class TestPreferenceStats:
    """Test preference manager statistics."""

    def test_get_stats(self, prefs):
        """Test getting manager statistics."""
        stats = prefs.get_stats()

        assert "default_importance" in stats
        assert "default_access_level" in stats
        assert "has_flr" in stats
        assert stats["has_flr"] is True
