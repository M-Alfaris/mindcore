"""
User Preferences Manager.

Manages amendable user preferences separately from read-only system data.
Provides safe methods for updating preferences through AI agent tools.
"""
from typing import Any, Tuple, Optional, Union

from .schemas import UserPreferences
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PreferencesManager:
    """
    Manage user preferences with safe update controls.

    Ensures only amendable fields can be updated, protecting
    read-only system data from modification.

    Example:
        >>> from mindcore.core import SQLiteManager
        >>> db = SQLiteManager("mindcore.db")
        >>> manager = PreferencesManager(db)
        >>>
        >>> # Get or create preferences
        >>> prefs = manager.get_preferences("user123")
        >>> print(prefs.language)  # "en"
        >>>
        >>> # Update a preference
        >>> success, msg = manager.update_preference("user123", "language", "es")
        >>> print(success, msg)  # True, "Updated language to es"
        >>>
        >>> # Try to update a non-amendable field
        >>> success, msg = manager.update_preference("user123", "user_id", "hacker")
        >>> print(success, msg)  # False, "Field 'user_id' is not amendable"
    """

    def __init__(self, db_manager):
        """
        Initialize preferences manager.

        Args:
            db_manager: Database manager instance (SQLiteManager or AsyncSQLiteManager)
        """
        self.db = db_manager

    def get_preferences(self, user_id: str) -> UserPreferences:
        """
        Get user preferences, creating default if not exists.

        Args:
            user_id: User identifier

        Returns:
            UserPreferences object (existing or newly created)
        """
        return self.db.get_or_create_preferences(user_id)

    def update_preference(
        self,
        user_id: str,
        field: str,
        value: Any
    ) -> Tuple[bool, str]:
        """
        Update a user preference field.

        Only amendable fields can be updated. See UserPreferences.AMENDABLE_FIELDS
        for the list of fields that can be modified.

        Args:
            user_id: User identifier
            field: Field name to update
            value: New value for the field

        Returns:
            Tuple of (success: bool, message: str)
        """
        # Check if field is amendable
        if field not in UserPreferences.AMENDABLE_FIELDS:
            logger.warning(f"Attempted to update non-amendable field: {field}")
            return False, f"Field '{field}' is not amendable. Amendable fields are: {', '.join(UserPreferences.AMENDABLE_FIELDS)}"

        # Get current preferences
        prefs = self.get_preferences(user_id)

        # Validate value based on field type
        validation_error = self._validate_value(field, value)
        if validation_error:
            return False, validation_error

        # Update the field
        if prefs.update(field, value):
            if self.db.save_preferences(prefs):
                logger.info(f"Updated {field} for user {user_id}")
                return True, f"Updated {field} to {value}"
            else:
                return False, "Failed to save preferences to database"
        else:
            return False, f"Failed to update {field}"

    def add_interest(self, user_id: str, interest: str) -> Tuple[bool, str]:
        """
        Add an interest to user's list.

        Args:
            user_id: User identifier
            interest: Interest to add

        Returns:
            Tuple of (success: bool, message: str)
        """
        prefs = self.get_preferences(user_id)

        if interest in prefs.interests:
            return True, f"Interest '{interest}' already exists"

        if prefs.add_to_list("interests", interest):
            if self.db.save_preferences(prefs):
                logger.info(f"Added interest '{interest}' for user {user_id}")
                return True, f"Added interest: {interest}"
        return False, f"Failed to add interest: {interest}"

    def remove_interest(self, user_id: str, interest: str) -> Tuple[bool, str]:
        """
        Remove an interest from user's list.

        Args:
            user_id: User identifier
            interest: Interest to remove

        Returns:
            Tuple of (success: bool, message: str)
        """
        prefs = self.get_preferences(user_id)

        if interest not in prefs.interests:
            return False, f"Interest '{interest}' not found"

        if prefs.remove_from_list("interests", interest):
            if self.db.save_preferences(prefs):
                logger.info(f"Removed interest '{interest}' for user {user_id}")
                return True, f"Removed interest: {interest}"
        return False, f"Failed to remove interest: {interest}"

    def add_goal(self, user_id: str, goal: str) -> Tuple[bool, str]:
        """
        Add a goal to user's list.

        Args:
            user_id: User identifier
            goal: Goal to add

        Returns:
            Tuple of (success: bool, message: str)
        """
        prefs = self.get_preferences(user_id)

        if goal in prefs.goals:
            return True, f"Goal '{goal}' already exists"

        if prefs.add_to_list("goals", goal):
            if self.db.save_preferences(prefs):
                logger.info(f"Added goal '{goal}' for user {user_id}")
                return True, f"Added goal: {goal}"
        return False, f"Failed to add goal: {goal}"

    def remove_goal(self, user_id: str, goal: str) -> Tuple[bool, str]:
        """
        Remove a goal from user's list.

        Args:
            user_id: User identifier
            goal: Goal to remove

        Returns:
            Tuple of (success: bool, message: str)
        """
        prefs = self.get_preferences(user_id)

        if goal not in prefs.goals:
            return False, f"Goal '{goal}' not found"

        if prefs.remove_from_list("goals", goal):
            if self.db.save_preferences(prefs):
                logger.info(f"Removed goal '{goal}' for user {user_id}")
                return True, f"Removed goal: {goal}"
        return False, f"Failed to remove goal: {goal}"

    def set_custom_context(
        self,
        user_id: str,
        key: str,
        value: Any
    ) -> Tuple[bool, str]:
        """
        Set a custom context key-value pair.

        Args:
            user_id: User identifier
            key: Context key
            value: Context value

        Returns:
            Tuple of (success: bool, message: str)
        """
        prefs = self.get_preferences(user_id)

        prefs.custom_context[key] = value
        prefs.update("custom_context", prefs.custom_context)

        if self.db.save_preferences(prefs):
            logger.info(f"Set custom context '{key}' for user {user_id}")
            return True, f"Set {key}: {value}"
        return False, f"Failed to set custom context: {key}"

    def remove_custom_context(self, user_id: str, key: str) -> Tuple[bool, str]:
        """
        Remove a custom context key.

        Args:
            user_id: User identifier
            key: Context key to remove

        Returns:
            Tuple of (success: bool, message: str)
        """
        prefs = self.get_preferences(user_id)

        if key not in prefs.custom_context:
            return False, f"Custom context key '{key}' not found"

        del prefs.custom_context[key]
        prefs.update("custom_context", prefs.custom_context)

        if self.db.save_preferences(prefs):
            logger.info(f"Removed custom context '{key}' for user {user_id}")
            return True, f"Removed custom context: {key}"
        return False, f"Failed to remove custom context: {key}"

    def get_context_string(self, user_id: str) -> str:
        """
        Get formatted preferences string for AI context.

        Args:
            user_id: User identifier

        Returns:
            Formatted string for inclusion in AI context
        """
        prefs = self.get_preferences(user_id)
        return prefs.to_context_string()

    def _validate_value(self, field: str, value: Any) -> Optional[str]:
        """
        Validate a value for a specific field.

        Args:
            field: Field name
            value: Value to validate

        Returns:
            Error message if invalid, None if valid
        """
        # Language validation
        if field == "language":
            if not isinstance(value, str) or len(value) < 2:
                return "Language must be a valid language code (e.g., 'en', 'es', 'fr')"

        # Communication style validation
        if field == "communication_style":
            valid_styles = ["formal", "casual", "technical", "balanced"]
            if value not in valid_styles:
                return f"Communication style must be one of: {', '.join(valid_styles)}"

        # List fields validation
        if field in ["interests", "goals", "notification_topics"]:
            if not isinstance(value, list):
                return f"{field} must be a list"
            if not all(isinstance(item, str) for item in value):
                return f"All items in {field} must be strings"

        # Custom context validation
        if field == "custom_context":
            if not isinstance(value, dict):
                return "custom_context must be a dictionary"

        return None


class AsyncPreferencesManager:
    """
    Async version of PreferencesManager for use with AsyncMindcoreClient.

    Same interface as PreferencesManager but with async methods.
    """

    def __init__(self, db_manager):
        """
        Initialize async preferences manager.

        Args:
            db_manager: Async database manager instance
        """
        self.db = db_manager

    async def get_preferences(self, user_id: str) -> UserPreferences:
        """Get user preferences, creating default if not exists."""
        return await self.db.get_or_create_preferences(user_id)

    async def update_preference(
        self,
        user_id: str,
        field: str,
        value: Any
    ) -> Tuple[bool, str]:
        """Update a user preference field."""
        if field not in UserPreferences.AMENDABLE_FIELDS:
            logger.warning(f"Attempted to update non-amendable field: {field}")
            return False, f"Field '{field}' is not amendable"

        prefs = await self.get_preferences(user_id)

        if prefs.update(field, value):
            if await self.db.save_preferences(prefs):
                logger.info(f"Updated {field} for user {user_id}")
                return True, f"Updated {field} to {value}"
        return False, f"Failed to update {field}"

    async def add_interest(self, user_id: str, interest: str) -> Tuple[bool, str]:
        """Add an interest to user's list."""
        prefs = await self.get_preferences(user_id)

        if interest in prefs.interests:
            return True, f"Interest '{interest}' already exists"

        if prefs.add_to_list("interests", interest):
            if await self.db.save_preferences(prefs):
                return True, f"Added interest: {interest}"
        return False, f"Failed to add interest: {interest}"

    async def remove_interest(self, user_id: str, interest: str) -> Tuple[bool, str]:
        """Remove an interest from user's list."""
        prefs = await self.get_preferences(user_id)

        if interest not in prefs.interests:
            return False, f"Interest '{interest}' not found"

        if prefs.remove_from_list("interests", interest):
            if await self.db.save_preferences(prefs):
                return True, f"Removed interest: {interest}"
        return False, f"Failed to remove interest: {interest}"

    async def get_context_string(self, user_id: str) -> str:
        """Get formatted preferences string for AI context."""
        prefs = await self.get_preferences(user_id)
        return prefs.to_context_string()
