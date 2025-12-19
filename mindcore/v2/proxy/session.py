"""Session management for Mindcore Proxy.

Tracks Claude Code sessions and their state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class SessionState(str, Enum):
    """State of a proxy session."""

    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Session:
    """Represents a Claude Code session being proxied.

    Attributes:
        session_id: Unique identifier for the session
        user_id: User who owns this session
        state: Current session state
        messages: List of messages in this session
        metadata: Additional session metadata
        created_at: When the session was created
        updated_at: When the session was last updated
    """

    session_id: str
    user_id: str
    state: SessionState = SessionState.PENDING
    messages: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = None

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to the session.

        Args:
            role: Message role (user, assistant, system)
            content: Message content
            **kwargs: Additional message metadata
        """
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs,
        })
        self.updated_at = datetime.now(timezone.utc)

    def activate(self) -> None:
        """Mark session as active."""
        self.state = SessionState.ACTIVE
        self.updated_at = datetime.now(timezone.utc)

    def complete(self) -> None:
        """Mark session as completed."""
        self.state = SessionState.COMPLETED
        self.updated_at = datetime.now(timezone.utc)

    def fail(self, reason: str | None = None) -> None:
        """Mark session as failed.

        Args:
            reason: Optional failure reason
        """
        self.state = SessionState.FAILED
        if reason:
            self.metadata["failure_reason"] = reason
        self.updated_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "state": self.state.value,
            "messages": self.messages,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class SessionManager:
    """Manages multiple proxy sessions.

    Provides session lifecycle management, lookup, and cleanup.
    """

    def __init__(self) -> None:
        """Initialize session manager."""
        self._sessions: dict[str, Session] = {}

    def create(self, session_id: str, user_id: str, **kwargs: Any) -> Session:
        """Create a new session.

        Args:
            session_id: Unique session identifier
            user_id: User who owns this session
            **kwargs: Additional session parameters

        Returns:
            The created Session

        Raises:
            ValueError: If session_id already exists
        """
        if session_id in self._sessions:
            raise ValueError(f"Session {session_id} already exists")

        session = Session(
            session_id=session_id,
            user_id=user_id,
            metadata=kwargs.get("metadata", {}),
        )
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        """Get a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session if found, None otherwise
        """
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: str, user_id: str, **kwargs: Any) -> Session:
        """Get existing session or create new one.

        Args:
            session_id: Session identifier
            user_id: User ID (used if creating)
            **kwargs: Additional parameters (used if creating)

        Returns:
            Existing or newly created Session
        """
        session = self.get(session_id)
        if session is None:
            session = self.create(session_id, user_id, **kwargs)
        return session

    def list_sessions(
        self,
        user_id: str | None = None,
        state: SessionState | None = None,
    ) -> list[Session]:
        """List sessions with optional filters.

        Args:
            user_id: Filter by user
            state: Filter by state

        Returns:
            List of matching sessions
        """
        sessions = list(self._sessions.values())

        if user_id is not None:
            sessions = [s for s in sessions if s.user_id == user_id]

        if state is not None:
            sessions = [s for s in sessions if s.state == state]

        return sessions

    def delete(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: Session to delete

        Returns:
            True if deleted, False if not found
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def cleanup_completed(self, max_age_seconds: int = 3600) -> int:
        """Remove old completed sessions.

        Args:
            max_age_seconds: Maximum age for completed sessions

        Returns:
            Number of sessions removed
        """
        now = datetime.now(timezone.utc)
        to_remove = []

        for session_id, session in self._sessions.items():
            if session.state in (SessionState.COMPLETED, SessionState.FAILED):
                age = (now - session.created_at).total_seconds()
                if age > max_age_seconds:
                    to_remove.append(session_id)

        for session_id in to_remove:
            del self._sessions[session_id]

        return len(to_remove)
