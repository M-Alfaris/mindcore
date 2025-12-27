"""Signal History Persistence for CLST.

This module provides persistent storage for reinforcement signals,
enabling:
- Audit trail of all signals applied to memories
- Trend analysis over time
- Signal source/type analytics
- Rollback capabilities

Signal Storage Design:
- Each signal is stored as a separate row (not just aggregated)
- Signals are linked to memory_id and optionally session_id
- Supports querying signal history by memory, session, or time range

Example:
    from mindcore.v2.clst import SignalStore

    store = SignalStore(connection_string="sqlite:///signals.db")

    # Store a signal
    store.record_signal(
        memory_id="mem_123",
        signal_type="usefulness",
        signal_value=0.8,
        source="user",
        session_id="sess_456",
    )

    # Get signal history
    history = store.get_signal_history(memory_id="mem_123", limit=100)

    # Get aggregate stats
    stats = store.get_signal_stats(memory_id="mem_123")
"""

from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Signal Data Types
# =============================================================================


@dataclass
class StoredSignal:
    """A persisted reinforcement signal."""

    signal_id: str
    memory_id: str
    signal_type: str  # relevance, usefulness, correctness, etc.
    signal_value: float  # Raw value (-1 to 1)
    weighted_value: float  # After source/type weighting
    source: str  # user, llm, automated
    source_weight: float
    type_weight: float
    context_similarity: float

    # Context
    session_id: str | None = None
    query_id: str | None = None
    user_id: str | None = None

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Additional context
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "memory_id": self.memory_id,
            "signal_type": self.signal_type,
            "signal_value": self.signal_value,
            "weighted_value": self.weighted_value,
            "source": self.source,
            "source_weight": self.source_weight,
            "type_weight": self.type_weight,
            "context_similarity": self.context_similarity,
            "session_id": self.session_id,
            "query_id": self.query_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "context": self.context,
        }


@dataclass
class SignalStats:
    """Aggregate statistics for signals on a memory."""

    memory_id: str
    total_signals: int
    positive_signals: int
    negative_signals: int

    # Weighted averages
    avg_signal_value: float
    avg_weighted_value: float

    # By source
    user_signal_count: int
    user_avg_value: float
    llm_signal_count: int
    llm_avg_value: float
    automated_signal_count: int
    automated_avg_value: float

    # By type
    signal_type_counts: dict[str, int]
    signal_type_avgs: dict[str, float]

    # Trend
    recent_trend: float  # Average of last N signals
    trend_direction: str  # "improving", "declining", "stable"

    # Time bounds
    first_signal_at: datetime | None
    last_signal_at: datetime | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "total_signals": self.total_signals,
            "positive_signals": self.positive_signals,
            "negative_signals": self.negative_signals,
            "avg_signal_value": self.avg_signal_value,
            "avg_weighted_value": self.avg_weighted_value,
            "user_signal_count": self.user_signal_count,
            "user_avg_value": self.user_avg_value,
            "llm_signal_count": self.llm_signal_count,
            "llm_avg_value": self.llm_avg_value,
            "automated_signal_count": self.automated_signal_count,
            "automated_avg_value": self.automated_avg_value,
            "signal_type_counts": self.signal_type_counts,
            "signal_type_avgs": self.signal_type_avgs,
            "recent_trend": self.recent_trend,
            "trend_direction": self.trend_direction,
        }


# =============================================================================
# Signal Store
# =============================================================================


class SignalStore:
    """Persistent storage for reinforcement signals.

    Stores individual signals for:
    - Audit trail
    - Trend analysis
    - Source/type analytics
    - Debugging and rollback
    """

    def __init__(
        self,
        db_path: str = "signals.db",
        retention_days: int = 90,
    ):
        """Initialize signal store.

        Args:
            db_path: SQLite database path
            retention_days: Days to retain signals (older signals are purged)
        """
        self._db_path = db_path
        self._retention_days = retention_days
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        with self._conn:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    signal_id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    signal_value REAL NOT NULL,
                    weighted_value REAL NOT NULL,
                    source TEXT NOT NULL,
                    source_weight REAL NOT NULL,
                    type_weight REAL NOT NULL,
                    context_similarity REAL DEFAULT 1.0,
                    session_id TEXT,
                    query_id TEXT,
                    user_id TEXT,
                    created_at TEXT NOT NULL,
                    context TEXT DEFAULT '{}'
                )
            """)

            # Indexes for common queries
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_memory
                ON signals(memory_id, created_at DESC)
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_session
                ON signals(session_id, created_at DESC)
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_created
                ON signals(created_at DESC)
            """)

    def record_signal(
        self,
        memory_id: str,
        signal_type: str,
        signal_value: float,
        source: str,
        source_weight: float = 1.0,
        type_weight: float = 1.0,
        context_similarity: float = 1.0,
        session_id: str | None = None,
        query_id: str | None = None,
        user_id: str | None = None,
        context: dict | None = None,
    ) -> StoredSignal:
        """Record a reinforcement signal.

        Args:
            memory_id: Memory being reinforced
            signal_type: Type of signal (relevance, usefulness, etc.)
            signal_value: Raw signal value (-1 to 1)
            source: Signal source (user, llm, automated)
            source_weight: Weight applied for source
            type_weight: Weight applied for type
            context_similarity: Context similarity factor
            session_id: Optional session context
            query_id: Optional query context
            user_id: Optional user context
            context: Additional context dict

        Returns:
            The stored signal
        """
        import json

        signal_id = f"sig_{uuid.uuid4().hex[:12]}"
        weighted_value = signal_value * source_weight * type_weight * context_similarity
        created_at = datetime.now(timezone.utc)

        signal = StoredSignal(
            signal_id=signal_id,
            memory_id=memory_id,
            signal_type=signal_type,
            signal_value=signal_value,
            weighted_value=weighted_value,
            source=source,
            source_weight=source_weight,
            type_weight=type_weight,
            context_similarity=context_similarity,
            session_id=session_id,
            query_id=query_id,
            user_id=user_id,
            created_at=created_at,
            context=context or {},
        )

        with self._conn:
            self._conn.execute(
                """
                INSERT INTO signals (
                    signal_id, memory_id, signal_type, signal_value, weighted_value,
                    source, source_weight, type_weight, context_similarity,
                    session_id, query_id, user_id, created_at, context
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signal.signal_id,
                    signal.memory_id,
                    signal.signal_type,
                    signal.signal_value,
                    signal.weighted_value,
                    signal.source,
                    signal.source_weight,
                    signal.type_weight,
                    signal.context_similarity,
                    signal.session_id,
                    signal.query_id,
                    signal.user_id,
                    signal.created_at.isoformat(),
                    json.dumps(signal.context),
                ),
            )

        return signal

    def get_signal_history(
        self,
        memory_id: str,
        limit: int = 100,
        since: datetime | None = None,
    ) -> list[StoredSignal]:
        """Get signal history for a memory.

        Args:
            memory_id: Memory to query
            limit: Maximum signals to return
            since: Only return signals after this time

        Returns:
            List of signals, newest first
        """
        import json

        query = "SELECT * FROM signals WHERE memory_id = ?"
        params: list[Any] = [memory_id]

        if since:
            query += " AND created_at > ?"
            params.append(since.isoformat())

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor = self._conn.execute(query, params)
        signals = []

        for row in cursor:
            signals.append(
                StoredSignal(
                    signal_id=row["signal_id"],
                    memory_id=row["memory_id"],
                    signal_type=row["signal_type"],
                    signal_value=row["signal_value"],
                    weighted_value=row["weighted_value"],
                    source=row["source"],
                    source_weight=row["source_weight"],
                    type_weight=row["type_weight"],
                    context_similarity=row["context_similarity"],
                    session_id=row["session_id"],
                    query_id=row["query_id"],
                    user_id=row["user_id"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    context=json.loads(row["context"]) if row["context"] else {},
                )
            )

        return signals

    def get_session_signals(
        self,
        session_id: str,
        limit: int = 1000,
    ) -> list[StoredSignal]:
        """Get all signals for a session.

        Args:
            session_id: Session to query
            limit: Maximum signals

        Returns:
            List of signals for the session
        """
        import json

        cursor = self._conn.execute(
            """
            SELECT * FROM signals
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (session_id, limit),
        )

        signals = []
        for row in cursor:
            signals.append(
                StoredSignal(
                    signal_id=row["signal_id"],
                    memory_id=row["memory_id"],
                    signal_type=row["signal_type"],
                    signal_value=row["signal_value"],
                    weighted_value=row["weighted_value"],
                    source=row["source"],
                    source_weight=row["source_weight"],
                    type_weight=row["type_weight"],
                    context_similarity=row["context_similarity"],
                    session_id=row["session_id"],
                    query_id=row["query_id"],
                    user_id=row["user_id"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    context=json.loads(row["context"]) if row["context"] else {},
                )
            )

        return signals

    def get_signal_stats(self, memory_id: str) -> SignalStats:
        """Get aggregate statistics for a memory's signals.

        Args:
            memory_id: Memory to analyze

        Returns:
            SignalStats with aggregate information
        """
        # Basic aggregates
        row = self._conn.execute(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN signal_value > 0 THEN 1 ELSE 0 END) as positive,
                SUM(CASE WHEN signal_value < 0 THEN 1 ELSE 0 END) as negative,
                AVG(signal_value) as avg_value,
                AVG(weighted_value) as avg_weighted,
                MIN(created_at) as first_at,
                MAX(created_at) as last_at
            FROM signals
            WHERE memory_id = ?
            """,
            (memory_id,),
        ).fetchone()

        total = row["total"] or 0
        if total == 0:
            return SignalStats(
                memory_id=memory_id,
                total_signals=0,
                positive_signals=0,
                negative_signals=0,
                avg_signal_value=0.0,
                avg_weighted_value=0.0,
                user_signal_count=0,
                user_avg_value=0.0,
                llm_signal_count=0,
                llm_avg_value=0.0,
                automated_signal_count=0,
                automated_avg_value=0.0,
                signal_type_counts={},
                signal_type_avgs={},
                recent_trend=0.0,
                trend_direction="stable",
                first_signal_at=None,
                last_signal_at=None,
            )

        # By source
        source_stats = {}
        for src in ["user", "llm", "automated"]:
            src_row = self._conn.execute(
                """
                SELECT COUNT(*) as cnt, AVG(signal_value) as avg_val
                FROM signals
                WHERE memory_id = ? AND source LIKE ?
                """,
                (memory_id, f"{src}%"),
            ).fetchone()
            source_stats[src] = {
                "count": src_row["cnt"] or 0,
                "avg": src_row["avg_val"] or 0.0,
            }

        # By type
        type_cursor = self._conn.execute(
            """
            SELECT signal_type, COUNT(*) as cnt, AVG(signal_value) as avg_val
            FROM signals
            WHERE memory_id = ?
            GROUP BY signal_type
            """,
            (memory_id,),
        )
        type_counts = {}
        type_avgs = {}
        for type_row in type_cursor:
            type_counts[type_row["signal_type"]] = type_row["cnt"]
            type_avgs[type_row["signal_type"]] = type_row["avg_val"]

        # Recent trend (last 10 signals)
        recent_rows = self._conn.execute(
            """
            SELECT signal_value FROM signals
            WHERE memory_id = ?
            ORDER BY created_at DESC
            LIMIT 10
            """,
            (memory_id,),
        ).fetchall()

        if recent_rows:
            recent_values = [r["signal_value"] for r in recent_rows]
            recent_trend = sum(recent_values) / len(recent_values)

            # Determine trend direction
            if len(recent_values) >= 5:
                first_half = sum(recent_values[len(recent_values) // 2 :]) / (
                    len(recent_values) // 2
                )
                second_half = sum(recent_values[: len(recent_values) // 2]) / (
                    len(recent_values) // 2
                )
                if second_half - first_half > 0.1:
                    trend_direction = "improving"
                elif first_half - second_half > 0.1:
                    trend_direction = "declining"
                else:
                    trend_direction = "stable"
            else:
                trend_direction = "stable"
        else:
            recent_trend = 0.0
            trend_direction = "stable"

        return SignalStats(
            memory_id=memory_id,
            total_signals=total,
            positive_signals=row["positive"] or 0,
            negative_signals=row["negative"] or 0,
            avg_signal_value=row["avg_value"] or 0.0,
            avg_weighted_value=row["avg_weighted"] or 0.0,
            user_signal_count=source_stats["user"]["count"],
            user_avg_value=source_stats["user"]["avg"],
            llm_signal_count=source_stats["llm"]["count"],
            llm_avg_value=source_stats["llm"]["avg"],
            automated_signal_count=source_stats["automated"]["count"],
            automated_avg_value=source_stats["automated"]["avg"],
            signal_type_counts=type_counts,
            signal_type_avgs=type_avgs,
            recent_trend=recent_trend,
            trend_direction=trend_direction,
            first_signal_at=datetime.fromisoformat(row["first_at"])
            if row["first_at"]
            else None,
            last_signal_at=datetime.fromisoformat(row["last_at"])
            if row["last_at"]
            else None,
        )

    def purge_old_signals(self) -> int:
        """Purge signals older than retention period.

        Returns:
            Number of signals deleted
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=self._retention_days)

        with self._conn:
            cursor = self._conn.execute(
                "DELETE FROM signals WHERE created_at < ?",
                (cutoff.isoformat(),),
            )
            return cursor.rowcount

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
