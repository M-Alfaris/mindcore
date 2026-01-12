"""Configuration for Mindcore storage and search features.

This module provides configuration dataclasses for search functionality,
including pg_trgm trigram search and ParadeDB BM25 indexing.

Example:
    from mindcore.storage import PostgresStorage, SearchConfig

    # Use default configuration
    storage = PostgresStorage(connection_string)

    # Or customize search settings
    config = SearchConfig(
        use_trigram_search=True,
        trigram_similarity_threshold=0.25,
        use_bm25_search=True,
        ranking_weights={
            "content": 0.2,
            "topic": 0.3,
            "recency": 0.1,
            "reinforcement": 0.2,
            "importance": 0.1,
            "popularity": 0.1,
        },
    )
    storage = PostgresStorage(connection_string, search_config=config)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchConfig:
    """Configuration for PostgreSQL search features.

    Controls which search extensions are used and how ranking is performed.
    Falls back gracefully if extensions are not available.

    Attributes:
        use_trigram_search: Enable pg_trgm for fuzzy similarity matching.
            Requires pg_trgm extension. Default True.
        trigram_similarity_threshold: Minimum similarity score (0-1) for
            trigram matches. Lower = more matches. Default 0.2.
        use_bm25_search: Enable ParadeDB BM25 for full-text search.
            Requires ParadeDB pg_search extension. Default False.
        use_sql_ranking: Use SQL rank_memory() function instead of
            Python-side scoring. Requires ranking_functions.sql. Default True.
        ranking_weights: Weights for multi-component scoring.
            Keys: content, topic, recency, reinforcement, importance, popularity.
            Should roughly sum to 1.0 for normalized output.

    Example:
        # High-recency configuration (favor recent memories)
        config = SearchConfig(
            ranking_weights={
                "content": 0.1,
                "topic": 0.2,
                "recency": 0.35,  # Increased
                "reinforcement": 0.15,
                "importance": 0.1,
                "popularity": 0.1,
            }
        )

        # Topic-focused configuration (favor topic matches)
        config = SearchConfig(
            ranking_weights={
                "content": 0.1,
                "topic": 0.4,  # Increased
                "recency": 0.1,
                "reinforcement": 0.2,
                "importance": 0.1,
                "popularity": 0.1,
            }
        )
    """

    # Extension toggles
    use_trigram_search: bool = True
    trigram_similarity_threshold: float = 0.2
    use_bm25_search: bool = False
    use_sql_ranking: bool = True

    # Ranking weights (must match keys in rank_memory() SQL function)
    ranking_weights: dict[str, float] = field(
        default_factory=lambda: {
            "content": 0.15,
            "topic": 0.25,
            "recency": 0.15,
            "reinforcement": 0.2,
            "importance": 0.15,
            "popularity": 0.1,
        }
    )

    # BM25 specific settings
    bm25_weight: float = 0.35  # Weight of BM25 score in hybrid search
    bm25_fetch_multiplier: int = 2  # Fetch N*limit from BM25 for re-ranking

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0 <= self.trigram_similarity_threshold <= 1:
            raise ValueError(
                f"trigram_similarity_threshold must be between 0 and 1, "
                f"got {self.trigram_similarity_threshold}"
            )

        if not 0 <= self.bm25_weight <= 1:
            raise ValueError(f"bm25_weight must be between 0 and 1, got {self.bm25_weight}")

        # Validate ranking weights keys
        expected_keys = {"content", "topic", "recency", "reinforcement", "importance", "popularity"}
        actual_keys = set(self.ranking_weights.keys())
        if actual_keys != expected_keys:
            missing = expected_keys - actual_keys
            extra = actual_keys - expected_keys
            msg = "Invalid ranking_weights keys."
            if missing:
                msg += f" Missing: {missing}."
            if extra:
                msg += f" Unexpected: {extra}."
            raise ValueError(msg)

        # Validate weight values are reasonable
        for key, value in self.ranking_weights.items():
            if not 0 <= value <= 1:
                raise ValueError(f"ranking_weights['{key}'] must be between 0 and 1, got {value}")

    def to_sql_weights_json(self) -> str:
        """Convert ranking weights to JSON string for SQL function.

        Returns:
            JSON string suitable for rank_memory() p_weights parameter.
        """
        import json

        return json.dumps(self.ranking_weights)

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to dictionary."""
        return {
            "use_trigram_search": self.use_trigram_search,
            "trigram_similarity_threshold": self.trigram_similarity_threshold,
            "use_bm25_search": self.use_bm25_search,
            "use_sql_ranking": self.use_sql_ranking,
            "ranking_weights": self.ranking_weights,
            "bm25_weight": self.bm25_weight,
            "bm25_fetch_multiplier": self.bm25_fetch_multiplier,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SearchConfig:
        """Create configuration from dictionary."""
        return cls(
            use_trigram_search=data.get("use_trigram_search", True),
            trigram_similarity_threshold=data.get("trigram_similarity_threshold", 0.2),
            use_bm25_search=data.get("use_bm25_search", False),
            use_sql_ranking=data.get("use_sql_ranking", True),
            ranking_weights=data.get(
                "ranking_weights",
                {
                    "content": 0.15,
                    "topic": 0.25,
                    "recency": 0.15,
                    "reinforcement": 0.2,
                    "importance": 0.15,
                    "popularity": 0.1,
                },
            ),
            bm25_weight=data.get("bm25_weight", 0.35),
            bm25_fetch_multiplier=data.get("bm25_fetch_multiplier", 2),
        )


# Preset configurations for common use cases
SEARCH_CONFIG_DEFAULT = SearchConfig()

SEARCH_CONFIG_RECENCY_FOCUSED = SearchConfig(
    ranking_weights={
        "content": 0.1,
        "topic": 0.15,
        "recency": 0.35,
        "reinforcement": 0.2,
        "importance": 0.1,
        "popularity": 0.1,
    }
)

SEARCH_CONFIG_TOPIC_FOCUSED = SearchConfig(
    ranking_weights={
        "content": 0.1,
        "topic": 0.4,
        "recency": 0.1,
        "reinforcement": 0.2,
        "importance": 0.1,
        "popularity": 0.1,
    }
)

SEARCH_CONFIG_REINFORCEMENT_FOCUSED = SearchConfig(
    ranking_weights={
        "content": 0.1,
        "topic": 0.2,
        "recency": 0.1,
        "reinforcement": 0.35,
        "importance": 0.15,
        "popularity": 0.1,
    }
)
