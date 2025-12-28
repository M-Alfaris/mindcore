"""Complete pipeline for real dataset benchmarking.

This module provides a unified pipeline that:
1. Downloads real datasets (LoCoMo, Persona-Chat, MultiWOZ)
2. Enriches them with SVL-compliant metadata using LLM
3. Stores them in PostgreSQL
4. Runs comprehensive tests on CLST, FLR, and SVL

Usage:
    from examples.real_datasets import DatasetPipeline

    pipeline = DatasetPipeline(
        postgres_dsn="postgresql://localhost:5432/mindcore_datasets",
        llm_provider="openai",
        api_key="your-api-key",
    )

    # Run complete pipeline
    results = pipeline.run()
    pipeline.print_summary()

    # Or run individual steps
    pipeline.download()
    pipeline.enrich()
    pipeline.store()
    pipeline.test()
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from examples.real_datasets.downloader import DatasetDownloader, DatasetSegment
from examples.real_datasets.enrichment import DatasetMetadataEnricher, EnrichmentConfig
from examples.real_datasets.postgres_store import EnrichedMemory, PostgresDatasetStore
from examples.real_datasets.test_scenarios import (
    CLSTTestScenario,
    FLRTestScenario,
    ScenarioResult,
    SVLValidationTest,
)


logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the dataset pipeline."""

    # PostgreSQL settings
    postgres_dsn: str = "postgresql://localhost:5432/mindcore_datasets"
    schema_name: str = "datasets"
    recreate_schema: bool = False

    # Dataset settings
    datasets: list[str] = field(default_factory=lambda: ["locomo", "persona_chat", "multiwoz"])
    max_sessions_per_dataset: int = 50
    max_turns_per_session: int = 30

    # LLM enrichment settings
    llm_provider: str = "local"  # openai, anthropic, local
    llm_api_key: str | None = None
    llm_model: str | None = None

    # SVL vocabulary
    vocabulary_topics: list[str] = field(
        default_factory=lambda: [
            "general",
            "technology",
            "programming",
            "work",
            "personal",
            "travel",
            "food",
            "health",
            "entertainment",
            "communication",
            "settings",
            "preferences",
            "hotel",
            "restaurant",
            "train",
            "booking",
            "reservation",
            "schedule",
            "meeting",
            "project",
        ]
    )
    vocabulary_categories: list[str] = field(
        default_factory=lambda: [
            "general",
            "user_preference",
            "task_oriented",
            "work",
            "personal",
            "system",
            "booking",
            "inquiry",
        ]
    )

    # Output settings
    output_dir: str | Path = "./benchmark_output"
    save_intermediate: bool = True


@dataclass
class PipelineResult:
    """Result of the complete pipeline run."""

    started_at: datetime
    completed_at: datetime | None = None

    # Download results
    segments_downloaded: list[dict[str, Any]] = field(default_factory=list)
    total_sessions: int = 0
    total_turns: int = 0

    # Enrichment results
    memories_enriched: int = 0
    enrichment_errors: list[str] = field(default_factory=list)

    # Storage results
    memories_stored: int = 0

    # Test results
    clst_results: ScenarioResult | None = None
    flr_results: ScenarioResult | None = None
    svl_results: ScenarioResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "segments_downloaded": self.segments_downloaded,
            "total_sessions": self.total_sessions,
            "total_turns": self.total_turns,
            "memories_enriched": self.memories_enriched,
            "memories_stored": self.memories_stored,
            "enrichment_errors": self.enrichment_errors,
            "test_results": {
                "clst": self.clst_results.to_dict() if self.clst_results else None,
                "flr": self.flr_results.to_dict() if self.flr_results else None,
                "svl": self.svl_results.to_dict() if self.svl_results else None,
            },
        }

    @property
    def tests_passed(self) -> int:
        total = 0
        if self.clst_results:
            total += self.clst_results.passed
        if self.flr_results:
            total += self.flr_results.passed
        if self.svl_results:
            total += self.svl_results.passed
        return total

    @property
    def tests_total(self) -> int:
        total = 0
        if self.clst_results:
            total += self.clst_results.total
        if self.flr_results:
            total += self.flr_results.total
        if self.svl_results:
            total += self.svl_results.total
        return total


class DatasetPipeline:
    """Complete pipeline for real dataset benchmarking.

    Orchestrates the complete flow:
    1. Download real datasets from HuggingFace
    2. Enrich with SVL-compliant metadata
    3. Store in PostgreSQL
    4. Run CLST, FLR, and SVL tests
    """

    def __init__(self, config: PipelineConfig | None = None, **kwargs):
        """Initialize the pipeline.

        Args:
            config: Pipeline configuration
            **kwargs: Override config values
        """
        self.config = config or PipelineConfig()

        # Apply overrides
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Initialize components
        self.downloader = DatasetDownloader()
        self.enricher = DatasetMetadataEnricher(
            EnrichmentConfig(
                llm_provider=self.config.llm_provider,
                api_key=self.config.llm_api_key or os.environ.get("OPENAI_API_KEY"),
                model=self.config.llm_model,
                vocabulary_topics=self.config.vocabulary_topics,
                vocabulary_categories=self.config.vocabulary_categories,
            )
        )
        self.store = PostgresDatasetStore(
            dsn=self.config.postgres_dsn,
            schema_name=self.config.schema_name,
        )

        # State
        self._segments: list[DatasetSegment] = []
        self._memories: list[EnrichedMemory] = []
        self._result: PipelineResult | None = None

        # Ensure output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def run(self) -> PipelineResult:
        """Run the complete pipeline.

        Returns:
            PipelineResult with all results
        """
        self._result = PipelineResult(started_at=datetime.now(timezone.utc))

        try:
            logger.info("Starting dataset pipeline...")

            # Step 1: Download
            self.download()

            # Step 2: Enrich
            self.enrich()

            # Step 3: Store
            self.store_data()

            # Step 4: Test
            self.test()

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            self._result.enrichment_errors.append(str(e))

        finally:
            self._result.completed_at = datetime.now(timezone.utc)
            self._save_results()

        return self._result

    def download(self) -> list[DatasetSegment]:
        """Download datasets from HuggingFace.

        Returns:
            List of downloaded DatasetSegments
        """
        logger.info(f"Downloading {len(self.config.datasets)} datasets...")

        self._segments = []

        for dataset_name in self.config.datasets:
            logger.info(f"  Downloading {dataset_name}...")

            try:
                if dataset_name == "locomo":
                    segment = self.downloader.download_locomo(
                        max_sessions=self.config.max_sessions_per_dataset,
                        max_turns_per_session=self.config.max_turns_per_session,
                    )
                elif dataset_name == "persona_chat":
                    segment = self.downloader.download_persona_chat(
                        max_sessions=self.config.max_sessions_per_dataset,
                        max_turns_per_session=self.config.max_turns_per_session,
                    )
                elif dataset_name == "multiwoz":
                    segment = self.downloader.download_multiwoz(
                        max_sessions=self.config.max_sessions_per_dataset,
                        max_turns_per_session=self.config.max_turns_per_session,
                    )
                else:
                    logger.warning(f"Unknown dataset: {dataset_name}")
                    continue

                self._segments.append(segment)

                # Update result
                self._result.segments_downloaded.append(
                    {
                        "name": segment.dataset_name,
                        "sessions": segment.total_sessions,
                        "turns": segment.total_turns,
                    }
                )
                self._result.total_sessions += segment.total_sessions
                self._result.total_turns += segment.total_turns

                logger.info(
                    f"    Downloaded {segment.total_sessions} sessions, "
                    f"{segment.total_turns} turns"
                )

                # Save intermediate if configured
                if self.config.save_intermediate:
                    path = Path(self.config.output_dir) / f"{dataset_name}_raw.json"
                    self.downloader.save_segment(segment, path)

            except Exception as e:
                logger.error(f"  Error downloading {dataset_name}: {e}")
                self._result.enrichment_errors.append(f"Download {dataset_name}: {e}")

        logger.info(
            f"Downloaded {len(self._segments)} datasets: "
            f"{self._result.total_sessions} sessions, {self._result.total_turns} turns"
        )

        return self._segments

    def enrich(self) -> list[EnrichedMemory]:
        """Enrich downloaded segments with SVL metadata.

        Returns:
            List of EnrichedMemory objects
        """
        if not self._segments:
            logger.warning("No segments to enrich. Run download() first.")
            return []

        logger.info(f"Enriching {len(self._segments)} segments...")

        self._memories = []

        for segment in self._segments:
            logger.info(f"  Enriching {segment.dataset_name}...")

            try:

                def progress(current, total):
                    if current % 50 == 0 or current == total:
                        logger.info(f"    Progress: {current}/{total} turns")

                memories = self.enricher.enrich_segment(segment, progress_callback=progress)
                self._memories.extend(memories)

                logger.info(f"    Enriched {len(memories)} memories")

                # Save intermediate
                if self.config.save_intermediate:
                    path = Path(self.config.output_dir) / f"{segment.dataset_name}_enriched.json"
                    self._save_memories_json(memories, path)

            except Exception as e:
                logger.error(f"  Error enriching {segment.dataset_name}: {e}")
                self._result.enrichment_errors.append(f"Enrich {segment.dataset_name}: {e}")

        self._result.memories_enriched = len(self._memories)
        logger.info(f"Enriched {len(self._memories)} total memories")

        return self._memories

    def store_data(self) -> int:
        """Store enriched memories in PostgreSQL.

        Returns:
            Number of memories stored
        """
        if not self._memories:
            logger.warning("No memories to store. Run enrich() first.")
            return 0

        logger.info(f"Storing {len(self._memories)} memories in PostgreSQL...")

        try:
            # Setup schema
            if self.config.recreate_schema:
                logger.info("  Recreating schema...")
                self.store.drop_schema(cascade=True)
            self.store.create_schema()

            # Store sessions first
            sessions_stored = set()
            for segment in self._segments:
                for session in segment.sessions:
                    if session.session_id not in sessions_stored:
                        self.store.store_session(
                            session_id=session.session_id,
                            user_id=session.user_id,
                            dataset_name=segment.dataset_name,
                            persona=session.persona,
                            domain=session.domain,
                            total_turns=len(session.turns),
                            metadata=session.metadata,
                        )
                        sessions_stored.add(session.session_id)

            logger.info(f"  Stored {len(sessions_stored)} sessions")

            # Store memories
            self.store.store_memories(self._memories)
            self._result.memories_stored = len(self._memories)

            logger.info(f"  Stored {len(self._memories)} memories")

            # Get and log stats
            for segment in self._segments:
                stats = self.store.get_dataset_stats(segment.dataset_name)
                logger.info(
                    f"  {segment.dataset_name}: "
                    f"{stats['total_memories']} memories, "
                    f"{stats['unique_sessions']} sessions"
                )

        except Exception as e:
            logger.error(f"Error storing data: {e}")
            self._result.enrichment_errors.append(f"Store: {e}")
            return 0

        finally:
            self.store.close()

        return self._result.memories_stored

    def test(self) -> dict[str, ScenarioResult]:
        """Run all test scenarios.

        Returns:
            Dict of scenario name to results
        """
        logger.info("Running test scenarios...")
        results = {}

        # Run CLST tests
        logger.info("  Running CLST cold-path tests...")
        try:
            clst_scenario = CLSTTestScenario(postgres_dsn=self.config.postgres_dsn)
            self._result.clst_results = clst_scenario.run()
            results["clst"] = self._result.clst_results
            logger.info(
                f"    CLST: {self._result.clst_results.passed}/"
                f"{self._result.clst_results.total} passed"
            )
        except Exception as e:
            logger.error(f"    CLST error: {e}")

        # Run FLR tests
        logger.info("  Running FLR hot-path tests...")
        try:
            flr_scenario = FLRTestScenario(postgres_dsn=self.config.postgres_dsn)
            self._result.flr_results = flr_scenario.run()
            results["flr"] = self._result.flr_results
            logger.info(
                f"    FLR: {self._result.flr_results.passed}/"
                f"{self._result.flr_results.total} passed"
            )
        except Exception as e:
            logger.error(f"    FLR error: {e}")

        # Run SVL validation tests
        logger.info("  Running SVL validation tests...")
        try:
            svl_scenario = SVLValidationTest(postgres_dsn=self.config.postgres_dsn)
            self._result.svl_results = svl_scenario.run()
            results["svl"] = self._result.svl_results
            logger.info(
                f"    SVL: {self._result.svl_results.passed}/"
                f"{self._result.svl_results.total} passed"
            )
        except Exception as e:
            logger.error(f"    SVL error: {e}")

        logger.info(
            f"Tests complete: {self._result.tests_passed}/{self._result.tests_total} passed"
        )

        return results

    def print_summary(self) -> None:
        """Print a summary of the pipeline results."""
        if not self._result:
            print("No results available. Run the pipeline first.")
            return

        print("\n" + "=" * 60)
        print("DATASET PIPELINE RESULTS")
        print("=" * 60)

        # Download summary
        print("\nDownload Summary:")
        print(f"  Datasets: {len(self._result.segments_downloaded)}")
        print(f"  Sessions: {self._result.total_sessions}")
        print(f"  Turns: {self._result.total_turns}")
        for seg in self._result.segments_downloaded:
            print(f"    - {seg['name']}: {seg['sessions']} sessions, {seg['turns']} turns")

        # Enrichment summary
        print("\nEnrichment Summary:")
        print(f"  Memories enriched: {self._result.memories_enriched}")
        print(f"  Memories stored: {self._result.memories_stored}")
        if self._result.enrichment_errors:
            print(f"  Errors: {len(self._result.enrichment_errors)}")

        # Test summary
        print("\nTest Results:")
        print("-" * 40)

        if self._result.clst_results:
            print(
                f"\nCLST Cold-Path Tests: {self._result.clst_results.passed}/{self._result.clst_results.total}"
            )
            for test in self._result.clst_results.tests:
                status = "PASS" if test.passed else "FAIL"
                print(f"  [{status}] {test.name} ({test.duration_ms:.1f}ms)")

        if self._result.flr_results:
            print(
                f"\nFLR Hot-Path Tests: {self._result.flr_results.passed}/{self._result.flr_results.total}"
            )
            for test in self._result.flr_results.tests:
                status = "PASS" if test.passed else "FAIL"
                print(f"  [{status}] {test.name} ({test.duration_ms:.1f}ms)")

        if self._result.svl_results:
            print(
                f"\nSVL Validation Tests: {self._result.svl_results.passed}/{self._result.svl_results.total}"
            )
            for test in self._result.svl_results.tests:
                status = "PASS" if test.passed else "FAIL"
                print(f"  [{status}] {test.name} ({test.duration_ms:.1f}ms)")

        print("\n" + "-" * 40)
        print(f"TOTAL: {self._result.tests_passed}/{self._result.tests_total} tests passed")
        print("=" * 60)

    def _save_memories_json(self, memories: list[EnrichedMemory], path: Path) -> None:
        """Save memories to JSON file."""
        data = {
            "count": len(memories),
            "memories": [m.to_dict() for m in memories],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _save_results(self) -> None:
        """Save pipeline results to JSON file."""
        path = Path(self.config.output_dir) / "pipeline_results.json"
        with open(path, "w") as f:
            json.dump(self._result.to_dict(), f, indent=2)
        logger.info(f"Saved results to {path}")
