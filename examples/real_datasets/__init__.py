"""Real industry-standard datasets for Mindcore benchmarking.

This module provides:
1. Dataset downloaders for LoCoMo, Persona-Chat, and MultiWOZ
2. PostgreSQL storage with full SVL-compliant metadata
3. Test scenarios for CLST (cold-path) and FLR (hot-path)
4. SVL validation testing across both scenarios

Datasets:
---------
- LoCoMo: Long-context conversational memory (Stanford SNAP)
  https://github.com/snap-stanford/locomo
  Paper: https://arxiv.org/abs/2402.17753

- Persona-Chat: Persona-based dialogues (Facebook Research)
  https://huggingface.co/datasets/bavard/personachat_truecased
  Paper: https://arxiv.org/abs/1801.07243

- MultiWOZ: Multi-domain task-oriented dialogues
  https://huggingface.co/datasets/multi_woz_v22
  Paper: https://arxiv.org/abs/1810.00278

Usage:
------
    # Complete pipeline
    from examples.real_datasets import DatasetPipeline, PipelineConfig

    config = PipelineConfig(
        postgres_dsn="postgresql://localhost:5432/mindcore_datasets",
        llm_provider="openai",  # or "anthropic", "local"
        llm_api_key="your-api-key",
    )

    pipeline = DatasetPipeline(config)
    results = pipeline.run()
    pipeline.print_summary()

    # Or run individual components
    from examples.real_datasets import DatasetDownloader, DatasetMetadataEnricher

    downloader = DatasetDownloader()
    segment = downloader.download_locomo(max_sessions=50)

    enricher = DatasetMetadataEnricher()
    memories = enricher.enrich_segment(segment)

CLI Usage:
----------
    # Run with local enrichment
    python -m examples.real_datasets.run_benchmark

    # Run with OpenAI
    python -m examples.real_datasets.run_benchmark --llm openai --api-key $OPENAI_API_KEY

    # Run specific datasets
    python -m examples.real_datasets.run_benchmark --datasets locomo,persona_chat
"""


# Lazy imports to avoid import errors if dependencies not installed
def __getattr__(name):
    if name == "DatasetDownloader":
        from examples.real_datasets.downloader import DatasetDownloader

        return DatasetDownloader
    if name == "DatasetMetadataEnricher":
        from examples.real_datasets.enrichment import DatasetMetadataEnricher

        return DatasetMetadataEnricher
    if name == "EnrichmentConfig":
        from examples.real_datasets.enrichment import EnrichmentConfig

        return EnrichmentConfig
    if name == "PostgresDatasetStore":
        from examples.real_datasets.postgres_store import PostgresDatasetStore

        return PostgresDatasetStore
    if name == "EnrichedMemory":
        from examples.real_datasets.postgres_store import EnrichedMemory

        return EnrichedMemory
    if name == "DatasetPipeline":
        from examples.real_datasets.pipeline import DatasetPipeline

        return DatasetPipeline
    if name == "PipelineConfig":
        from examples.real_datasets.pipeline import PipelineConfig

        return PipelineConfig
    if name == "CLSTTestScenario":
        from examples.real_datasets.test_scenarios import CLSTTestScenario

        return CLSTTestScenario
    if name == "FLRTestScenario":
        from examples.real_datasets.test_scenarios import FLRTestScenario

        return FLRTestScenario
    if name == "SVLValidationTest":
        from examples.real_datasets.test_scenarios import SVLValidationTest

        return SVLValidationTest
    if name == "DatasetSegment":
        from examples.real_datasets.downloader import DatasetSegment

        return DatasetSegment
    if name == "ConversationSession":
        from examples.real_datasets.downloader import ConversationSession

        return ConversationSession
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Downloader
    "DatasetDownloader",
    "DatasetSegment",
    "ConversationSession",
    # Enrichment
    "DatasetMetadataEnricher",
    "EnrichmentConfig",
    # Storage
    "PostgresDatasetStore",
    "EnrichedMemory",
    # Pipeline
    "DatasetPipeline",
    "PipelineConfig",
    # Test scenarios
    "CLSTTestScenario",
    "FLRTestScenario",
    "SVLValidationTest",
]
