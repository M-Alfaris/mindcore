"""Mindcore Benchmarks - System-Level Evaluation Suite.

This module provides comprehensive benchmarks that go beyond typical model
benchmarks (accuracy, latency) to measure what actually matters for production
AI memory systems:

1. DETERMINISM & REPRODUCIBILITY - Can you replay and get identical results?
2. AUDITABILITY & TRACEABILITY - Can you explain what happened and why?
3. MEMORY QUALITY OVER TIME - Does memory drift or remain stable?
4. COST & OPERATIONAL EFFICIENCY - FLR vs CLST cost/latency trade-offs
5. FAILURE & ADVERSARIAL ROBUSTNESS - Does the system degrade gracefully?

Benchmark Categories:
--------------------
- Standard Benchmarks: Industry datasets for credibility (LoCoMo, MultiWOZ, etc.)
- System Benchmarks: Unique to Mindcore (deterministic replay, audit time, drift)
- Performance Benchmarks: QPS, latency percentiles, recall@k
- Cost Benchmarks: Token usage, FLR vs CLST efficiency

Usage:
    from mindcore.benchmarks import BenchmarkRunner, BenchmarkSuite

    runner = BenchmarkRunner()
    results = runner.run_suite(BenchmarkSuite.FULL)
    runner.export_report("benchmark_results.json")

References:
    - LoCoMo: https://arxiv.org/abs/2402.17753
    - MemoryAgentBench: https://github.com/HUST-AI-HYZ/MemoryAgentBench
    - AgentBench: https://github.com/THUDM/AgentBench
    - VDBBench: https://milvus.io/blog/vdbbench-1-0-benchmarking-with-your-real-world-production-workloads.md
"""

from mindcore.benchmarks.dashboard import generate_dashboard
from mindcore.benchmarks.datasets import DatasetLoader
from mindcore.benchmarks.enrichment import (
    DatasetEnrichmentPipeline,
    EnrichedDataset,
    EnrichedMemory,
)
from mindcore.benchmarks.metrics import BenchmarkMetrics
from mindcore.benchmarks.runner import BenchmarkRunner, BenchmarkSuite


__all__ = [
    "BenchmarkMetrics",
    "BenchmarkRunner",
    "BenchmarkSuite",
    "DatasetEnrichmentPipeline",
    "DatasetLoader",
    "EnrichedDataset",
    "EnrichedMemory",
    "generate_dashboard",
]
