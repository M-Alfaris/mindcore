"""
Model registry and metadata for recommended GGUF models.

Provides curated list of models optimized for Mindcore's use cases:
- Metadata enrichment (fast, accurate JSON extraction)
- Context assembly (summarization, key point extraction)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ModelInfo:
    """Information about a downloadable model."""

    name: str
    description: str
    size_gb: float
    url: str
    filename: str
    recommended_for: List[str]
    min_ram_gb: int
    quantization: str
    base_model: str


# Curated list of recommended models for Mindcore
# Focus on small, fast models optimized for metadata extraction
RECOMMENDED_MODELS: Dict[str, ModelInfo] = {
    "llama-3.2-3b": ModelInfo(
        name="Llama 3.2 3B Instruct (Q4_K_M)",
        description="Meta's latest small model. Excellent for structured extraction.",
        size_gb=2.0,
        url="https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        recommended_for=["metadata", "context", "general"],
        min_ram_gb=4,
        quantization="Q4_K_M",
        base_model="meta-llama/Llama-3.2-3B-Instruct",
    ),
    "llama-3.2-1b": ModelInfo(
        name="Llama 3.2 1B Instruct (Q4_K_M)",
        description="Ultra-lightweight model. Fast but less capable.",
        size_gb=0.75,
        url="https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        filename="Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        recommended_for=["metadata", "low-resource"],
        min_ram_gb=2,
        quantization="Q4_K_M",
        base_model="meta-llama/Llama-3.2-1B-Instruct",
    ),
    "qwen2.5-3b": ModelInfo(
        name="Qwen 2.5 3B Instruct (Q4_K_M)",
        description="Alibaba's efficient model. Strong at structured output.",
        size_gb=2.1,
        url="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
        filename="qwen2.5-3b-instruct-q4_k_m.gguf",
        recommended_for=["metadata", "context", "multilingual"],
        min_ram_gb=4,
        quantization="Q4_K_M",
        base_model="Qwen/Qwen2.5-3B-Instruct",
    ),
    "phi-3.5-mini": ModelInfo(
        name="Phi 3.5 Mini Instruct (Q4_K_M)",
        description="Microsoft's compact model. Good reasoning capabilities.",
        size_gb=2.2,
        url="https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf",
        filename="Phi-3.5-mini-instruct-Q4_K_M.gguf",
        recommended_for=["metadata", "context", "reasoning"],
        min_ram_gb=4,
        quantization="Q4_K_M",
        base_model="microsoft/Phi-3.5-mini-instruct",
    ),
    "gemma-2-2b": ModelInfo(
        name="Gemma 2 2B Instruct (Q4_K_M)",
        description="Google's efficient model. Good instruction following.",
        size_gb=1.6,
        url="https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf",
        filename="gemma-2-2b-it-Q4_K_M.gguf",
        recommended_for=["metadata", "context"],
        min_ram_gb=3,
        quantization="Q4_K_M",
        base_model="google/gemma-2-2b-it",
    ),
    "smollm2-1.7b": ModelInfo(
        name="SmolLM2 1.7B Instruct (Q4_K_M)",
        description="HuggingFace's tiny powerhouse. Surprisingly capable.",
        size_gb=1.1,
        url="https://huggingface.co/bartowski/SmolLM2-1.7B-Instruct-GGUF/resolve/main/SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
        filename="SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
        recommended_for=["metadata", "low-resource"],
        min_ram_gb=2,
        quantization="Q4_K_M",
        base_model="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    ),
}

# Default model recommendation
DEFAULT_MODEL = "llama-3.2-3b"


def get_model_info(model_key: str) -> Optional[ModelInfo]:
    """Get model information by key."""
    return RECOMMENDED_MODELS.get(model_key)


def list_available_models() -> List[str]:
    """Get list of available model keys."""
    return list(RECOMMENDED_MODELS.keys())


def get_default_models_dir() -> str:
    """Get the default directory for storing models."""
    from pathlib import Path

    return str(Path.home() / ".mindcore" / "models")


def format_model_table() -> str:
    """Format models as a readable table."""
    lines = []
    lines.append("Available Models:")
    lines.append("-" * 80)
    lines.append(f"{'Key':<16} {'Name':<35} {'Size':<8} {'RAM':<6}")
    lines.append("-" * 80)

    for key, model in RECOMMENDED_MODELS.items():
        default_marker = " (default)" if key == DEFAULT_MODEL else ""
        lines.append(
            f"{key:<16} {model.name[:33]:<35} {model.size_gb:.1f} GB  {model.min_ram_gb}+ GB{default_marker}"
        )

    lines.append("-" * 80)
    return "\n".join(lines)
