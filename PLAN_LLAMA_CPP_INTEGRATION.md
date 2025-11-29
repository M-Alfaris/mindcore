# Llama.cpp Integration Plan for Mindcore

## Executive Summary

This plan outlines how to integrate **llama.cpp** into Mindcore for CPU-only agents that handle:
1. **Metadata Enrichment** - Enriching messages with topics, categories, importance, sentiment, intent, tags, entities, and key phrases
2. **Context Retrieval** - Fast, low-latency retrieval assistance without blocking the main workflow

OpenAI API will remain as a fallback for reliability, but llama.cpp becomes the **primary** inference engine for cost-effective, memory-efficient, and offline-capable operation.

---

## Architecture Overview

### Current Architecture
```
MindcoreClient
├── MetadataAgent (OpenAI GPT-4o-mini)
├── ContextAgent (OpenAI GPT-4o-mini)
├── Database Layer
└── Cache Layer
```

### Proposed Architecture
```
MindcoreClient
├── LLMProvider (abstraction layer)
│   ├── LlamaCppProvider (primary - CPU-optimized)
│   └── OpenAIProvider (fallback)
├── MetadataAgent (uses LLMProvider)
├── ContextAgent (uses LLMProvider)
├── Database Layer
└── Cache Layer
```

---

## Implementation Plan

### Phase 1: LLM Provider Abstraction Layer

**Goal:** Create a unified interface that abstracts LLM backends

#### 1.1 Create `mindcore/llm/__init__.py`
```python
from .base_provider import BaseLLMProvider, LLMResponse
from .llama_cpp_provider import LlamaCppProvider
from .openai_provider import OpenAIProvider
from .provider_factory import create_provider, ProviderType

__all__ = [
    "BaseLLMProvider",
    "LLMResponse",
    "LlamaCppProvider",
    "OpenAIProvider",
    "create_provider",
    "ProviderType",
]
```

#### 1.2 Create `mindcore/llm/base_provider.py`
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class LLMResponse:
    """Unified response from any LLM provider."""
    content: str
    model: str
    tokens_used: Optional[int] = None
    provider: str = "unknown"
    latency_ms: Optional[float] = None

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1000,
        json_mode: bool = False
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging."""
        pass
```

#### 1.3 Create `mindcore/llm/llama_cpp_provider.py`
```python
import os
import time
from typing import List, Dict, Optional
from .base_provider import BaseLLMProvider, LLMResponse
from ..utils.logger import get_logger

logger = get_logger(__name__)

class LlamaCppProvider(BaseLLMProvider):
    """
    llama.cpp provider for CPU-optimized local inference.

    Uses llama-cpp-python bindings for efficient CPU inference.
    Supports GGUF models optimized for memory efficiency.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,  # 0 = CPU only
        verbose: bool = False
    ):
        """
        Initialize llama.cpp provider.

        Args:
            model_path: Path to GGUF model file
            n_ctx: Context window size (default 4096)
            n_threads: CPU threads (None = auto-detect)
            n_gpu_layers: GPU layers to offload (0 = CPU only)
            verbose: Enable verbose logging
        """
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads or os.cpu_count()
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        self._llm = None
        self._load_model()

    def _load_model(self):
        """Lazy load the model."""
        try:
            from llama_cpp import Llama

            logger.info(f"Loading llama.cpp model from {self.model_path}")
            self._llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                verbose=self.verbose,
                chat_format="chatml"  # Works with most models
            )
            logger.info(f"Model loaded successfully with {self.n_threads} threads")
        except ImportError:
            logger.error("llama-cpp-python not installed. Run: pip install llama-cpp-python")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1000,
        json_mode: bool = False
    ) -> LLMResponse:
        """Generate response using llama.cpp."""
        start_time = time.time()

        try:
            response = self._llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"} if json_mode else None
            )

            content = response["choices"][0]["message"]["content"]
            tokens_used = response.get("usage", {}).get("total_tokens")
            latency_ms = (time.time() - start_time) * 1000

            return LLMResponse(
                content=content,
                model=os.path.basename(self.model_path),
                tokens_used=tokens_used,
                provider="llama.cpp",
                latency_ms=latency_ms
            )
        except Exception as e:
            logger.error(f"llama.cpp generation failed: {e}")
            raise

    def is_available(self) -> bool:
        """Check if llama.cpp is available."""
        return self._llm is not None

    @property
    def name(self) -> str:
        return "llama.cpp"
```

#### 1.4 Create `mindcore/llm/openai_provider.py`
```python
import time
from typing import List, Dict, Optional
from openai import OpenAI, APIError, RateLimitError
from .base_provider import BaseLLMProvider, LLMResponse
from ..utils.logger import get_logger

logger = get_logger(__name__)

class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider (fallback)."""

    MAX_RETRIES = 3
    RETRY_DELAY = 1.0

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        self._client = None
        if api_key:
            self._client = OpenAI(api_key=api_key, timeout=60)

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1000,
        json_mode: bool = False
    ) -> LLMResponse:
        """Generate response using OpenAI API."""
        start_time = time.time()

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(self.MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                tokens_used = response.usage.total_tokens if response.usage else None
                latency_ms = (time.time() - start_time) * 1000

                return LLMResponse(
                    content=content,
                    model=self.model,
                    tokens_used=tokens_used,
                    provider="openai",
                    latency_ms=latency_ms
                )
            except RateLimitError:
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY * (2 ** attempt))
                else:
                    raise
            except Exception as e:
                logger.error(f"OpenAI generation failed: {e}")
                raise

    def is_available(self) -> bool:
        return self._client is not None and bool(self.api_key)

    @property
    def name(self) -> str:
        return "openai"
```

#### 1.5 Create `mindcore/llm/provider_factory.py`
```python
from enum import Enum
from typing import Optional
from .base_provider import BaseLLMProvider
from .llama_cpp_provider import LlamaCppProvider
from .openai_provider import OpenAIProvider
from ..utils.logger import get_logger

logger = get_logger(__name__)

class ProviderType(Enum):
    LLAMA_CPP = "llama_cpp"
    OPENAI = "openai"
    AUTO = "auto"  # Try llama.cpp first, fallback to OpenAI

class FallbackProvider(BaseLLMProvider):
    """Provider with automatic fallback."""

    def __init__(self, primary: BaseLLMProvider, fallback: BaseLLMProvider):
        self.primary = primary
        self.fallback = fallback

    def generate(self, messages, temperature=0.3, max_tokens=1000, json_mode=False):
        try:
            if self.primary.is_available():
                return self.primary.generate(messages, temperature, max_tokens, json_mode)
        except Exception as e:
            logger.warning(f"Primary provider failed: {e}, using fallback")

        return self.fallback.generate(messages, temperature, max_tokens, json_mode)

    def is_available(self) -> bool:
        return self.primary.is_available() or self.fallback.is_available()

    @property
    def name(self) -> str:
        return f"{self.primary.name}+{self.fallback.name}"

def create_provider(
    provider_type: ProviderType,
    llama_config: Optional[dict] = None,
    openai_config: Optional[dict] = None
) -> BaseLLMProvider:
    """
    Factory function to create LLM providers.

    Args:
        provider_type: Type of provider to create
        llama_config: Config for llama.cpp (model_path, n_ctx, n_threads, etc.)
        openai_config: Config for OpenAI (api_key, model)

    Returns:
        Configured LLM provider
    """
    llama_config = llama_config or {}
    openai_config = openai_config or {}

    if provider_type == ProviderType.LLAMA_CPP:
        return LlamaCppProvider(**llama_config)

    elif provider_type == ProviderType.OPENAI:
        return OpenAIProvider(**openai_config)

    elif provider_type == ProviderType.AUTO:
        # Create both providers, with llama.cpp as primary
        primary = None
        fallback = None

        if llama_config.get("model_path"):
            try:
                primary = LlamaCppProvider(**llama_config)
            except Exception as e:
                logger.warning(f"Could not initialize llama.cpp: {e}")

        if openai_config.get("api_key"):
            fallback = OpenAIProvider(**openai_config)

        if primary and fallback:
            return FallbackProvider(primary, fallback)
        elif primary:
            return primary
        elif fallback:
            return fallback
        else:
            raise ValueError("No LLM provider could be initialized")

    raise ValueError(f"Unknown provider type: {provider_type}")
```

---

### Phase 2: Update Configuration

#### 2.1 Update `config.yaml`
```yaml
# Mindcore Configuration File
# Version: 0.2.0

# LLM Provider Configuration
llm:
  # Provider mode: "llama_cpp", "openai", or "auto"
  # "auto" uses llama.cpp as primary with OpenAI fallback
  provider: auto

  # llama.cpp Configuration (CPU-optimized local inference)
  llama_cpp:
    # Path to GGUF model file
    # Recommended: Llama-3.2-3B-Instruct-Q4_K_M.gguf for good balance
    model_path: ${MINDCORE_LLAMA_MODEL_PATH}

    # Context window size (tokens)
    n_ctx: 4096

    # CPU threads (null = auto-detect based on CPU cores)
    n_threads: null

    # GPU layers to offload (0 = pure CPU, -1 = all to GPU if available)
    n_gpu_layers: 0

    # Generation settings
    temperature: 0.3
    max_tokens: 1000

# OpenAI Configuration (fallback)
openai:
  api_key: ${OPENAI_API_KEY}
  model: gpt-4o-mini
  temperature: 0.3
  max_tokens: 1000

# Database Configuration
database:
  host: localhost
  port: 5432
  database: mindcore
  user: postgres
  password: postgres

# Cache Configuration
cache:
  max_size: 50
  ttl: 3600

# API Server Configuration
api:
  host: 0.0.0.0
  port: 8000
  debug: false
  cors:
    allow_origins: ["*"]
    allow_credentials: true

# Logging Configuration
logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

#### 2.2 Update `ConfigLoader`
Add new method in `mindcore/core/config_loader.py`:
```python
def get_llm_config(self) -> Dict[str, Any]:
    """Get LLM provider configuration."""
    return {
        "provider": self.get("llm.provider", "auto"),
        "llama_cpp": {
            "model_path": self.get("llm.llama_cpp.model_path"),
            "n_ctx": self.get("llm.llama_cpp.n_ctx", 4096),
            "n_threads": self.get("llm.llama_cpp.n_threads"),
            "n_gpu_layers": self.get("llm.llama_cpp.n_gpu_layers", 0),
            "temperature": self.get("llm.llama_cpp.temperature", 0.3),
            "max_tokens": self.get("llm.llama_cpp.max_tokens", 1000),
        },
        "openai": self.get_openai_config()
    }
```

---

### Phase 3: Refactor Agents to Use Provider

#### 3.1 Update `BaseAgent`
Modify `mindcore/agents/base_agent.py`:
```python
class BaseAgent(ABC):
    """Abstract base class for AI agents using LLM providers."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        temperature: float = 0.3,
        max_tokens: int = 1000
    ):
        self.llm = llm_provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info(f"Initialized {self.__class__.__name__} with {llm_provider.name}")

    def _call_llm(
        self,
        messages: list,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Call LLM provider with unified interface."""
        response = self.llm.generate(
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            json_mode=json_mode
        )
        logger.debug(f"LLM call: {response.provider}, {response.latency_ms:.0f}ms")
        return response.content

    # Keep _call_openai for backward compatibility
    def _call_openai(self, messages, response_format=None, **kwargs):
        """Backward compatible method."""
        json_mode = response_format is not None
        return self._call_llm(messages, json_mode=json_mode, **kwargs)
```

#### 3.2 Update `EnrichmentAgent` and `ContextAssemblerAgent`
Update constructors to accept LLM provider:
```python
class EnrichmentAgent(BaseAgent):
    def __init__(
        self,
        llm_provider: BaseLLMProvider = None,
        # Legacy parameters for backward compatibility
        api_key: str = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3
    ):
        # Support legacy initialization
        if llm_provider is None and api_key:
            from ..llm import OpenAIProvider
            llm_provider = OpenAIProvider(api_key, model)

        super().__init__(llm_provider, temperature, max_tokens=800)
        self.system_prompt = self._create_system_prompt()
```

---

### Phase 4: Update MindcoreClient

#### 4.1 Modify `MindcoreClient.__init__`
```python
def __init__(
    self,
    config_path: Optional[str] = None,
    use_sqlite: bool = False,
    sqlite_path: str = "mindcore.db",
    llm_provider: Optional[str] = None  # "llama_cpp", "openai", "auto"
):
    # ... existing initialization ...

    # Initialize LLM provider
    llm_config = self.config.get_llm_config()
    provider_type = ProviderType(llm_provider or llm_config.get("provider", "auto"))

    self.llm_provider = create_provider(
        provider_type=provider_type,
        llama_config=llm_config.get("llama_cpp"),
        openai_config=llm_config.get("openai")
    )

    # Initialize agents with the provider
    self.metadata_agent = MetadataAgent(
        llm_provider=self.llm_provider,
        temperature=llm_config.get("llama_cpp", {}).get("temperature", 0.3)
    )

    self.context_agent = ContextAgent(
        llm_provider=self.llm_provider,
        temperature=llm_config.get("llama_cpp", {}).get("temperature", 0.3)
    )
```

---

### Phase 5: Model Download & Setup Utilities

#### 5.1 Create `mindcore/llm/model_manager.py`
```python
import os
import hashlib
from pathlib import Path
from typing import Optional
import urllib.request
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Recommended models for different use cases
RECOMMENDED_MODELS = {
    "default": {
        "name": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "url": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "size_mb": 2020,
        "description": "Good balance of speed and quality for enrichment"
    },
    "small": {
        "name": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "url": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "size_mb": 820,
        "description": "Fastest, lowest memory for simple enrichment"
    },
    "quality": {
        "name": "Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "url": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "size_mb": 4920,
        "description": "Higher quality for complex context assembly"
    }
}

def get_models_dir() -> Path:
    """Get default models directory."""
    return Path.home() / ".mindcore" / "models"

def download_model(
    model_key: str = "default",
    target_dir: Optional[Path] = None,
    show_progress: bool = True
) -> str:
    """
    Download a recommended model.

    Args:
        model_key: Key from RECOMMENDED_MODELS ("default", "small", "quality")
        target_dir: Directory to save model (default: ~/.mindcore/models)
        show_progress: Show download progress

    Returns:
        Path to downloaded model
    """
    if model_key not in RECOMMENDED_MODELS:
        raise ValueError(f"Unknown model: {model_key}. Choose from: {list(RECOMMENDED_MODELS.keys())}")

    model_info = RECOMMENDED_MODELS[model_key]
    target_dir = target_dir or get_models_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    model_path = target_dir / model_info["name"]

    if model_path.exists():
        logger.info(f"Model already exists: {model_path}")
        return str(model_path)

    logger.info(f"Downloading {model_info['name']} ({model_info['size_mb']}MB)...")

    # Download with progress
    def progress_hook(count, block_size, total_size):
        if show_progress and total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            print(f"\rDownloading: {percent}%", end="", flush=True)

    urllib.request.urlretrieve(model_info["url"], model_path, progress_hook)

    if show_progress:
        print()  # New line after progress

    logger.info(f"Model downloaded to: {model_path}")
    return str(model_path)

def list_local_models(models_dir: Optional[Path] = None) -> list:
    """List locally available models."""
    models_dir = models_dir or get_models_dir()
    if not models_dir.exists():
        return []
    return [f.name for f in models_dir.glob("*.gguf")]
```

#### 5.2 Create CLI Setup Script `mindcore/cli.py`
```python
import argparse
from .llm.model_manager import download_model, list_local_models, RECOMMENDED_MODELS

def main():
    parser = argparse.ArgumentParser(description="Mindcore CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Download model command
    download_parser = subparsers.add_parser("download-model", help="Download a recommended model")
    download_parser.add_argument(
        "--model", "-m",
        choices=list(RECOMMENDED_MODELS.keys()),
        default="default",
        help="Model to download (default: default)"
    )

    # List models command
    subparsers.add_parser("list-models", help="List local models")

    args = parser.parse_args()

    if args.command == "download-model":
        path = download_model(args.model)
        print(f"Model ready at: {path}")
        print(f"Set MINDCORE_LLAMA_MODEL_PATH={path}")

    elif args.command == "list-models":
        models = list_local_models()
        if models:
            print("Local models:")
            for m in models:
                print(f"  - {m}")
        else:
            print("No local models found. Run: mindcore download-model")

if __name__ == "__main__":
    main()
```

---

### Phase 6: Update Dependencies

#### 6.1 Update `requirements.txt`
```
# Core dependencies
pyyaml>=6.0
psycopg2-binary>=2.9.0

# LLM Providers
openai>=1.0.0
llama-cpp-python>=0.2.0

# Optional: For GPU acceleration (uncomment if needed)
# llama-cpp-python[cuda]>=0.2.0  # NVIDIA GPU
# llama-cpp-python[metal]>=0.2.0  # Apple Silicon

# API Server
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0

# Utilities
python-dotenv>=1.0.0
```

#### 6.2 Update `setup.py` / `pyproject.toml`
Add entry point for CLI:
```python
entry_points={
    "console_scripts": [
        "mindcore=mindcore.cli:main",
    ],
}
```

---

### Phase 7: Quick Start Documentation

#### 7.1 Update README with Quick Start
```markdown
## Quick Start with Local LLM (No OpenAI Required!)

1. **Install Mindcore:**
   ```bash
   pip install mindcore
   ```

2. **Download a model:**
   ```bash
   mindcore download-model  # Downloads ~2GB Llama-3.2-3B
   ```

3. **Set the model path:**
   ```bash
   export MINDCORE_LLAMA_MODEL_PATH=~/.mindcore/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
   ```

4. **Use Mindcore:**
   ```python
   from mindcore import MindcoreClient

   # Automatically uses llama.cpp (falls back to OpenAI if configured)
   client = MindcoreClient(use_sqlite=True)

   # Ingest with automatic metadata enrichment
   message = client.ingest_message({
       "user_id": "user123",
       "thread_id": "thread456",
       "session_id": "session789",
       "role": "user",
       "text": "How do I implement caching in my app?"
   })

   print(message.metadata.topics)  # ['caching', 'implementation']
   print(message.metadata.intent)  # 'ask_question'
   ```

## Model Recommendations

| Use Case | Model | Size | RAM Required |
|----------|-------|------|--------------|
| Fast enrichment | Llama-3.2-1B | 820MB | ~2GB |
| Balanced (default) | Llama-3.2-3B | 2GB | ~4GB |
| High quality | Llama-3.1-8B | 5GB | ~8GB |
```

---

## File Structure Summary

After implementation, the new structure will be:

```
mindcore/
├── __init__.py              # Updated with new imports
├── config.yaml              # Updated with llm section
├── cli.py                   # NEW: CLI for model management
│
├── llm/                     # NEW: LLM Provider Layer
│   ├── __init__.py
│   ├── base_provider.py     # Abstract provider interface
│   ├── llama_cpp_provider.py # llama.cpp implementation
│   ├── openai_provider.py   # OpenAI implementation
│   ├── provider_factory.py  # Factory with fallback logic
│   └── model_manager.py     # Model download utilities
│
├── core/
│   ├── config_loader.py     # Updated with get_llm_config()
│   └── ...
│
├── agents/
│   ├── base_agent.py        # Updated to use LLM providers
│   ├── enrichment_agent.py  # Updated constructor
│   └── context_assembler_agent.py  # Updated constructor
│
└── ...
```

---

## Implementation Order

1. **Phase 1** - Create LLM provider abstraction (foundation)
2. **Phase 2** - Update configuration system
3. **Phase 3** - Refactor agents to use providers
4. **Phase 4** - Update MindcoreClient initialization
5. **Phase 5** - Add model download utilities
6. **Phase 6** - Update dependencies
7. **Phase 7** - Update documentation

---

## Benefits

1. **Cost Reduction**: No API costs for metadata enrichment and retrieval
2. **Memory Efficiency**: Quantized models use 2-8GB RAM vs cloud API costs
3. **Speed**: Local inference = no network latency (~50-200ms vs 500-2000ms)
4. **Offline Capable**: Works without internet connection
5. **Privacy**: All data stays local
6. **Fallback Safety**: OpenAI API as automatic fallback if local fails
7. **Easy Setup**: One command to download model, works out of the box
8. **Backward Compatible**: Existing OpenAI-only code continues to work

---

## Estimated Effort

- Phase 1: LLM Provider Layer - Core abstraction
- Phase 2: Configuration Updates - Config changes
- Phase 3: Agent Refactoring - Backward-compatible updates
- Phase 4: Client Updates - Integration
- Phase 5: Model Management - User experience
- Phase 6: Dependencies - Build changes
- Phase 7: Documentation - Docs

Total: Moderate effort, high impact
