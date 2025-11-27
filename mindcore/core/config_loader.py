"""
Configuration loader for Mindcore framework.

Supports environment variable substitution in YAML files:
- ${VAR} - Required variable, empty string if not set
- ${VAR:default} - Variable with default value
"""
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml


class ConfigLoader:
    """Loads and manages configuration from YAML file."""

    # Pattern for environment variable substitution: ${VAR} or ${VAR:default}
    # \$ in regex matches literal $ character
    ENV_VAR_PATTERN = re.compile(r'\$\{([^}:]+)(?::([^}]*))?\}')

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config loader.

        Args:
            config_path: Path to config.yaml file. If None, looks in default locations.
        """
        self.config_path = self._resolve_config_path(config_path)
        self.config: Dict[str, Any] = self._load_config()

    def _resolve_config_path(self, config_path: Optional[str]) -> Path:
        """
        Resolve the configuration file path.

        Args:
            config_path: Optional path to config file.

        Returns:
            Resolved Path object.
        """
        if config_path:
            return Path(config_path)

        # Check environment variable
        env_path = os.getenv("MINDCORE_CONFIG")
        if env_path:
            return Path(env_path)

        # Check default locations
        default_locations = [
            Path.cwd() / "config.yaml",
            Path.cwd() / "mindcore" / "config.yaml",
            Path(__file__).parent.parent / "config.yaml",
        ]

        for path in default_locations:
            if path.exists():
                return path

        # Return default location even if it doesn't exist
        return Path(__file__).parent.parent / "config.yaml"

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Returns:
            Configuration dictionary with environment variables resolved.
        """
        if not self.config_path.exists():
            # Return default configuration
            return self._default_config()

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Resolve environment variables in the config
        config = self._resolve_env_vars(config) if config else {}

        return config or self._default_config()

    def _resolve_env_vars(self, obj: Any) -> Any:
        """
        Recursively resolve environment variables in config values.

        Supports:
        - ${VAR} - Returns env var value or empty string
        - ${VAR:default} - Returns env var value or default

        Args:
            obj: Config object (dict, list, or scalar)

        Returns:
            Object with environment variables resolved
        """
        if isinstance(obj, dict):
            return {k: self._resolve_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._resolve_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            return self._substitute_env_vars(obj)
        else:
            return obj

    def _substitute_env_vars(self, value: str) -> Union[str, int, float, None]:
        """
        Substitute environment variables in a string value.

        Args:
            value: String potentially containing ${VAR} or ${VAR:default}

        Returns:
            String with env vars substituted, or typed value if entire string is a var
        """
        def replace_match(match):
            var_name = match.group(1)
            default = match.group(2)  # None if no default specified
            env_value = os.getenv(var_name)

            if env_value is not None:
                return env_value
            elif default is not None:
                return default
            else:
                return ""

        result = self.ENV_VAR_PATTERN.sub(replace_match, value)

        # Try to convert to appropriate type if the entire string was replaced
        if result != value and result:
            # Try int
            try:
                return int(result)
            except ValueError:
                pass
            # Try float
            try:
                return float(result)
            except ValueError:
                pass

        return result if result else None

    def _default_config(self) -> Dict[str, Any]:
        """
        Return default configuration.

        Returns:
            Default configuration dictionary.
        """
        return {
            "llm": {
                "provider": "auto",
                "llama_cpp": {
                    "model_path": os.getenv("MINDCORE_LLAMA_MODEL_PATH"),
                    "n_ctx": 4096,
                    "n_threads": None,
                    "n_gpu_layers": 0,
                    "chat_format": None,
                    "verbose": False,
                },
                "openai": {
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "base_url": os.getenv("MINDCORE_OPENAI_BASE_URL"),
                    "model": os.getenv("MINDCORE_OPENAI_MODEL", "gpt-4o-mini"),
                    "timeout": 60,
                    "max_retries": 3,
                },
                "defaults": {
                    "temperature": 0.3,
                    "max_tokens_enrichment": 800,
                    "max_tokens_context": 1500,
                },
            },
            "database": {
                "host": os.getenv("DB_HOST", "localhost"),
                "port": int(os.getenv("DB_PORT", "5432")),
                "database": os.getenv("DB_NAME", "mindcore"),
                "user": os.getenv("DB_USER", "postgres"),
                "password": os.getenv("DB_PASSWORD", "postgres"),
            },
            "cache": {
                "max_size": 50,
                "ttl": 3600,
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "debug": False,
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Configuration key (supports dot notation, e.g., 'database.host').
            default: Default value if key not found.

        Returns:
            Configuration value.
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        return self.config.get("database", {})

    def get_openai_config(self) -> Dict[str, Any]:
        """
        Get OpenAI configuration from llm.openai section.

        Returns:
            OpenAI configuration dictionary with api_key, model, temperature, max_tokens.
        """
        llm_openai = self.config.get("llm", {}).get("openai", {})
        defaults = self._default_config()["llm"]["openai"]
        gen_defaults = self.config.get("llm", {}).get("defaults", {})

        return {
            "api_key": llm_openai.get("api_key") or defaults.get("api_key"),
            "model": llm_openai.get("model") or defaults.get("model", "gpt-4o-mini"),
            "temperature": gen_defaults.get("temperature", 0.3),
            "max_tokens": gen_defaults.get("max_tokens_enrichment", 800),
            "timeout": llm_openai.get("timeout") or defaults.get("timeout", 60),
            "max_retries": llm_openai.get("max_retries") or defaults.get("max_retries", 3),
        }

    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache configuration."""
        return self.config.get("cache", {})

    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration."""
        return self.config.get("api", {})

    def get_llm_config(self) -> Dict[str, Any]:
        """
        Get LLM provider configuration.

        Returns a structured dict with:
        - provider: "auto", "llama_cpp", or "openai"
        - llama_cpp: Config dict for LlamaCppProvider
        - openai: Config dict for OpenAIProvider
        - defaults: Default generation settings

        Returns:
            LLM configuration dictionary.
        """
        llm_config = self.config.get("llm", {})
        defaults = self._default_config()["llm"]

        # Get llama_cpp config with defaults
        llama_cpp_config = llm_config.get("llama_cpp", {})
        llama_defaults = defaults["llama_cpp"]

        # Get openai config with defaults
        openai_config = llm_config.get("openai", {})
        openai_defaults = defaults["openai"]

        # Get generation defaults
        gen_defaults = llm_config.get("defaults", {})
        gen_defaults_default = defaults["defaults"]

        return {
            "provider": llm_config.get("provider", defaults["provider"]),
            "llama_cpp": {
                "model_path": llama_cpp_config.get("model_path") or llama_defaults["model_path"],
                "n_ctx": llama_cpp_config.get("n_ctx", llama_defaults["n_ctx"]),
                "n_threads": llama_cpp_config.get("n_threads", llama_defaults["n_threads"]),
                "n_gpu_layers": llama_cpp_config.get("n_gpu_layers", llama_defaults["n_gpu_layers"]),
                "chat_format": llama_cpp_config.get("chat_format", llama_defaults["chat_format"]),
                "verbose": llama_cpp_config.get("verbose", llama_defaults["verbose"]),
            },
            "openai": {
                "api_key": openai_config.get("api_key") or openai_defaults["api_key"],
                "base_url": openai_config.get("base_url") or openai_defaults.get("base_url"),
                "model": openai_config.get("model") or openai_defaults["model"],
                "timeout": openai_config.get("timeout", openai_defaults["timeout"]),
                "max_retries": openai_config.get("max_retries", openai_defaults["max_retries"]),
            },
            "defaults": {
                "temperature": gen_defaults.get("temperature", gen_defaults_default["temperature"]),
                "max_tokens_enrichment": gen_defaults.get(
                    "max_tokens_enrichment",
                    gen_defaults_default["max_tokens_enrichment"]
                ),
                "max_tokens_context": gen_defaults.get(
                    "max_tokens_context",
                    gen_defaults_default["max_tokens_context"]
                ),
            },
        }

    def reload(self) -> None:
        """Reload configuration from file."""
        self.config = self._load_config()
