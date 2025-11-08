"""
Configuration loader for Mindcore framework.
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


class ConfigLoader:
    """Loads and manages configuration from YAML file."""

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
            Configuration dictionary.
        """
        if not self.config_path.exists():
            # Return default configuration
            return self._default_config()

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config or self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """
        Return default configuration.

        Returns:
            Default configuration dictionary.
        """
        return {
            "database": {
                "host": os.getenv("DB_HOST", "localhost"),
                "port": int(os.getenv("DB_PORT", "5432")),
                "database": os.getenv("DB_NAME", "mindcore"),
                "user": os.getenv("DB_USER", "postgres"),
                "password": os.getenv("DB_PASSWORD", "postgres"),
            },
            "openai": {
                "api_key": os.getenv("OPENAI_API_KEY", ""),
                "model": "gpt-4o-mini",
                "temperature": 0.3,
                "max_tokens": 1000,
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
        """Get OpenAI configuration."""
        return self.config.get("openai", {})

    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache configuration."""
        return self.config.get("cache", {})

    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration."""
        return self.config.get("api", {})

    def reload(self) -> None:
        """Reload configuration from file."""
        self.config = self._load_config()
