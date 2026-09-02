"""Configuration management for the pipeline.

Loads API keys and settings from config.yaml or environment variables.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class APIConfig:
    """API keys configuration."""
    newsapi_key: Optional[str] = None
    anthropic_key: Optional[str] = None


@dataclass
class CollectionConfig:
    """Data collection settings."""
    delay: float = 1.0
    yahoo_max_per_ticker: int = 10
    newsapi_days_back: int = 7
    newsapi_articles_per_company: int = 20
    edgar_filings_per_company: int = 10
    sec_user_agent: str = "PersonalResearch contact@example.com"


@dataclass
class ModelConfig:
    """Model settings."""
    encoder_name: str = "allenai/longformer-base-4096"
    hidden_size: int = 768
    num_attention_heads: int = 8
    max_length: int = 4096
    device: str = "auto"  # auto, cuda, mps, cpu


@dataclass
class Config:
    """Main configuration container."""
    api: APIConfig = field(default_factory=APIConfig)
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file and environment variables.

    Priority (highest to lowest):
    1. Environment variables
    2. Config file (config.yaml or secrets.yaml)
    3. Default values

    Args:
        config_path: Optional path to config file

    Returns:
        Config object with all settings
    """
    config = Config()

    # Determine config directory
    if config_path:
        config_file = Path(config_path)
    else:
        config_dir = Path(__file__).parent
        config_file = config_dir / "secrets.yaml"
        if not config_file.exists():
            config_file = config_dir / "config.yaml"

    # Load from YAML if exists
    if config_file.exists():
        try:
            import yaml
            with open(config_file) as f:
                data = yaml.safe_load(f) or {}

            # API keys
            api_data = data.get("api", {})
            config.api.newsapi_key = api_data.get("newsapi_key")
            config.api.anthropic_key = api_data.get("anthropic_key")

            # Collection settings
            collection_data = data.get("collection", {})
            for key, value in collection_data.items():
                if hasattr(config.collection, key):
                    setattr(config.collection, key, value)

            # Model settings
            model_data = data.get("model", {})
            for key, value in model_data.items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)

        except ImportError:
            # PyYAML not installed, try JSON
            pass
        except Exception as e:
            print(f"Warning: Failed to load config from {config_file}: {e}")

    # Override with environment variables (highest priority)
    if os.environ.get("NEWSAPI_KEY"):
        config.api.newsapi_key = os.environ["NEWSAPI_KEY"]
    if os.environ.get("ANTHROPIC_API_KEY"):
        config.api.anthropic_key = os.environ["ANTHROPIC_API_KEY"]

    return config


def get_newsapi_key() -> Optional[str]:
    """Get NewsAPI key from config or environment."""
    config = load_config()
    return config.api.newsapi_key


def get_anthropic_key() -> Optional[str]:
    """Get Anthropic API key from config or environment."""
    config = load_config()
    return config.api.anthropic_key


# Global config instance (lazy loaded)
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global config instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
