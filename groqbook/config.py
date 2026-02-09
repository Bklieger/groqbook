"""
Configuration module for Groqbook.

Contains all configuration constants, model definitions, book types,
and application settings.
"""

from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Default models (updated to latest Groq models - Llama 4)
MODEL_LARGE = "meta-llama/llama-4-maverick-17b-128e-instruct"
MODEL_SMALL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Available models for user selection
AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
    "meta-llama/llama-4-maverick-17b-128e-instruct": {
        "name": "Llama 4 Maverick",
        "description": "Latest Llama 4, most capable MoE model (128 experts)",
        "context_window": 131072,
        "speed": "fast"
    },
    "meta-llama/llama-4-scout-17b-16e-instruct": {
        "name": "Llama 4 Scout",
        "description": "Latest Llama 4, efficient MoE model (16 experts)",
        "context_window": 131072,
        "speed": "very fast"
    },
    "openai/gpt-oss-120b": {
        "name": "GPT-OSS 120B",
        "description": "OpenAI's 120B open-weight model, high quality",
        "context_window": 131072,
        "speed": "moderate"
    },
    "openai/gpt-oss-20b": {
        "name": "GPT-OSS 20B",
        "description": "OpenAI's 20B open-weight model, very fast",
        "context_window": 131072,
        "speed": "very fast"
    },
    "llama-3.3-70b-versatile": {
        "name": "Llama 3.3 70B",
        "description": "Proven 70B model, great for complex topics",
        "context_window": 131072,
        "speed": "moderate"
    },
    "llama-3.1-8b-instant": {
        "name": "Llama 3.1 8B",
        "description": "Fast and efficient for most content",
        "context_window": 131072,
        "speed": "very fast"
    }
}



# ============================================================================
# BOOK TYPE TEMPLATES
# ============================================================================

BOOK_TYPES: Dict[str, Dict[str, str]] = {
    "general": {
        "name": "📚 General Non-Fiction",
        "description": "Standard informative book structure",
        "structure_prompt": "Write a comprehensive structure for a long (>300 page) book",
        "content_prompt": "Generate a long, comprehensive, structured chapter",
        "style_guide": ""
    },
    "technical": {
        "name": "💻 Technical/Programming",
        "description": "Code examples, detailed explanations",
        "structure_prompt": "Write a comprehensive technical book structure with practical examples and code sections",
        "content_prompt": "Generate a detailed technical chapter with clear explanations, code examples, and best practices",
        "style_guide": "Include code snippets where relevant. Use clear headings. Explain concepts step-by-step."
    },
    "educational": {
        "name": "🎓 Educational/Textbook",
        "description": "Learning objectives, exercises, summaries",
        "structure_prompt": "Write an educational textbook structure with learning objectives and review sections",
        "content_prompt": "Generate an educational chapter with learning objectives, clear explanations, examples, and review questions",
        "style_guide": "Start with learning objectives. Include examples and exercises. End with a summary."
    },
    "business": {
        "name": "💼 Business/Self-Help",
        "description": "Actionable insights, case studies",
        "structure_prompt": "Write a business book structure with actionable insights and case studies",
        "content_prompt": "Generate a business-focused chapter with practical advice, case studies, and actionable takeaways",
        "style_guide": "Include real-world examples. Provide actionable steps. Use bullet points for key takeaways."
    },
    "creative": {
        "name": "✨ Creative/Storytelling",
        "description": "Narrative elements, engaging style",
        "structure_prompt": "Write an engaging book structure with narrative elements and storytelling approach",
        "content_prompt": "Generate an engaging chapter with narrative flow, storytelling elements, and vivid descriptions",
        "style_guide": "Use engaging narrative voice. Include anecdotes and stories. Make it compelling to read."
    },
    "academic": {
        "name": "📖 Academic/Research",
        "description": "Formal style, citations, thorough analysis",
        "structure_prompt": "Write an academic book structure with thorough analysis and formal organization",
        "content_prompt": "Generate an academic chapter with formal tone, thorough analysis, and scholarly approach",
        "style_guide": "Use formal academic tone. Include thorough analysis. Reference concepts and theories."
    }
}


# ============================================================================
# WORD COUNT OPTIONS
# ============================================================================

WORD_COUNT_OPTIONS: Dict[str, Dict[str, Any]] = {
    "short": {"name": "Short (~500 words)", "target": 500, "description": "Concise sections"},
    "medium": {"name": "Medium (~1000 words)", "target": 1000, "description": "Standard length"},
    "long": {"name": "Long (~2000 words)", "target": 2000, "description": "Detailed sections"},
    "extended": {"name": "Extended (~3000+ words)", "target": 3000, "description": "Very comprehensive"}
}


# ============================================================================
# APPLICATION CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """
    Application configuration with sensible defaults.
    Can be loaded from environment variables or a config file.
    """
    # API Configuration
    groq_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    
    # Model Configuration
    structure_model: str = MODEL_LARGE
    content_model: str = MODEL_SMALL
    
    # Generation Settings
    default_book_type: str = "general"
    default_word_count: str = "medium"
    enable_context: bool = True
    
    # Retry Configuration
    max_retries: int = 3
    retry_delay_base: float = 2.0
    
    # Input Validation
    min_topic_length: int = 10
    max_topic_length: int = 1500
    max_instructions_length: int = 2000
    max_seed_content_length: int = 5000
    
    # Export Settings
    pdf_font_family: str = "Georgia, serif"
    pdf_page_size: str = "A4"
    
    # Logging
    log_level: str = "INFO"
    
    @classmethod
    def from_file(cls, path: str | Path) -> "Config":
        """
        Load configuration from a JSON or YAML file.
        
        Args:
            path: Path to the configuration file
            
        Returns:
            Config instance with loaded values
        """
        path = Path(path)
        
        if not path.exists():
            logger.warning(f"Config file not found: {path}. Using defaults.")
            return cls()
        
        try:
            with open(path) as f:
                if path.suffix in ('.yaml', '.yml'):
                    try:
                        import yaml
                        data = yaml.safe_load(f)
                    except ImportError:
                        logger.warning("PyYAML not installed. Cannot load YAML config.")
                        return cls()
                else:
                    data = json.load(f)
            
            # Filter to only valid fields
            valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
            filtered_data = {k: v for k, v in data.items() if k in valid_fields}
            
            return cls(**filtered_data)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return cls()
    
    @classmethod
    def from_env(cls) -> "Config":
        """
        Load configuration from environment variables.
        Environment variables should be prefixed with GROQBOOK_.
        
        Returns:
            Config instance with values from environment
        """
        config = cls()
        
        env_mappings = {
            "GROQBOOK_API_KEY": "groq_api_key",
            "GROQBOOK_STRUCTURE_MODEL": "structure_model",
            "GROQBOOK_CONTENT_MODEL": "content_model",
            "GROQBOOK_BOOK_TYPE": "default_book_type",
            "GROQBOOK_WORD_COUNT": "default_word_count",
            "GROQBOOK_MAX_RETRIES": "max_retries",
            "GROQBOOK_LOG_LEVEL": "log_level",
        }
        
        for env_var, attr in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Handle type conversion
                current_value = getattr(config, attr)
                if isinstance(current_value, bool):
                    value = value.lower() in ('true', '1', 'yes')
                elif isinstance(current_value, int):
                    value = int(value)
                elif isinstance(current_value, float):
                    value = float(value)
                setattr(config, attr, value)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "groq_api_key": "***" if self.groq_api_key else None,  # Mask API key
            "structure_model": self.structure_model,
            "content_model": self.content_model,
            "default_book_type": self.default_book_type,
            "default_word_count": self.default_word_count,
            "enable_context": self.enable_context,
            "max_retries": self.max_retries,
            "retry_delay_base": self.retry_delay_base,
            "min_topic_length": self.min_topic_length,
            "max_topic_length": self.max_topic_length,
            "max_instructions_length": self.max_instructions_length,
            "max_seed_content_length": self.max_seed_content_length,
        }
    
    def save(self, path: str | Path) -> None:
        """
        Save configuration to a file.
        
        Args:
            path: Path to save the configuration
        """
        path = Path(path)
        data = self.to_dict()
        
        # Don't save masked API key
        if "groq_api_key" in data:
            del data["groq_api_key"]
        
        with open(path, 'w') as f:
            if path.suffix in ('.yaml', '.yml'):
                try:
                    import yaml
                    yaml.dump(data, f, default_flow_style=False)
                except ImportError:
                    logger.warning("PyYAML not installed. Saving as JSON.")
                    json.dump(data, f, indent=2)
            else:
                json.dump(data, f, indent=2)
        
        logger.info(f"Configuration saved to {path}")


# Global default config instance
_default_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _default_config
    if _default_config is None:
        _default_config = Config.from_env()
    return _default_config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _default_config
    _default_config = config
