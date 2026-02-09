"""
Tests for Groqbook configuration.
"""

import pytest
import json
import tempfile
from pathlib import Path

from groqbook.config import (
    Config,
    AVAILABLE_MODELS,
    BOOK_TYPES,
    WORD_COUNT_OPTIONS,
    MODEL_LARGE,
    MODEL_SMALL,
    get_config,
    set_config
)


class TestConfig:
    """Tests for Config class."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = Config()
        
        assert config.structure_model == MODEL_LARGE
        assert config.content_model == MODEL_SMALL
        assert config.default_book_type == "general"
        assert config.default_word_count == "medium"
        assert config.enable_context is True
        assert config.max_retries == 3
        assert config.min_topic_length == 10
        assert config.max_topic_length == 500
    
    def test_custom_values(self):
        """Test configuration with custom values."""
        config = Config(
            structure_model="mixtral-8x7b-32768",
            content_model="gemma2-9b-it",
            default_book_type="technical",
            max_retries=5
        )
        
        assert config.structure_model == "mixtral-8x7b-32768"
        assert config.content_model == "gemma2-9b-it"
        assert config.default_book_type == "technical"
        assert config.max_retries == 5
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = Config(groq_api_key="gsk_test_key")
        result = config.to_dict()
        
        assert isinstance(result, dict)
        assert result["groq_api_key"] == "***"  # Should be masked
        assert result["structure_model"] == MODEL_LARGE
    
    def test_from_file_json(self):
        """Test loading from JSON file."""
        config_data = {
            "structure_model": "mixtral-8x7b-32768",
            "max_retries": 10
        }
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            json.dump(config_data, f)
            f.flush()
            
            config = Config.from_file(f.name)
            
            assert config.structure_model == "mixtral-8x7b-32768"
            assert config.max_retries == 10
            # Other values should be defaults
            assert config.content_model == MODEL_SMALL
        
        Path(f.name).unlink()
    
    def test_from_file_not_found(self):
        """Test loading from non-existent file returns defaults."""
        config = Config.from_file("/non/existent/path.json")
        
        assert config.structure_model == MODEL_LARGE
        assert config.content_model == MODEL_SMALL
    
    def test_from_file_ignores_invalid_fields(self):
        """Test that invalid fields are ignored."""
        config_data = {
            "structure_model": "mixtral-8x7b-32768",
            "invalid_field": "should be ignored",
            "another_invalid": 12345
        }
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            json.dump(config_data, f)
            f.flush()
            
            config = Config.from_file(f.name)
            
            assert config.structure_model == "mixtral-8x7b-32768"
            assert not hasattr(config, 'invalid_field')
        
        Path(f.name).unlink()
    
    def test_save_json(self):
        """Test saving to JSON file."""
        config = Config(
            structure_model="mixtral-8x7b-32768",
            max_retries=7
        )
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            config.save(f.name)
            
            with open(f.name) as saved:
                data = json.load(saved)
            
            assert data["structure_model"] == "mixtral-8x7b-32768"
            assert data["max_retries"] == 7
            assert "groq_api_key" not in data  # Should not save API key
        
        Path(f.name).unlink()


class TestModels:
    """Tests for model configuration constants."""
    
    def test_available_models_structure(self):
        """Test AVAILABLE_MODELS has correct structure."""
        assert len(AVAILABLE_MODELS) >= 6
        
        for model_id, info in AVAILABLE_MODELS.items():
            assert "name" in info
            assert "description" in info
            assert "context_window" in info
            assert "speed" in info
            assert isinstance(info["context_window"], int)
    
    def test_default_models_exist(self):
        """Test default models are in available models."""
        assert MODEL_LARGE in AVAILABLE_MODELS
        assert MODEL_SMALL in AVAILABLE_MODELS


class TestBookTypes:
    """Tests for book type configuration constants."""
    
    def test_book_types_structure(self):
        """Test BOOK_TYPES has correct structure."""
        required_keys = ["name", "description", "structure_prompt", "content_prompt"]
        
        assert "general" in BOOK_TYPES
        
        for book_type, info in BOOK_TYPES.items():
            for key in required_keys:
                assert key in info, f"Missing {key} in {book_type}"
    
    def test_all_book_types(self):
        """Test all expected book types exist."""
        expected = ["general", "technical", "educational", "business", "creative", "academic"]
        
        for book_type in expected:
            assert book_type in BOOK_TYPES


class TestWordCountOptions:
    """Tests for word count configuration constants."""
    
    def test_word_count_structure(self):
        """Test WORD_COUNT_OPTIONS has correct structure."""
        for option, info in WORD_COUNT_OPTIONS.items():
            assert "name" in info
            assert "target" in info
            assert "description" in info
            assert isinstance(info["target"], int)
    
    def test_word_counts_ascending(self):
        """Test word counts are in ascending order."""
        targets = [
            WORD_COUNT_OPTIONS["short"]["target"],
            WORD_COUNT_OPTIONS["medium"]["target"],
            WORD_COUNT_OPTIONS["long"]["target"],
            WORD_COUNT_OPTIONS["extended"]["target"],
        ]
        
        assert targets == sorted(targets)


class TestGlobalConfig:
    """Tests for global config functions."""
    
    def test_get_config(self):
        """Test getting global config."""
        config = get_config()
        assert isinstance(config, Config)
    
    def test_set_config(self):
        """Test setting global config."""
        custom = Config(max_retries=99)
        set_config(custom)
        
        config = get_config()
        assert config.max_retries == 99
        
        # Reset to defaults
        set_config(Config())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
