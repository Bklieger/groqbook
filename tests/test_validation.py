"""
Tests for Groqbook validation functions.
"""

import pytest
from groqbook.validation import (
    sanitize_input,
    validate_api_key,
    validate_topic,
    validate_instructions,
    validate_seed_content,
    validate_book_type,
    validate_model,
    validate_word_count,
    validate_all_inputs
)
from groqbook.exceptions import ValidationError
from groqbook.config import Config


class TestSanitizeInput:
    """Tests for sanitize_input function."""
    
    def test_empty_string(self):
        """Test empty string input."""
        assert sanitize_input("") == ""
    
    def test_none_like(self):
        """Test None-like input (empty)."""
        assert sanitize_input("") == ""
    
    def test_normal_text(self):
        """Test normal text passes through."""
        text = "Hello, this is normal text!"
        assert sanitize_input(text) == text
    
    def test_removes_null_bytes(self):
        """Test null bytes are removed."""
        text = "Hello\x00World"
        assert sanitize_input(text) == "HelloWorld"
    
    def test_removes_control_chars(self):
        """Test control characters are removed."""
        text = "Hello\x01\x02\x03World"
        assert sanitize_input(text) == "HelloWorld"
    
    def test_preserves_newlines(self):
        """Test newlines are preserved."""
        text = "Line 1\nLine 2\nLine 3"
        assert sanitize_input(text) == text
    
    def test_preserves_tabs(self):
        """Test tabs are preserved."""
        text = "Column1\tColumn2\tColumn3"
        assert sanitize_input(text) == text
    
    def test_strips_whitespace(self):
        """Test leading/trailing whitespace is stripped."""
        text = "  Hello World  "
        assert sanitize_input(text) == "Hello World"
    
    def test_unicode_preserved(self):
        """Test unicode characters are preserved."""
        text = "Hello 世界 🌍 Привет"
        assert sanitize_input(text) == text


class TestValidateApiKey:
    """Tests for validate_api_key function."""
    
    def test_valid_key(self):
        """Test valid API key passes."""
        assert validate_api_key("gsk_abcdefghijklmnopqrstuvwxyz1234567890")
    
    def test_empty_key(self):
        """Test empty key fails."""
        assert not validate_api_key("")
    
    def test_wrong_prefix(self):
        """Test key without gsk_ prefix fails."""
        assert not validate_api_key("sk_abcdefghijklmnopqrstuvwxyz")
    
    def test_too_short(self):
        """Test too short key fails."""
        assert not validate_api_key("gsk_abc")
    
    def test_minimum_length(self):
        """Test minimum valid length."""
        assert validate_api_key("gsk_" + "a" * 20)
    
    def test_invalid_characters(self):
        """Test key with invalid characters fails."""
        assert not validate_api_key("gsk_abc!@#$%^&*()")


class TestValidateTopic:
    """Tests for validate_topic function."""
    
    @pytest.fixture
    def config(self):
        """Create config with known values."""
        return Config(min_topic_length=10, max_topic_length=100)
    
    def test_valid_topic(self, config):
        """Test valid topic passes."""
        result = validate_topic("This is a valid topic for testing", config)
        assert result == "This is a valid topic for testing"
    
    def test_too_short(self, config):
        """Test too short topic raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_topic("Short", config)
        assert "at least" in str(exc_info.value)
        assert exc_info.value.field == "topic"
    
    def test_too_long(self, config):
        """Test too long topic raises error."""
        long_topic = "A" * 150
        with pytest.raises(ValidationError) as exc_info:
            validate_topic(long_topic, config)
        assert "less than" in str(exc_info.value)
        assert exc_info.value.field == "topic"
    
    def test_sanitizes_input(self, config):
        """Test topic is sanitized."""
        result = validate_topic("  This topic has\x00null bytes  ", config)
        assert "\x00" not in result
        assert result == "This topic hasnull bytes"


class TestValidateInstructions:
    """Tests for validate_instructions function."""
    
    @pytest.fixture
    def config(self):
        """Create config with known values."""
        return Config(max_instructions_length=100)
    
    def test_valid_instructions(self, config):
        """Test valid instructions pass."""
        result = validate_instructions("Keep it simple", config)
        assert result == "Keep it simple"
    
    def test_empty_instructions(self, config):
        """Test empty instructions are allowed."""
        result = validate_instructions("", config)
        assert result == ""
    
    def test_too_long(self, config):
        """Test too long instructions raise error."""
        long_instructions = "A" * 200
        with pytest.raises(ValidationError) as exc_info:
            validate_instructions(long_instructions, config)
        assert "less than" in str(exc_info.value)


class TestValidateSeedContent:
    """Tests for validate_seed_content function."""
    
    @pytest.fixture
    def config(self):
        """Create config with known values."""
        return Config(max_seed_content_length=500)
    
    def test_valid_seed(self, config):
        """Test valid seed content passes."""
        seed = "This is sample text for style matching."
        result = validate_seed_content(seed, config)
        assert result == seed
    
    def test_empty_seed(self, config):
        """Test empty seed returns None."""
        result = validate_seed_content("", config)
        assert result is None
    
    def test_none_seed(self, config):
        """Test None seed returns None."""
        result = validate_seed_content(None, config)
        assert result is None
    
    def test_too_long(self, config):
        """Test too long seed raises error."""
        long_seed = "A" * 600
        with pytest.raises(ValidationError) as exc_info:
            validate_seed_content(long_seed, config)
        assert "less than" in str(exc_info.value)


class TestValidateBookType:
    """Tests for validate_book_type function."""
    
    def test_valid_types(self):
        """Test all valid book types."""
        valid_types = ["general", "technical", "educational", "business", "creative", "academic"]
        for book_type in valid_types:
            assert validate_book_type(book_type) == book_type
    
    def test_invalid_type(self):
        """Test invalid book type raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_book_type("fiction")
        assert "Invalid book type" in str(exc_info.value)
        assert exc_info.value.field == "book_type"


class TestValidateModel:
    """Tests for validate_model function."""
    
    def test_valid_models(self):
        """Test valid model names."""
        valid_models = ["meta-llama/llama-4-maverick-17b-128e-instruct", "meta-llama/llama-4-scout-17b-16e-instruct", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
        for model in valid_models:
            assert validate_model(model) == model
    
    def test_invalid_model(self):
        """Test invalid model raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_model("gpt-4")
        assert "Invalid model" in str(exc_info.value)


class TestValidateWordCount:
    """Tests for validate_word_count function."""
    
    def test_valid_options(self):
        """Test valid word count options."""
        valid_options = ["short", "medium", "long", "extended"]
        for option in valid_options:
            assert validate_word_count(option) == option
    
    def test_invalid_option(self):
        """Test invalid option raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_word_count("tiny")
        assert "Invalid word count" in str(exc_info.value)


class TestValidateAllInputs:
    """Tests for validate_all_inputs function."""
    
    def test_minimal_valid_inputs(self):
        """Test minimal valid inputs."""
        result = validate_all_inputs(
            topic="This is a valid topic for the book"
        )
        
        assert "topic" in result
        assert result["book_type"] == "general"
        assert result["word_count"] == "medium"
    
    def test_full_valid_inputs(self):
        """Test full valid inputs."""
        result = validate_all_inputs(
            topic="This is a valid topic for the book",
            instructions="Keep it technical",
            book_type="technical",
            model="llama3-70b-8192",
            word_count="long",
            seed_content="Sample text for style matching"
        )
        
        assert result["topic"] == "This is a valid topic for the book"
        assert result["book_type"] == "technical"
        assert result["model"] == "llama3-70b-8192"
        assert result["word_count"] == "long"
    
    def test_invalid_topic(self):
        """Test invalid topic raises error."""
        with pytest.raises(ValidationError):
            validate_all_inputs(topic="Short")
    
    def test_invalid_api_key(self):
        """Test invalid API key raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_all_inputs(
                topic="This is a valid topic for the book",
                api_key="invalid_key"
            )
        assert "API key" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
