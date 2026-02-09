"""
Tests for Groqbook models.
"""

import pytest
from groqbook.models import Book, GenerationStatistics


class TestGenerationStatistics:
    """Tests for GenerationStatistics class."""
    
    def test_init_defaults(self):
        """Test default initialization."""
        stats = GenerationStatistics()
        
        assert stats.input_time == 0.0
        assert stats.output_time == 0.0
        assert stats.input_tokens == 0
        assert stats.output_tokens == 0
        assert stats.total_time == 0.0
        assert stats.model_name == "meta-llama/llama-4-scout-17b-16e-instruct"
    
    def test_init_with_values(self):
        """Test initialization with custom values."""
        stats = GenerationStatistics(
            input_time=1.5,
            output_time=2.5,
            input_tokens=100,
            output_tokens=500,
            total_time=4.0,
            model_name="meta-llama/llama-4-maverick-17b-128e-instruct"
        )
        
        assert stats.input_time == 1.5
        assert stats.output_time == 2.5
        assert stats.input_tokens == 100
        assert stats.output_tokens == 500
        assert stats.total_time == 4.0
        assert stats.model_name == "meta-llama/llama-4-maverick-17b-128e-instruct"
    
    def test_get_input_speed(self):
        """Test input speed calculation."""
        stats = GenerationStatistics(input_time=2.0, input_tokens=100)
        assert stats.get_input_speed() == 50.0
        
        # Zero time should return 0
        stats_zero = GenerationStatistics(input_time=0, input_tokens=100)
        assert stats_zero.get_input_speed() == 0.0
    
    def test_get_output_speed(self):
        """Test output speed calculation."""
        stats = GenerationStatistics(output_time=5.0, output_tokens=500)
        assert stats.get_output_speed() == 100.0
        
        # Zero time should return 0
        stats_zero = GenerationStatistics(output_time=0, output_tokens=500)
        assert stats_zero.get_output_speed() == 0.0
    
    def test_add(self):
        """Test adding statistics together."""
        stats1 = GenerationStatistics(
            input_time=1.0,
            output_time=2.0,
            input_tokens=100,
            output_tokens=200,
            total_time=3.0
        )
        
        stats2 = GenerationStatistics(
            input_time=0.5,
            output_time=1.5,
            input_tokens=50,
            output_tokens=150,
            total_time=2.0
        )
        
        stats1.add(stats2)
        
        assert stats1.input_time == 1.5
        assert stats1.output_time == 3.5
        assert stats1.input_tokens == 150
        assert stats1.output_tokens == 350
        assert stats1.total_time == 5.0
    
    def test_get_total_tokens(self):
        """Test total tokens calculation."""
        stats = GenerationStatistics(input_tokens=100, output_tokens=500)
        assert stats.get_total_tokens() == 600
    
    def test_str_output(self):
        """Test string representation."""
        stats = GenerationStatistics(
            input_tokens=100,
            output_tokens=500,
            model_name="test-model"
        )
        output = str(stats)
        
        assert "test-model" in output
        assert "100" in output
        assert "500" in output
        assert "600" in output
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        stats = GenerationStatistics(
            input_time=1.0,
            output_time=2.0,
            input_tokens=100,
            output_tokens=200,
            total_time=3.0,
            model_name="test-model"
        )
        
        result = stats.to_dict()
        
        assert result["model_name"] == "test-model"
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 200
        assert result["total_tokens"] == 300


class TestBook:
    """Tests for Book class."""
    
    @pytest.fixture
    def simple_structure(self):
        """Simple book structure for testing."""
        return {
            "Chapter 1": "Introduction to the topic",
            "Chapter 2": "Deep dive into details",
            "Chapter 3": "Advanced concepts"
        }
    
    @pytest.fixture
    def nested_structure(self):
        """Nested book structure for testing."""
        return {
            "Part 1: Basics": {
                "Chapter 1": "Getting started",
                "Chapter 2": "Core concepts"
            },
            "Part 2: Advanced": {
                "Chapter 3": "Advanced topics",
                "Chapter 4": "Best practices"
            }
        }
    
    def test_init_simple(self, simple_structure):
        """Test initialization with simple structure."""
        book = Book("Test Book", simple_structure)
        
        assert book.book_title == "Test Book"
        assert len(book.get_section_list()) == 3
        assert all(title in book.contents for title in simple_structure.keys())
    
    def test_init_nested(self, nested_structure):
        """Test initialization with nested structure."""
        book = Book("Nested Book", nested_structure)
        
        # Should flatten all sections
        sections = book.get_section_list()
        assert "Part 1: Basics" in sections
        assert "Chapter 1" in sections
        assert "Part 2: Advanced" in sections
        assert len(sections) == 6
    
    def test_has_content_empty(self, simple_structure):
        """Test has_content returns False for empty sections."""
        book = Book("Test Book", simple_structure)
        
        assert not book.has_content("Chapter 1")
        assert not book.has_content("Chapter 2")
    
    def test_has_content_with_content(self, simple_structure):
        """Test has_content returns True after content is added."""
        book = Book("Test Book", simple_structure)
        book.update_content("Chapter 1", "Some content here")
        
        assert book.has_content("Chapter 1")
        assert not book.has_content("Chapter 2")
    
    def test_has_content_whitespace_only(self, simple_structure):
        """Test has_content returns False for whitespace-only content."""
        book = Book("Test Book", simple_structure)
        book.update_content("Chapter 1", "   \n\t  ")
        
        assert not book.has_content("Chapter 1")
    
    def test_update_content(self, simple_structure):
        """Test content streaming updates."""
        book = Book("Test Book", simple_structure)
        
        book.update_content("Chapter 1", "Hello ")
        book.update_content("Chapter 1", "World!")
        
        assert book.get_content("Chapter 1") == "Hello World!"
    
    def test_set_content(self, simple_structure):
        """Test setting full content."""
        book = Book("Test Book", simple_structure)
        
        book.set_content("Chapter 1", "Initial content")
        book.set_content("Chapter 1", "Replaced content")
        
        assert book.get_content("Chapter 1") == "Replaced content"
    
    def test_clear_section(self, simple_structure):
        """Test clearing section content."""
        book = Book("Test Book", simple_structure)
        book.update_content("Chapter 1", "Some content")
        
        assert book.has_content("Chapter 1")
        
        book.clear_section("Chapter 1")
        
        assert not book.has_content("Chapter 1")
        assert book.get_content("Chapter 1") == ""
    
    def test_get_section_description(self, simple_structure):
        """Test getting section description."""
        book = Book("Test Book", simple_structure)
        
        desc = book.get_section_description("Chapter 1")
        assert desc == "Introduction to the topic"
    
    def test_get_section_description_nested(self, nested_structure):
        """Test getting section description from nested structure."""
        book = Book("Test Book", nested_structure)
        
        desc = book.get_section_description("Chapter 3")
        assert desc == "Advanced topics"
    
    def test_get_section_description_not_found(self, simple_structure):
        """Test getting description for non-existent section."""
        book = Book("Test Book", simple_structure)
        
        desc = book.get_section_description("Non-existent Chapter")
        assert desc == ""
    
    def test_get_markdown_content(self, simple_structure):
        """Test markdown export."""
        book = Book("Test Book", simple_structure)
        book.update_content("Chapter 1", "This is chapter 1 content.")
        book.update_content("Chapter 2", "This is chapter 2 content.")
        
        markdown = book.get_markdown_content()
        
        assert "# Test Book" in markdown
        assert "## Chapter 1" in markdown or "# Chapter 1" in markdown
        assert "This is chapter 1 content." in markdown
        assert "This is chapter 2 content." in markdown
    
    def test_iter_sections(self, simple_structure):
        """Test iterating over sections."""
        book = Book("Test Book", simple_structure)
        book.update_content("Chapter 1", "Content 1")
        
        sections = list(book.iter_sections())
        
        assert len(sections) == 3
        assert sections[0][0] == "Chapter 1"  # title
        assert sections[0][1] == "Introduction to the topic"  # description
        assert sections[0][2] == "Content 1"  # content
    
    def test_get_statistics(self, simple_structure):
        """Test getting book statistics."""
        book = Book("Test Book", simple_structure)
        book.update_content("Chapter 1", "Word one two three four five")
        book.update_content("Chapter 2", "Six seven eight nine ten")
        
        stats = book.get_statistics()
        
        assert stats["title"] == "Test Book"
        assert stats["total_sections"] == 3
        assert stats["completed_sections"] == 2
        assert stats["total_words"] == 10
        assert stats["avg_words_per_section"] == 5.0
    
    def test_repr(self, simple_structure):
        """Test string representation."""
        book = Book("Test Book", simple_structure)
        book.update_content("Chapter 1", "Some content here")
        
        repr_str = repr(book)
        
        assert "Test Book" in repr_str
        assert "1/3" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
