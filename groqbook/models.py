"""
Data models for Groqbook.

Contains the Book and GenerationStatistics classes with full type hints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Iterator

logger = logging.getLogger(__name__)


@dataclass
class GenerationStatistics:
    """
    Tracks statistics from LLM generation calls.
    
    Aggregates time and token counts across multiple API calls for
    comprehensive performance monitoring.
    """
    input_time: float = 0.0
    output_time: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_time: float = 0.0
    model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    
    def get_input_speed(self) -> float:
        """
        Calculate tokens per second for input processing.
        
        Returns:
            Input tokens per second, or 0 if no time recorded
        """
        if self.input_time > 0:
            return self.input_tokens / self.input_time
        return 0.0
    
    def get_output_speed(self) -> float:
        """
        Calculate tokens per second for output generation.
        
        Returns:
            Output tokens per second, or 0 if no time recorded
        """
        if self.output_time > 0:
            return self.output_tokens / self.output_time
        return 0.0
    
    def add(self, other: "GenerationStatistics") -> None:
        """
        Add statistics from another generation call.
        
        Args:
            other: GenerationStatistics to add to this instance
        """
        self.input_time += other.input_time
        self.output_time += other.output_time
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_time += other.total_time
    
    def get_total_tokens(self) -> int:
        """Get total tokens (input + output)."""
        return self.input_tokens + self.output_tokens
    
    def __str__(self) -> str:
        return (
            f"## 📊 Generation Statistics\n\n"
            f"- **Model**: `{self.model_name}`\n"
            f"- **Input tokens**: {self.input_tokens:,}\n"
            f"- **Output tokens**: {self.output_tokens:,}\n"
            f"- **Total tokens**: {self.get_total_tokens():,}\n"
            f"- **Input speed**: {self.get_input_speed():.1f} tokens/sec\n"
            f"- **Output speed**: {self.get_output_speed():.1f} tokens/sec\n"
            f"- **Total time**: {self.total_time:.2f}s"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "input_time": self.input_time,
            "output_time": self.output_time,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_time": self.total_time,
            "input_speed": self.get_input_speed(),
            "output_speed": self.get_output_speed(),
            "total_tokens": self.get_total_tokens(),
        }


class Book:
    """
    Represents a book with its structure and content.
    
    Manages the hierarchical structure of sections, their content,
    and provides methods for display and export.
    """
    
    def __init__(self, book_title: str, structure: Dict[str, Any]):
        """
        Initialize a new Book.
        
        Args:
            book_title: The title of the book
            structure: Nested dictionary of section titles and descriptions
        """
        self.book_title = book_title
        self.structure = structure
        self._section_titles = self._flatten_structure(structure)
        self.contents: Dict[str, str] = {title: "" for title in self._section_titles}
        
        # Placeholders are only used when running in Streamlit
        self._placeholders: Dict[str, Any] = {}
        self._streamlit_initialized = False
        
        logger.info(f"Book initialized: '{book_title}' with {len(self._section_titles)} sections")
    
    def _flatten_structure(self, structure: Dict[str, Any]) -> List[str]:
        """
        Flatten nested structure into a list of section titles.
        
        Args:
            structure: Nested dictionary structure
            
        Returns:
            Flat list of all section titles
        """
        sections: List[str] = []
        for title, content in structure.items():
            sections.append(title)
            if isinstance(content, dict):
                sections.extend(self._flatten_structure(content))
        return sections
    
    def init_streamlit(self) -> None:
        """Initialize Streamlit placeholders for live content display."""
        try:
            import streamlit as st
            self._placeholders = {title: st.empty() for title in self._section_titles}
            self._streamlit_initialized = True
            
            # Display initial structure
            st.markdown(f"# {self.book_title}")
            st.markdown("## Generating the following:")
            toc_columns = st.columns(4)
            self._display_toc_streamlit(self.structure, toc_columns)
            st.markdown("---")
        except ImportError:
            logger.warning("Streamlit not available. Running in headless mode.")
    
    def _display_toc_streamlit(
        self, 
        structure: Dict[str, Any], 
        columns: List[Any], 
        level: int = 1, 
        col_index: int = 0
    ) -> int:
        """Display table of contents in Streamlit columns."""
        import streamlit as st
        
        for title, content in structure.items():
            with columns[col_index % len(columns)]:
                st.markdown(f"{' ' * (level-1) * 2}- {title}")
            col_index += 1
            if isinstance(content, dict):
                col_index = self._display_toc_streamlit(content, columns, level + 1, col_index)
        return col_index
    
    def get_section_list(self) -> List[str]:
        """
        Get a list of all section titles.
        
        Returns:
            List of section title strings in order
        """
        return list(self._section_titles)
    
    def has_content(self, title: str) -> bool:
        """
        Check if a section has generated content.
        
        Args:
            title: Section title to check
            
        Returns:
            True if section has non-empty content
        """
        return title in self.contents and bool(self.contents[title].strip())
    
    def get_content(self, title: str) -> str:
        """
        Get the content of a specific section.
        
        Args:
            title: Section title
            
        Returns:
            Section content or empty string if not found
        """
        return self.contents.get(title, "")
    
    def clear_section(self, title: str) -> None:
        """
        Clear the content of a specific section.
        
        Args:
            title: Title of the section to clear
        """
        if title in self.contents:
            self.contents[title] = ""
            if self._streamlit_initialized and title in self._placeholders:
                self._placeholders[title].empty()
            logger.info(f"Cleared section: {title}")
    
    def update_content(self, title: str, new_content: str) -> None:
        """
        Append content to a section (used during streaming).
        
        Args:
            title: Section title to update
            new_content: Content to append
        """
        if title not in self.contents:
            logger.warning(f"Unknown section: {title}")
            return
            
        self.contents[title] += new_content
        
        # Update Streamlit display if available
        if self._streamlit_initialized and title in self._placeholders:
            if self.contents[title].strip():
                self._placeholders[title].markdown(
                    f"## {title}\n{self.contents[title]}"
                )
    
    def set_content(self, title: str, content: str) -> None:
        """
        Set the full content of a section (replaces existing).
        
        Args:
            title: Section title
            content: Full content to set
        """
        if title in self.contents:
            self.contents[title] = content
            if self._streamlit_initialized and title in self._placeholders:
                self._placeholders[title].markdown(f"## {title}\n{content}")
    
    def get_section_description(self, title: str, structure: Optional[Dict[str, Any]] = None) -> str:
        """
        Get the description of a section from the structure.
        
        Args:
            title: Section title to find
            structure: Optional structure dict (uses self.structure if None)
            
        Returns:
            Section description or empty string if not found
        """
        if structure is None:
            structure = self.structure
            
        for sec_title, content in structure.items():
            if sec_title == title:
                if isinstance(content, str):
                    return content
                return ""
            if isinstance(content, dict):
                result = self.get_section_description(title, content)
                if result:
                    return result
        return ""
    
    def display_structure(self, structure: Optional[Dict[str, Any]] = None, level: int = 1) -> None:
        """
        Display the book structure with content in Streamlit.
        
        Args:
            structure: Optional structure dict to display
            level: Current heading level
        """
        if not self._streamlit_initialized:
            return
            
        import streamlit as st
        
        if structure is None:
            structure = self.structure
            
        for title, content in structure.items():
            if self.contents.get(title, "").strip():
                st.markdown(f"{'#' * level} {title}")
                self._placeholders[title].markdown(self.contents[title])
            if isinstance(content, dict):
                self.display_structure(content, level + 1)
    
    def get_markdown_content(
        self, 
        structure: Optional[Dict[str, Any]] = None, 
        level: int = 1
    ) -> str:
        """
        Get the complete book content as markdown.
        
        Args:
            structure: Optional structure to use
            level: Current heading level
            
        Returns:
            Complete book content in markdown format
        """
        if structure is None:
            structure = self.structure
        
        if level == 1:
            markdown_content = f"# {self.book_title}\n\n"
        else:
            markdown_content = ""
        
        for title, content in structure.items():
            section_content = self.contents.get(title, "")
            if section_content.strip():
                markdown_content += f"{'#' * level} {title}\n{section_content}\n\n"
            if isinstance(content, dict):
                markdown_content += self.get_markdown_content(content, level + 1)
        
        return markdown_content
    
    def iter_sections(self) -> Iterator[tuple[str, str, str]]:
        """
        Iterate over all sections with their content.
        
        Yields:
            Tuples of (title, description, content)
        """
        for title in self._section_titles:
            description = self.get_section_description(title)
            content = self.contents.get(title, "")
            yield title, description, content
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get book statistics.
        
        Returns:
            Dictionary with section counts and word counts
        """
        total_sections = len(self._section_titles)
        completed_sections = sum(1 for t in self._section_titles if self.has_content(t))
        
        total_words = sum(
            len(self.contents.get(t, "").split()) 
            for t in self._section_titles
        )
        
        return {
            "title": self.book_title,
            "total_sections": total_sections,
            "completed_sections": completed_sections,
            "completion_percent": (completed_sections / total_sections * 100) if total_sections > 0 else 0,
            "total_words": total_words,
            "avg_words_per_section": total_words / completed_sections if completed_sections > 0 else 0,
        }
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"Book(title='{self.book_title}', "
            f"sections={stats['completed_sections']}/{stats['total_sections']}, "
            f"words={stats['total_words']:,})"
        )
