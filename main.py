import streamlit as st
from groq import Groq
import json
import os
import time
import logging
import uuid
from datetime import datetime
from functools import wraps
from io import BytesIO
from markdown import markdown
from weasyprint import HTML, CSS
from dotenv import load_dotenv

# EPUB support (optional - graceful fallback if not installed)
try:
    from ebooklib import epub
    EPUB_AVAILABLE = True
except ImportError:
    EPUB_AVAILABLE = False
    logging.warning("ebooklib not installed. EPUB export will be disabled.")

# load .env file to environment
load_dotenv()

# PAGE CONFIG MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Groqbook - AI Book Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)

# Model Configuration (updated to latest Groq models - Llama 4)
MODEL_LARGE = "meta-llama/llama-4-maverick-17b-128e-instruct"
MODEL_SMALL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Available models for user selection
AVAILABLE_MODELS = {
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


# Book type templates
BOOK_TYPES = {
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

# Word count options per section
WORD_COUNT_OPTIONS = {
    "short": {"name": "Short (~500 words)", "target": 500, "description": "Concise sections"},
    "medium": {"name": "Medium (~1000 words)", "target": 1000, "description": "Standard length"},
    "long": {"name": "Long (~2000 words)", "target": 2000, "description": "Detailed sections"},
    "extended": {"name": "Extended (~3000+ words)", "target": 3000, "description": "Very comprehensive"}
}

# Retry Configuration
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2  # Base delay in seconds (exponential backoff)

# Input Validation
MIN_TOPIC_LENGTH = 10
MAX_TOPIC_LENGTH = 500
MAX_INSTRUCTIONS_LENGTH = 2000
MAX_SEED_CONTENT_LENGTH = 5000


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class GroqbookError(Exception):
    """Base exception for Groqbook errors."""
    pass


class APIError(GroqbookError):
    """Error related to API calls."""
    pass


class RateLimitError(APIError):
    """Rate limit exceeded error."""
    pass


class ValidationError(GroqbookError):
    """Input validation error."""
    pass


class GenerationError(GroqbookError):
    """Error during content generation."""
    pass


# ============================================================================
# RETRY DECORATOR
# ============================================================================

def retry_on_rate_limit(max_retries: int = MAX_RETRIES, base_delay: float = RETRY_DELAY_BASE):
    """
    Decorator that retries a function on rate limit errors with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds (will be multiplied exponentially)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e).lower()
                    # Check for rate limit errors
                    if 'rate_limit' in error_str or 'rate limit' in error_str or '429' in error_str:
                        last_exception = e
                        if attempt < max_retries:
                            delay = base_delay * (2 ** attempt)
                            logger.warning(f"Rate limit hit. Retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                            time.sleep(delay)
                            continue
                        else:
                            logger.error(f"Rate limit exceeded after {max_retries} retries")
                            raise RateLimitError(f"API rate limit exceeded. Please wait a moment and try again.") from e
                    else:
                        # Re-raise non-rate-limit errors immediately
                        raise
            raise last_exception
        return wrapper
    return decorator


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if "api_key" not in st.session_state:
    st.session_state.api_key = GROQ_API_KEY

if "groq" not in st.session_state:
    if GROQ_API_KEY:
        st.session_state.groq = Groq()


class GenerationStatistics:
    def __init__(
        self,
        input_time=0,
        output_time=0,
        input_tokens=0,
        output_tokens=0,
        total_time=0,
        model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    ):
        self.input_time = input_time
        self.output_time = output_time
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_time = (
            total_time  # Sum of queue, prompt (input), and completion (output) times
        )
        self.model_name = model_name

    def get_input_speed(self):
        """
        Tokens per second calculation for input
        """
        if self.input_time != 0:
            return self.input_tokens / self.input_time
        else:
            return 0

    def get_output_speed(self):
        """
        Tokens per second calculation for output
        """
        if self.output_time != 0:
            return self.output_tokens / self.output_time
        else:
            return 0

    def add(self, other):
        """
        Add statistics from another GenerationStatistics object to this one.
        """
        if not isinstance(other, GenerationStatistics):
            raise TypeError("Can only add GenerationStatistics objects")

        self.input_time += other.input_time
        self.output_time += other.output_time
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_time += other.total_time

    def __str__(self):
        return (
            f"\n## {self.get_output_speed():.2f} T/s ⚡\nRound trip time: {self.total_time:.2f}s  Model: {self.model_name}\n\n"
            f"| Metric          | Input          | Output          | Total          |\n"
            f"|-----------------|----------------|-----------------|----------------|\n"
            f"| Speed (T/s)     | {self.get_input_speed():.2f}            | {self.get_output_speed():.2f}            | {(self.input_tokens + self.output_tokens) / self.total_time if self.total_time != 0 else 0:.2f}            |\n"
            f"| Tokens          | {self.input_tokens}            | {self.output_tokens}            | {self.input_tokens + self.output_tokens}            |\n"
            f"| Inference Time (s) | {self.input_time:.2f}            | {self.output_time:.2f}            | {self.total_time:.2f}            |"
        )


class Book:
    def __init__(self, book_title, structure):
        self.book_title = book_title
        self.structure = structure
        self.contents = {title: "" for title in self.flatten_structure(structure)}
        self.placeholders = {title: st.empty() for title in self.flatten_structure(structure)}
        st.markdown(f"# {self.book_title}")
        st.markdown("## Generating the following:")
        toc_columns = st.columns(4)
        self.display_toc(self.structure, toc_columns)
        st.markdown("---")

    def flatten_structure(self, structure):
        sections = []
        for title, content in structure.items():
            sections.append(title)
            if isinstance(content, dict):
                sections.extend(self.flatten_structure(content))
        return sections
    
    def get_section_list(self) -> list:
        """
        Get a list of all section titles.
        
        Returns:
            List of section title strings
        """
        return list(self.contents.keys())
    
    def has_content(self, title: str) -> bool:
        """
        Check if a section has generated content.
        
        Args:
            title: Section title to check
            
        Returns:
            True if section has content, False otherwise
        """
        return title in self.contents and bool(self.contents[title].strip())
    
    def clear_section(self, title: str):
        """
        Clear the content of a specific section.
        
        Args:
            title: Title of the section to clear
        """
        if title in self.contents:
            self.contents[title] = ""
            if title in self.placeholders:
                self.placeholders[title].empty()
            logger.info(f"Cleared section: {title}")

    def update_content(self, title, new_content):
        try:
            self.contents[title] += new_content
            self.display_content(title)
        except TypeError as e:
            pass

    def display_content(self, title):
        if self.contents[title].strip():
            self.placeholders[title].markdown(f"## {title}\n{self.contents[title]}")

    def display_structure(self, structure=None, level=1):
        if structure is None:
            structure = self.structure
            
        for title, content in structure.items():
            if self.contents[title].strip():  # Only display title if there is content
                st.markdown(f"{'#' * level} {title}")
                self.placeholders[title].markdown(self.contents[title])
            if isinstance(content, dict):
                self.display_structure(content, level + 1)

    def display_toc(self, structure, columns, level=1, col_index=0):
        for title, content in structure.items():
            with columns[col_index % len(columns)]:
                st.markdown(f"{' ' * (level-1) * 2}- {title}")
            col_index += 1
            if isinstance(content, dict):
                col_index = self.display_toc(content, columns, level + 1, col_index)
        return col_index

    def get_markdown_content(self, structure=None, level=1):
        """
        Returns the markdown styled pure string with the contents.
        """
        if structure is None:
            structure = self.structure
        
        if level==1:
            markdown_content = f"# {self.book_title}\n\n"
            
        else:
            markdown_content = ""
        
        for title, content in structure.items():
            if self.contents[title].strip():  # Only include title if there is content
                markdown_content += f"{'#' * level} {title}\n{self.contents[title]}\n\n"
            if isinstance(content, dict):
                markdown_content += self.get_markdown_content(content, level + 1)
        return markdown_content
    
    def get_section_description(self, title: str, structure=None) -> str:
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


def create_markdown_file(content: str) -> BytesIO:
    """
    Create a Markdown file from the provided content.
    """
    markdown_file = BytesIO()
    markdown_file.write(content.encode("utf-8"))
    markdown_file.seek(0)
    return markdown_file


def create_pdf_file(content: str, book_title: str = "Untitled Book") -> BytesIO:
    """
    Create a professionally styled PDF file from the provided Markdown content.
    Includes cover page, table of contents, page numbers, and enhanced styling.
    
    Args:
        content: Markdown content of the book
        book_title: Title of the book for the cover page
        
    Returns:
        BytesIO buffer containing the PDF
    """
    html_content = markdown(content, extensions=["extra", "codehilite", "toc"])
    
    # Generate current date for cover
    current_date = datetime.now().strftime("%B %Y")
    
    styled_html = f"""
    <html>
        <head>
            <style>
                @page {{
                    margin: 2.5cm 2cm;
                    @bottom-center {{
                        content: counter(page);
                        font-size: 10pt;
                        color: #666;
                    }}
                }}
                @page cover {{
                    margin: 0;
                    @bottom-center {{
                        content: none;
                    }}
                }}
                body {{
                    font-family: 'Georgia', 'Times New Roman', serif;
                    line-height: 1.8;
                    font-size: 11pt;
                    color: #333;
                }}
                /* Cover Page Styles */
                .cover-page {{
                    page: cover;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    text-align: center;
                    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                    color: white;
                    page-break-after: always;
                }}
                .cover-title {{
                    font-size: 36pt;
                    font-weight: bold;
                    margin-bottom: 30px;
                    line-height: 1.2;
                    padding: 0 40px;
                }}
                .cover-subtitle {{
                    font-size: 14pt;
                    color: #aaa;
                    margin-top: 20px;
                }}
                .cover-date {{
                    font-size: 12pt;
                    color: #888;
                    margin-top: 40px;
                }}
                .cover-badge {{
                    margin-top: 60px;
                    padding: 10px 30px;
                    border: 2px solid #4a90e2;
                    border-radius: 30px;
                    color: #4a90e2;
                    font-size: 10pt;
                    text-transform: uppercase;
                    letter-spacing: 2px;
                }}
                /* Content Styles */
                h1 {{
                    color: #1a1a2e;
                    font-size: 24pt;
                    margin-top: 2em;
                    margin-bottom: 0.5em;
                    border-bottom: 3px solid #4a90e2;
                    padding-bottom: 10px;
                    page-break-before: always;
                }}
                h1:first-of-type {{
                    page-break-before: avoid;
                }}
                h2 {{
                    color: #16213e;
                    font-size: 18pt;
                    margin-top: 1.5em;
                    margin-bottom: 0.5em;
                }}
                h3 {{
                    color: #0f3460;
                    font-size: 14pt;
                    margin-top: 1.2em;
                    margin-bottom: 0.4em;
                }}
                h4, h5, h6 {{
                    color: #333;
                    margin-top: 1em;
                    margin-bottom: 0.3em;
                }}
                p {{
                    margin-bottom: 0.8em;
                    text-align: justify;
                }}
                code {{
                    background-color: #f8f9fa;
                    padding: 2px 6px;
                    border-radius: 4px;
                    font-family: 'Courier New', monospace;
                    font-size: 0.9em;
                    border: 1px solid #e9ecef;
                }}
                pre {{
                    background-color: #2d2d2d;
                    color: #f8f8f2;
                    padding: 1.2em;
                    border-radius: 8px;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                    font-size: 0.85em;
                    margin: 1em 0;
                    overflow-x: auto;
                }}
                pre code {{
                    background: none;
                    border: none;
                    padding: 0;
                    color: inherit;
                }}
                blockquote {{
                    border-left: 4px solid #4a90e2;
                    padding-left: 1.5em;
                    margin: 1.5em 0;
                    font-style: italic;
                    color: #555;
                    background-color: #f8f9fa;
                    padding: 1em 1em 1em 1.5em;
                    border-radius: 0 8px 8px 0;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 1.5em 0;
                    font-size: 0.95em;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 12px 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #1a1a2e;
                    color: white;
                    font-weight: bold;
                }}
                tr:nth-child(even) {{
                    background-color: #f8f9fa;
                }}
                ul, ol {{
                    margin: 1em 0;
                    padding-left: 2em;
                }}
                li {{
                    margin-bottom: 0.5em;
                }}
                a {{
                    color: #4a90e2;
                    text-decoration: none;
                }}
                strong {{
                    color: #1a1a2e;
                }}
                hr {{
                    border: none;
                    border-top: 2px solid #e9ecef;
                    margin: 2em 0;
                }}
            </style>
        </head>
        <body>
            <!-- Cover Page -->
            <div class="cover-page">
                <div class="cover-title">{book_title}</div>
                <div class="cover-subtitle">Generated with Groqbook</div>
                <div class="cover-date">{current_date}</div>
                <div class="cover-badge">AI-Powered Book Generation</div>
            </div>
            
            <!-- Book Content -->
            {html_content}
        </body>
    </html>
    """

    pdf_buffer = BytesIO()
    HTML(string=styled_html).write_pdf(pdf_buffer)
    pdf_buffer.seek(0)
    
    logger.info(f"PDF created for: {book_title}")
    return pdf_buffer


def create_epub_file(content: str, book_title: str = "Untitled Book", structure: dict = None) -> BytesIO:
    """
    Create an EPUB file from the provided Markdown content.
    
    Args:
        content: Markdown content of the book
        book_title: Title of the book
        structure: Optional book structure dict for chapter organization
        
    Returns:
        BytesIO buffer containing the EPUB file
    """
    if not EPUB_AVAILABLE:
        raise ImportError("ebooklib is not installed. Please install it with: pip install ebooklib")
    
    # Create EPUB book
    book = epub.EpubBook()
    
    # Set metadata
    book_id = str(uuid.uuid4())
    book.set_identifier(book_id)
    book.set_title(book_title)
    book.set_language('en')
    book.add_author('Groqbook AI')
    book.add_metadata('DC', 'description', f'Generated by Groqbook on {datetime.now().strftime("%Y-%m-%d")}')
    
    # Create CSS styles
    style = '''
    body {
        font-family: Georgia, serif;
        line-height: 1.6;
        margin: 5%;
    }
    h1, h2, h3 {
        color: #333366;
        margin-top: 1em;
    }
    h1 {
        border-bottom: 2px solid #4a90e2;
        padding-bottom: 0.3em;
    }
    code {
        background-color: #f4f4f4;
        padding: 2px 4px;
        border-radius: 3px;
        font-family: monospace;
    }
    pre {
        background-color: #f4f4f4;
        padding: 1em;
        border-radius: 5px;
        overflow-x: auto;
    }
    blockquote {
        border-left: 3px solid #4a90e2;
        padding-left: 1em;
        margin-left: 0;
        font-style: italic;
        color: #555;
    }
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 1em 0;
    }
    th, td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }
    th {
        background-color: #f2f2f2;
    }
    '''
    
    nav_css = epub.EpubItem(
        uid="style_nav",
        file_name="style/nav.css",
        media_type="text/css",
        content=style
    )
    book.add_item(nav_css)
    
    # Convert markdown content to HTML
    html_content = markdown(content, extensions=["extra", "codehilite"])
    
    # Create cover chapter
    cover_html = f'''
    <html>
    <head><title>{book_title}</title></head>
    <body>
        <div style="text-align: center; margin-top: 40%;">
            <h1 style="font-size: 2em; color: #333366;">{book_title}</h1>
            <p style="color: #666; margin-top: 2em;">Generated with Groqbook</p>
            <p style="color: #888; margin-top: 1em;">{datetime.now().strftime("%B %Y")}</p>
        </div>
    </body>
    </html>
    '''
    
    cover = epub.EpubHtml(title='Cover', file_name='cover.xhtml', lang='en')
    cover.content = cover_html
    book.add_item(cover)
    
    # Create main content chapter
    main_content = epub.EpubHtml(title='Content', file_name='content.xhtml', lang='en')
    main_content.content = f'''
    <html>
    <head>
        <title>{book_title}</title>
        <link rel="stylesheet" type="text/css" href="style/nav.css"/>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    '''
    main_content.add_item(nav_css)
    book.add_item(main_content)
    
    # Define Table of Contents and spine
    book.toc = [
        epub.Link('cover.xhtml', 'Cover', 'cover'),
        epub.Link('content.xhtml', 'Content', 'content')
    ]
    
    # Add navigation files
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    
    # Define spine
    book.spine = ['nav', cover, main_content]
    
    # Write to buffer
    epub_buffer = BytesIO()
    epub.write_epub(epub_buffer, book)
    epub_buffer.seek(0)
    
    logger.info(f"EPUB created for: {book_title}")
    return epub_buffer

@retry_on_rate_limit()
def generate_book_title(prompt: str) -> str:
    """
    Generate a book title using AI.
    
    Args:
        prompt: The book topic to generate a title for
        
    Returns:
        Generated book title string
        
    Raises:
        APIError: If the API call fails
        RateLimitError: If rate limit is exceeded after retries
    """
    logger.info("Generating book title...")
    try:
        completion = st.session_state.groq.chat.completions.create(
            model=MODEL_LARGE,
            messages=[
                {
                    "role": "system",
                    "content": "Generate suitable book titles for the provided topics. There is only one generated book title! Don't give any explanation or add any symbols, just write the title of the book. The requirement for this title is that it must be between 7 and 25 words long, and it must be attractive enough!"
                },
                {
                    "role": "user",
                    "content": f"Generate a book title for the following topic. There is only one generated book title! Don't give any explanation or add any symbols, just write the title of the book. The requirement for this title is that it must be at least 7 words and 25 words long, and it must be attractive enough:\n\n{prompt}"
                }
            ],
            temperature=0.7,
            max_tokens=100,
            top_p=1,
            stream=False,
            stop=None,
        )
        
        title = completion.choices[0].message.content.strip()
        logger.info(f"Book title generated: {title}")
        return title
    except RateLimitError:
        raise
    except Exception as e:
        logger.error(f"Failed to generate book title: {e}")
        raise APIError(f"Failed to generate book title: {e}") from e


@retry_on_rate_limit()
def generate_book_structure(
    prompt: str, 
    additional_instructions: str = "",
    book_type: str = "general",
    structure_model: str = None
):
    """
    Returns book structure content as well as total tokens and total time for generation.
    
    Args:
        prompt: The main topic/subject for the book
        additional_instructions: Optional additional guidelines for structure generation
        book_type: Type of book (general, technical, educational, etc.)
        structure_model: Model to use for structure generation (defaults to MODEL_LARGE)
        
    Returns:
        Tuple of (GenerationStatistics, book_structure_json_string)
        
    Raises:
        APIError: If the API call fails
        RateLimitError: If rate limit is exceeded after retries
    """
    model = structure_model or MODEL_LARGE
    book_config = BOOK_TYPES.get(book_type, BOOK_TYPES["general"])
    
    # Build the structure prompt based on book type
    structure_base = book_config["structure_prompt"]
    style_guide = book_config.get("style_guide", "")
    
    system_prompt = f'''Write in JSON format:

{{"Title of section goes here":"Description of section goes here",
"Title of section goes here":{{"Title of section goes here":"Description of section goes here","Title of section goes here":"Description of section goes here","Title of section goes here":"Description of section goes here"}}}}

{style_guide}'''
    
    user_prompt = f"""{structure_base}, omitting introduction and conclusion sections (foreword, author's note, summary).

Very Important: Use the following subject and additional instructions to write the book structure.

Subject:
<subject>{prompt}</subject>

Book Type: {book_config["name"]}

Additional instructions:
<instructions>{additional_instructions}</instructions>"""

    logger.info(f"Generating book structure with model {model} for type {book_type}...")
    try:
        completion = st.session_state.groq.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=8000,
            top_p=1,
            stream=False,
            response_format={"type": "json_object"},
            stop=None,
        )

        usage = completion.usage
        statistics_to_return = GenerationStatistics(
            input_time=usage.prompt_time,
            output_time=usage.completion_time,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            total_time=usage.total_time,
            model_name=model,
        )
        
        logger.info(f"Book structure generated: {usage.completion_tokens} tokens in {usage.total_time:.2f}s")
        return statistics_to_return, completion.choices[0].message.content
    except RateLimitError:
        raise
    except Exception as e:
        logger.error(f"Failed to generate book structure: {e}")
        raise APIError(f"Failed to generate book structure: {e}") from e


def generate_section(
    prompt: str, 
    additional_instructions: str = "",
    content_model: str = None,
    book_type: str = "general",
    word_count: str = "medium",
    previous_section_content: str = None,
    seed_content: str = None
):
    """
    Generate content for a book section using streaming.
    
    Args:
        prompt: The section title and description
        additional_instructions: Additional guidelines for content generation
        content_model: Model to use for content generation (defaults to MODEL_SMALL)
        book_type: Type of book for style matching
        word_count: Target word count (short, medium, long, extended)
        previous_section_content: Content from the previous section for context
        seed_content: Sample content to match writing style
        
    Yields:
        Either string content tokens or GenerationStatistics object
        
    Note: This is a generator function, so retry decorator is not applied.
          Rate limit handling should be done at the call site.
    """
    model = content_model or MODEL_SMALL
    book_config = BOOK_TYPES.get(book_type, BOOK_TYPES["general"])
    word_config = WORD_COUNT_OPTIONS.get(word_count, WORD_COUNT_OPTIONS["medium"])
    
    # Build system prompt with book type styling
    content_base = book_config["content_prompt"]
    style_guide = book_config.get("style_guide", "")
    word_target = word_config["target"]
    
    system_parts = [
        f"You are an expert writer. {content_base} for the section provided.",
        f"Target approximately {word_target} words for this section.",
    ]
    
    if style_guide:
        system_parts.append(f"Style guidelines: {style_guide}")
    
    if seed_content:
        system_parts.append(f"\nMatch this writing style and tone:\n<style_example>\n{seed_content[:1500]}\n</style_example>")
    
    system_prompt = " ".join(system_parts)
    
    # Build user prompt with context awareness
    user_parts = [
        f"{content_base}.",
        f"\nSection:\n<section_title>{prompt}</section_title>",
    ]
    
    # Add previous section context for coherence
    if previous_section_content:
        # Use last ~500 chars of previous section for context
        context_snippet = previous_section_content[-500:] if len(previous_section_content) > 500 else previous_section_content
        user_parts.append(f"\nPrevious section ended with:\n<context>{context_snippet}</context>")
        user_parts.append("\nEnsure a smooth transition from the previous section.")
    
    if additional_instructions:
        user_parts.append(f"\nAdditional instructions:\n<instructions>{additional_instructions}</instructions>")
    
    user_parts.append(f"\nTarget length: approximately {word_target} words.")
    
    user_prompt = "\n".join(user_parts)
    
    logger.info(f"Generating section with model {model}: {prompt[:50]}...")
    
    stream = st.session_state.groq.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=8000,
        top_p=1,
        stream=True,
        stop=None,
    )

    for chunk in stream:
        tokens = chunk.choices[0].delta.content
        if tokens:
            yield tokens
        if x_groq := chunk.x_groq:
            if not x_groq.usage:
                continue
            usage = x_groq.usage
            statistics_to_return = GenerationStatistics(
                input_time=usage.prompt_time,
                output_time=usage.completion_time,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_time=usage.total_time,
                model_name=model,
            )
            logger.info(f"Section completed: {usage.completion_tokens} tokens in {usage.total_time:.2f}s")
            yield statistics_to_return


# Initialize
if "button_disabled" not in st.session_state:
    st.session_state.button_disabled = False

if "button_text" not in st.session_state:
    st.session_state.button_text = "Generate"

if "statistics_text" not in st.session_state:
    st.session_state.statistics_text = ""

if 'book_title' not in st.session_state:
    st.session_state.book_title = ""

st.write(
    """
# Groqbook: Write full books using llama3 (8b and 70b) on Groq
"""
)


def disable():
    st.session_state.button_disabled = True


def enable():
    st.session_state.button_disabled = False


def validate_api_key(api_key: str) -> bool:
    """
    Validate that an API key is properly formatted.
    Groq API keys start with 'gsk_' and are typically 56 characters.
    
    Args:
        api_key: The API key to validate
        
    Returns:
        True if the API key appears valid, False otherwise
    """
    if not api_key:
        return False
    if not api_key.startswith('gsk_'):
        return False
    if len(api_key) < 20:  # Reasonable minimum length
        return False
    return True


def sanitize_input(text: str) -> str:
    """
    Sanitize user input by removing potentially problematic characters.
    
    Args:
        text: The input text to sanitize
        
    Returns:
        Sanitized text string
    """
    if not text:
        return ""
    # Remove null bytes and other control characters (except newlines and tabs)
    sanitized = ''.join(char for char in text if char == '\n' or char == '\t' or (ord(char) >= 32 and ord(char) != 127))
    # Trim whitespace
    sanitized = sanitized.strip()
    return sanitized


def validate_topic(topic: str) -> str:
    """
    Validate and sanitize the book topic.
    
    Args:
        topic: The book topic to validate
        
    Returns:
        Sanitized topic string
        
    Raises:
        ValidationError: If the topic is invalid
    """
    sanitized = sanitize_input(topic)
    
    if len(sanitized) < MIN_TOPIC_LENGTH:
        raise ValidationError(
            f"Book topic must be at least {MIN_TOPIC_LENGTH} characters long. "
            f"Current length: {len(sanitized)} characters."
        )
    
    if len(sanitized) > MAX_TOPIC_LENGTH:
        raise ValidationError(
            f"Book topic must be less than {MAX_TOPIC_LENGTH} characters. "
            f"Current length: {len(sanitized)} characters."
        )
    
    logger.info(f"Topic validated: {len(sanitized)} characters")
    return sanitized


def validate_instructions(instructions: str) -> str:
    """
    Validate and sanitize additional instructions.
    
    Args:
        instructions: The additional instructions to validate
        
    Returns:
        Sanitized instructions string
        
    Raises:
        ValidationError: If the instructions are too long
    """
    sanitized = sanitize_input(instructions)
    
    if len(sanitized) > MAX_INSTRUCTIONS_LENGTH:
        raise ValidationError(
            f"Additional instructions must be less than {MAX_INSTRUCTIONS_LENGTH} characters. "
            f"Current length: {len(sanitized)} characters."
        )
    
    if sanitized:
        logger.info(f"Instructions validated: {len(sanitized)} characters")
    return sanitized


def count_sections(structure: dict) -> int:
    """
    Recursively count the total number of sections in the book structure.
    
    Args:
        structure: The book structure dictionary
        
    Returns:
        Total count of sections
    """
    count = 0
    for title, content in structure.items():
        count += 1
        if isinstance(content, dict):
            count += count_sections(content)
    return count



# ============================================================================
# SIDEBAR
# ============================================================================

# Sidebar
with st.sidebar:
    st.markdown("# 📚 Groqbook")
    st.markdown("*AI-Powered Book Generator*")
    st.divider()
    
    # App info
    st.markdown("### ℹ️ About")
    st.markdown("""
    Groqbook uses AI models on Groq Cloud to generate comprehensive books on any topic.
    """)
    
    with st.expander("🤖 Available Models", expanded=False):
        for model_id, info in AVAILABLE_MODELS.items():
            st.markdown(f"**{info['name']}**")
            st.caption(f"{info['description']} • {info['speed']} • {info['context_window']:,} tokens")
    
    st.divider()
    
    # Book Statistics (if book exists)
    if "book" in st.session_state and st.session_state.book:
        st.markdown("### 📊 Book Statistics")
        book = st.session_state.book
        section_list = book.get_section_list()
        completed = sum(1 for s in section_list if book.has_content(s))
        
        st.metric("Total Sections", len(section_list))
        st.metric("Completed", f"{completed}/{len(section_list)}")
        st.progress(completed / len(section_list) if section_list else 0)
        
        st.divider()
        
        # Section Regeneration
        st.markdown("### 🔄 Regenerate Section")
        st.markdown("*Select a section to regenerate its content*")
        
        # Only show sections that have content
        sections_with_content = [s for s in section_list if book.has_content(s)]
        
        if sections_with_content:
            selected_section = st.selectbox(
                "Select Section",
                options=sections_with_content,
                key="regen_section_select"
            )
            
            # Model selection for regeneration
            regen_model = st.selectbox(
                "Model",
                options=list(AVAILABLE_MODELS.keys()),
                index=1,  # Default to fast model
                format_func=lambda x: f"{AVAILABLE_MODELS[x]['name']} ({AVAILABLE_MODELS[x]['speed']})",
                key="regen_model"
            )
            
            regen_instructions = st.text_area(
                "Additional Instructions (optional)",
                placeholder="Any specific changes or focus areas for this section...",
                key="regen_instructions",
                height=80
            )
            
            if st.button("🔄 Regenerate Section", use_container_width=True):
                if selected_section:
                    try:
                        # Clear existing content
                        book.clear_section(selected_section)
                        
                        # Get section description
                        description = book.get_section_description(selected_section)
                        prompt = f"{selected_section}: {description}"
                        
                        # Generate new content
                        st.info(f"Regenerating: {selected_section}...")
                        
                        content_stream = generate_section(
                            prompt=prompt,
                            additional_instructions=regen_instructions or "",
                            content_model=regen_model
                        )
                        
                        for chunk in content_stream:
                            if isinstance(chunk, GenerationStatistics):
                                pass  # Skip statistics for sidebar
                            elif chunk is not None:
                                book.update_content(selected_section, chunk)
                        
                        st.success(f"✅ Regenerated: {selected_section}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.info("Generate a book first to enable section regeneration.")
    else:
        st.info("📖 Generate a book to see statistics and options here.")
    
    st.divider()
    
    # Export formats info
    st.markdown("### 📥 Export Formats")
    st.markdown("""
    - 📄 **Text/Markdown** - Plain text format
    - 📕 **PDF** - Styled with cover page
    - 📚 **EPUB** - E-reader compatible
    """)
    
    # Footer
    st.divider()
    st.markdown(
        "<small>Built with Streamlit & Groq</small>",
        unsafe_allow_html=True
    )


try:
    if st.button("📥 End Generation and Download Book"):
        if "book" in st.session_state and st.session_state.book_title:
            st.success("✅ Your book is ready for download!")
            
            book_content = st.session_state.book.get_markdown_content()
            book_title = st.session_state.book_title
            
            # Create download buttons in columns for better layout
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Create markdown/text file
                markdown_file = create_markdown_file(book_content)
                st.download_button(
                    label="📄 Download Text",
                    data=markdown_file,
                    file_name=f'{book_title}.txt',
                    mime='text/plain',
                    use_container_width=True
                )
            
            with col2:
                # Create enhanced PDF file with cover page
                pdf_file = create_pdf_file(book_content, book_title)
                st.download_button(
                    label="📕 Download PDF",
                    data=pdf_file,
                    file_name=f'{book_title}.pdf',
                    mime='application/pdf',
                    use_container_width=True
                )
            
            with col3:
                # Create EPUB file (if available)
                if EPUB_AVAILABLE:
                    try:
                        epub_file = create_epub_file(book_content, book_title)
                        st.download_button(
                            label="📚 Download EPUB",
                            data=epub_file,
                            file_name=f'{book_title}.epub',
                            mime='application/epub+zip',
                            use_container_width=True
                        )
                    except Exception as e:
                        st.warning(f"EPUB generation unavailable: {e}")
                else:
                    st.info("📚 EPUB export requires ebooklib. Install with: `pip install ebooklib`")
        else:
            st.warning("⚠️ Please generate content first before downloading the book.")

    with st.form("groqform"):
        if not GROQ_API_KEY:
            groq_input_key = st.text_input(
                "Enter your Groq API Key (gsk_yA...):", "", type="password"
            )

        topic_text = st.text_input(
            "📖 What do you want the book to be about?",
            value="",
            help="Enter the main topic or title of your book",
        )
        
        # Book Type Selection
        st.markdown("### 📚 Book Type")
        book_type_options = {k: v["name"] for k, v in BOOK_TYPES.items()}
        selected_book_type = st.selectbox(
            "Select the type of book to generate",
            options=list(book_type_options.keys()),
            format_func=lambda x: book_type_options[x],
            help="Different book types use specialized prompts and styling"
        )
        
        # Show book type description
        if selected_book_type:
            st.caption(f"*{BOOK_TYPES[selected_book_type]['description']}*")

        additional_instructions = st.text_area(
            "✍️ Additional Instructions (optional)",
            help="Provide any specific guidelines or preferences for the book's content",
            placeholder="E.g., 'Focus on beginner-friendly content', 'Include case studies', etc.",
            value="",
        )
        
        # Advanced Options (collapsible)
        with st.expander("⚙️ Advanced Options", expanded=False):
            st.markdown("#### 🎯 Word Count per Section")
            word_count_options = {k: v["name"] for k, v in WORD_COUNT_OPTIONS.items()}
            selected_word_count = st.selectbox(
                "Target word count for each section",
                options=list(word_count_options.keys()),
                index=1,  # Default to "medium"
                format_func=lambda x: word_count_options[x],
                help="Longer sections provide more detail but take longer to generate"
            )
            
            st.markdown("#### 🤖 Model Selection")
            col_m1, col_m2 = st.columns(2)
            
            with col_m1:
                structure_model_options = {k: f"{v['name']} ({v['speed']})" for k, v in AVAILABLE_MODELS.items()}
                selected_structure_model = st.selectbox(
                    "Structure Generation Model",
                    options=list(structure_model_options.keys()),
                    index=0,  # Default to llama3-70b
                    format_func=lambda x: structure_model_options[x],
                    help="Larger models create better structure but are slower"
                )
            
            with col_m2:
                content_model_options = {k: f"{v['name']} ({v['speed']})" for k, v in AVAILABLE_MODELS.items()}
                selected_content_model = st.selectbox(
                    "Content Generation Model",
                    options=list(content_model_options.keys()),
                    index=1,  # Default to llama3-8b
                    format_func=lambda x: content_model_options[x],
                    help="Faster models are better for content generation"
                )
            
            st.markdown("#### 🎨 Writing Style (Seed Content)")
            seed_content = st.text_area(
                "Paste sample text to match writing style (optional)",
                help="The AI will try to match the tone and style of this sample text",
                placeholder="Paste a paragraph or two from a book or article whose style you want to emulate...",
                value="",
                height=100
            )
            
            st.markdown("#### 🔗 Context-Aware Generation")
            use_context = st.checkbox(
                "Enable section transitions",
                value=True,
                help="Pass context from previous sections for smoother transitions between chapters"
            )

        # Generate button
        submitted = st.form_submit_button(
            st.session_state.button_text,
            on_click=disable,
            disabled=st.session_state.button_disabled,
            use_container_width=True
        )

        # Statistics
        placeholder = st.empty()

        def display_statistics():
            with placeholder.container():
                if st.session_state.statistics_text:
                    if (
                        "Generating structure in background"
                        not in st.session_state.statistics_text
                    ):
                        st.markdown(
                            st.session_state.statistics_text + "\n\n---\n"
                        )  # Format with line if showing statistics
                    else:
                        st.markdown(st.session_state.statistics_text)
                else:
                    placeholder.empty()

        if submitted:
            # Validate and sanitize inputs
            validated_topic = validate_topic(topic_text)
            validated_instructions = validate_instructions(additional_instructions)
            
            # Get selected options (with defaults if expandable section wasn't opened)
            book_type = selected_book_type if 'selected_book_type' in dir() else "general"
            word_count = selected_word_count if 'selected_word_count' in dir() else "medium"
            structure_model = selected_structure_model if 'selected_structure_model' in dir() else MODEL_LARGE
            content_model = selected_content_model if 'selected_content_model' in dir() else MODEL_SMALL
            validated_seed = seed_content.strip()[:MAX_SEED_CONTENT_LENGTH] if 'seed_content' in dir() and seed_content else None
            context_enabled = use_context if 'use_context' in dir() else True

            # Validate API key if user provided one
            if not GROQ_API_KEY:
                if not validate_api_key(groq_input_key):
                    raise ValidationError("Invalid API key format. Groq API keys should start with 'gsk_'")
                st.session_state.groq = Groq(api_key=groq_input_key)

            st.session_state.button_disabled = True
            st.session_state.statistics_text = f"🔄 Generating book title and structure ({BOOK_TYPES[book_type]['name']})..."
            display_statistics()
            logger.info(f"Starting book generation for topic: {validated_topic[:50]}... [type={book_type}, models={structure_model}/{content_model}]")

            # Generate AI book title
            st.session_state.book_title = generate_book_title(validated_topic)
            st.write(f"## {st.session_state.book_title}")

            # Generate book structure with new parameters
            large_model_generation_statistics, book_structure = generate_book_structure(
                validated_topic, 
                validated_instructions,
                book_type=book_type,
                structure_model=structure_model
            )

            total_generation_statistics = GenerationStatistics(
                model_name=content_model
            )

            try:
                book_structure_json = json.loads(book_structure)
                book = Book(st.session_state.book_title, book_structure_json)
                
                # Always update session state with the new book
                st.session_state.book = book

                # Print the book structure to the terminal to show structure
                logger.info(f"Book structure:\n{json.dumps(book_structure_json, indent=2)}")

                # Count total sections for progress tracking
                total_sections = count_sections(book_structure_json)
                current_section = [0]  # Use list to allow modification in nested function
                section_times = []  # Track time per section for ETA
                previous_content = [None]  # Track previous section for context
                logger.info(f"Total sections to generate: {total_sections}")
                
                # Create progress placeholders
                st.divider()
                progress_bar = st.progress(0.0, text="Starting generation...")
                progress_status = st.empty()

                st.session_state.book.display_structure()
    
                def stream_section_content(sections):
                    for title, content in sections.items():
                        if isinstance(content, str):
                            current_section[0] += 1
                            section_start_time = time.time()
                            
                            # Update progress bar
                            progress_percent = current_section[0] / total_sections
                            progress_bar.progress(
                                progress_percent, 
                                text=f"Generating: {current_section[0]}/{total_sections} sections ({progress_percent*100:.0f}%)"
                            )
                            
                            # Calculate ETA
                            if section_times:
                                avg_time = sum(section_times) / len(section_times)
                                remaining = total_sections - current_section[0]
                                eta_seconds = avg_time * remaining
                                eta_display = f"~{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s remaining" if eta_seconds > 60 else f"~{int(eta_seconds)}s remaining"
                            else:
                                eta_display = "Calculating..."
                            
                            progress_status.markdown(
                                f"📝 **Section {current_section[0]} of {total_sections}:** _{title}_  \n"
                                f"⏱️ {eta_display}"
                            )
                            
                            try:
                                # Use enhanced generate_section with all options
                                content_stream = generate_section(
                                    prompt=title + ": " + content,
                                    additional_instructions=validated_instructions,
                                    content_model=content_model,
                                    book_type=book_type,
                                    word_count=word_count,
                                    previous_section_content=previous_content[0] if context_enabled else None,
                                    seed_content=validated_seed
                                )
                                
                                section_text = ""  # Collect section text for context
                                
                                for chunk in content_stream:
                                    # Check if GenerationStatistics data is returned instead of str tokens
                                    if isinstance(chunk, GenerationStatistics):
                                        total_generation_statistics.add(chunk)

                                        st.session_state.statistics_text = str(
                                            total_generation_statistics
                                        )
                                        display_statistics()

                                    elif chunk is not None:
                                        st.session_state.book.update_content(title, chunk)
                                        section_text += chunk
                                
                                # Update previous content for next section
                                if context_enabled:
                                    previous_content[0] = section_text
                                
                                # Track section time
                                section_times.append(time.time() - section_start_time)
                                        
                            except Exception as e:
                                logger.error(f"Error generating section '{title}': {e}")
                                st.warning(f"⚠️ Error generating section '{title}'. Continuing with next section...")
                                continue
                                
                        elif isinstance(content, dict):
                            stream_section_content(content)

                stream_section_content(book_structure_json)
                
                # Clear progress and show completion
                progress_bar.progress(1.0, text="✅ Generation complete!")
                
                # Calculate total time
                total_time = sum(section_times)
                time_display = f"{int(total_time // 60)}m {int(total_time % 60)}s" if total_time > 60 else f"{total_time:.1f}s"
                
                progress_status.markdown(
                    f"✅ **Book generation complete!**  \n"
                    f"📊 Generated **{total_sections}** sections in **{time_display}**  \n"
                    f"📥 Click the download button above to save your book!"
                )
                logger.info(f"Book generation complete: {total_sections} sections in {time_display}")

            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode book structure: {e}")
                raise GenerationError("Failed to decode the book structure. Please try again.")

            enable()

except ValidationError as e:
    st.session_state.button_disabled = False
    st.error(f"⚠️ Validation Error: {e}")
    logger.warning(f"Validation error: {e}")

except RateLimitError as e:
    st.session_state.button_disabled = False
    st.error(f"🚫 Rate Limit Exceeded: {e}")
    st.info("💡 Tip: Wait a few seconds and try again, or use a different API key.")
    logger.warning(f"Rate limit error: {e}")

except APIError as e:
    st.session_state.button_disabled = False
    st.error(f"⚠️ API Error: {e}")
    st.info("💡 Tip: Check your API key and try again.")
    logger.error(f"API error: {e}")

except GenerationError as e:
    st.session_state.button_disabled = False
    st.error(f"⚠️ Generation Error: {e}")
    logger.error(f"Generation error: {e}")

except Exception as e:
    st.session_state.button_disabled = False
    st.error(f"❌ Unexpected Error: {e}")
    logger.exception(f"Unexpected error: {e}")

    if st.button("Clear"):
        st.rerun()
