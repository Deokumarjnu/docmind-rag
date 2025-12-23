"""Code detection and AST-based chunking.

This module detects programming code in documents and chunks it
while preserving logical units like functions and classes.
"""

import ast
import logging
import re
from typing import Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def detect_code_block(text: str) -> bool:
    """
    Detect if text contains programming code.
    
    Uses pattern matching for common code indicators across multiple languages.
    
    Args:
        text: Text to analyze
        
    Returns:
        True if code detected
    """
    code_indicators = [
        (r'def\s+\w+\s*\(', 2),           # Python function
        (r'function\s+\w+\s*\(', 2),       # JavaScript function
        (r'class\s+\w+[\s:{]', 2),         # Class definition
        (r'import\s+[\w.{},\s]+', 1),      # Import statements
        (r'from\s+\w+\s+import', 2),       # Python imports
        (r'const\s+\w+\s*=', 1),           # JS const
        (r'let\s+\w+\s*=', 1),             # JS let
        (r'var\s+\w+\s*=', 1),             # JS var
        (r'^\s{4,}\S', 1),                 # Significant indentation
        (r'if\s*\(.*\)\s*\{', 1),          # If statements (C-style)
        (r'for\s*\(.*\)\s*\{', 1),         # For loops (C-style)
        (r'=>', 1),                         # Arrow functions
        (r'console\.log\(', 1),            # JS logging
        (r'print\(', 1),                   # Python print
        (r'return\s+\w', 1),               # Return statements
        (r'async\s+(def|function)', 2),    # Async functions
        (r'await\s+\w', 1),                # Await expressions
        (r'\}\s*else\s*\{', 1),            # Else blocks
        (r'try\s*[:{]', 1),                # Try blocks
        (r'except\s+\w', 1),               # Python except
        (r'catch\s*\(', 1),                # JS catch
    ]
    
    score = sum(
        weight for pattern, weight in code_indicators
        if re.search(pattern, text, re.MULTILINE)
    )
    return score >= 3


def detect_programming_language(text: str) -> str:
    """
    Detect the programming language of code.
    
    Args:
        text: Code text
        
    Returns:
        Language name: python, javascript, typescript, java, go, rust, sql, or unknown
    """
    patterns = {
        'python': [
            r'def\s+\w+\s*\(', 
            r'import\s+\w+', 
            r'from\s+\w+\s+import', 
            r':\s*$', 
            r'elif\s+',
            r'self\.',
            r'__\w+__',
        ],
        'javascript': [
            r'function\s+\w+', 
            r'const\s+\w+', 
            r'let\s+\w+', 
            r'=>', 
            r'console\.',
            r'require\(',
            r'module\.exports',
        ],
        'typescript': [
            r':\s*(string|number|boolean|any)', 
            r'interface\s+\w+', 
            r'type\s+\w+\s*=',
            r'<\w+>',
            r'as\s+\w+',
        ],
        'java': [
            r'public\s+class', 
            r'private\s+\w+', 
            r'System\.out\.',
            r'void\s+\w+',
            r'@\w+',
        ],
        'go': [
            r'func\s+\w+', 
            r'package\s+\w+', 
            r':=',
            r'fmt\.',
            r'import\s*\(',
        ],
        'rust': [
            r'fn\s+\w+', 
            r'let\s+mut', 
            r'impl\s+\w+',
            r'pub\s+fn',
            r'->',
        ],
        'sql': [
            r'SELECT\s+', 
            r'FROM\s+', 
            r'WHERE\s+', 
            r'INSERT\s+INTO',
            r'CREATE\s+TABLE',
        ],
    }
    
    scores = {}
    for lang, lang_patterns in patterns.items():
        scores[lang] = sum(
            1 for p in lang_patterns
            if re.search(p, text, re.IGNORECASE | re.MULTILINE)
        )
    
    if max(scores.values()) > 0:
        return max(scores, key=scores.get)
    return 'unknown'


def chunk_code_content(doc: Document) -> list[Document]:
    """
    Chunk code while preserving logical units.
    
    Args:
        doc: Document containing code
        
    Returns:
        List of chunked code documents
    """
    text = doc.page_content
    language = detect_programming_language(text)
    
    if language == 'python':
        chunks = chunk_python_code(text)
    elif language in ['javascript', 'typescript']:
        chunks = chunk_js_code(text)
    else:
        chunks = chunk_generic_code(text)
    
    return [
        Document(
            page_content=chunk,
            metadata={
                **doc.metadata,
                "content_type": "code",
                "language": language,
                "chunk_method": "code_aware",
            }
        )
        for chunk in chunks
    ]


def chunk_python_code(text: str) -> list[str]:
    """
    Split Python code by functions/classes using AST.
    
    Args:
        text: Python code
        
    Returns:
        List of code chunks
    """
    try:
        tree = ast.parse(text)
        chunks = []
        lines = text.split('\n')
        
        # Track what lines are covered by top-level definitions
        covered_lines = set()
        
        # Extract top-level definitions
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = node.lineno - 1
                end = node.end_lineno if hasattr(node, 'end_lineno') else start + 20
                code_chunk = '\n'.join(lines[start:end])
                chunks.append(code_chunk)
                covered_lines.update(range(start, end))
        
        # Get any code outside functions/classes (imports, globals, etc.)
        other_lines = []
        for i, line in enumerate(lines):
            if i not in covered_lines and line.strip():
                other_lines.append(line)
        
        if other_lines:
            other_code = '\n'.join(other_lines)
            if len(other_code.strip()) > 20:
                chunks.insert(0, other_code)
        
        if not chunks:
            return chunk_generic_code(text)
        
        return chunks
        
    except SyntaxError:
        logger.debug("Python AST parsing failed, using generic chunking")
        return chunk_generic_code(text)


def chunk_js_code(text: str) -> list[str]:
    """
    Split JavaScript code by functions.
    
    Args:
        text: JavaScript code
        
    Returns:
        List of code chunks
    """
    # Match function declarations and arrow functions
    function_pattern = (
        r'((?:async\s+)?'
        r'(?:function\s+\w+|'
        r'const\s+\w+\s*=\s*(?:async\s+)?\([^)]*\)\s*=>|'
        r'\w+\s*:\s*(?:async\s+)?function)'
        r'[^{]*\{)'
    )
    
    parts = re.split(function_pattern, text)
    chunks = []
    current_chunk = ""
    brace_count = 0
    
    for part in parts:
        current_chunk += part
        brace_count += part.count('{') - part.count('}')
        
        if brace_count == 0 and current_chunk.strip():
            chunks.append(current_chunk.strip())
            current_chunk = ""
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks if chunks else chunk_generic_code(text)


def chunk_generic_code(text: str, max_chunk_size: int = 1500) -> list[str]:
    """
    Fallback: Split by blank lines while keeping logical blocks together.
    
    Args:
        text: Code text
        max_chunk_size: Maximum chunk size in characters
        
    Returns:
        List of code chunks
    """
    # Split by double newlines (blank lines)
    blocks = re.split(r'\n\s*\n', text)
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        
        block_size = len(block)
        
        # If single block exceeds max, split it further
        if block_size > max_chunk_size:
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            # Split large block by single newlines
            sub_lines = block.split('\n')
            sub_chunk = []
            sub_size = 0
            
            for line in sub_lines:
                if sub_size + len(line) > max_chunk_size and sub_chunk:
                    chunks.append('\n'.join(sub_chunk))
                    sub_chunk = []
                    sub_size = 0
                sub_chunk.append(line)
                sub_size += len(line)
            
            if sub_chunk:
                chunks.append('\n'.join(sub_chunk))
        
        elif current_size + block_size > max_chunk_size:
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
            current_chunk = [block]
            current_size = block_size
        else:
            current_chunk.append(block)
            current_size += block_size
    
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks if chunks else [text]


def process_document_with_code_detection(doc: Document) -> list[Document]:
    """
    Process document with code-aware chunking if code is detected.
    
    Args:
        doc: Document to process
        
    Returns:
        List of chunked documents
    """
    if detect_code_block(doc.page_content):
        return chunk_code_content(doc)
    else:
        # Return as-is for regular text processing
        return [doc]


class CodeChunker:
    """
    Code-aware chunker for processing code documents.
    """

    def __init__(self, max_chunk_size: int = 1500):
        """
        Initialize code chunker.
        
        Args:
            max_chunk_size: Maximum chunk size for generic code
        """
        self.max_chunk_size = max_chunk_size

    def is_code(self, text: str) -> bool:
        """Check if text contains code."""
        return detect_code_block(text)

    def detect_language(self, text: str) -> str:
        """Detect programming language."""
        return detect_programming_language(text)

    def chunk(self, doc: Document) -> list[Document]:
        """Chunk a code document."""
        return chunk_code_content(doc)

    def process(self, documents: list[Document]) -> list[Document]:
        """Process documents, chunking code documents."""
        result = []
        for doc in documents:
            if self.is_code(doc.page_content):
                result.extend(self.chunk(doc))
            else:
                result.append(doc)
        return result

