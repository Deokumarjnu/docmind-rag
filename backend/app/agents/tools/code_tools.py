"""Tools for Code Specialist subagent.

These tools enable the code specialist to detect, parse,
and chunk code from documents.
"""

import logging
from typing import Optional

from langchain_core.tools import tool

from app.ingestion.code_chunker import (
    detect_code_block,
    detect_programming_language,
    chunk_python_code,
    chunk_js_code,
    chunk_generic_code,
)

logger = logging.getLogger(__name__)


@tool
def detect_programming_language_tool(content: str) -> str:
    """
    Detect the programming language of code content.
    
    Args:
        content: Code text
        
    Returns:
        Language name (python, javascript, etc.) or 'unknown'
    """
    return detect_programming_language(content)


@tool
def extract_code_blocks(content: str) -> list[dict]:
    """
    Extract code blocks from document content.
    
    Args:
        content: Document text content
        
    Returns:
        List of code blocks with language info
    """
    blocks = []
    
    # Simple detection: look for code patterns
    if not detect_code_block(content):
        return blocks
    
    language = detect_programming_language(content)
    
    # Split by blank lines and analyze each block
    sections = content.split('\n\n')
    
    for i, section in enumerate(sections):
        section = section.strip()
        if section and detect_code_block(section):
            blocks.append({
                "index": i,
                "content": section,
                "language": language,
                "line_count": len(section.split('\n')),
            })
    
    return blocks


@tool
def parse_with_ast(code: str, language: str) -> dict:
    """
    Parse code and extract structure using AST.
    
    Args:
        code: Code content
        language: Programming language
        
    Returns:
        Dictionary with parsed structure
    """
    result = {
        "language": language,
        "functions": [],
        "classes": [],
        "imports": [],
        "parse_success": False,
    }
    
    if language == "python":
        try:
            import ast
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    result["functions"].append({
                        "name": node.name,
                        "line": node.lineno,
                        "args": [arg.arg for arg in node.args.args],
                    })
                elif isinstance(node, ast.AsyncFunctionDef):
                    result["functions"].append({
                        "name": node.name,
                        "line": node.lineno,
                        "args": [arg.arg for arg in node.args.args],
                        "async": True,
                    })
                elif isinstance(node, ast.ClassDef):
                    result["classes"].append({
                        "name": node.name,
                        "line": node.lineno,
                    })
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        result["imports"].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    result["imports"].append(f"{node.module}")
            
            result["parse_success"] = True
            
        except SyntaxError as e:
            result["parse_error"] = str(e)
    
    return result


@tool
def chunk_by_functions(code: str, language: str) -> list[str]:
    """
    Split code into chunks by functions/classes.
    
    Args:
        code: Code content
        language: Programming language
        
    Returns:
        List of code chunks
    """
    if language == "python":
        return chunk_python_code(code)
    elif language in ["javascript", "typescript"]:
        return chunk_js_code(code)
    else:
        return chunk_generic_code(code)

