"""
Stage 1: Safe AST Parsing.

Accepts explicitly supplied Python source files, performs safe UTF-8 reading with fallbacks,
computes SHA-256 hashes for source provenance, parses using ast.parse(), handles syntax errors
gracefully without terminating, and provides safe AST helper inspection utilities.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union


class ParsedSourceFile:
    """Represents a safely parsed Python source file and its AST."""

    def __init__(
        self,
        path: Path,
        source: str,
        sha256_hash: str,
        tree: Optional[ast.AST],
        status: str = "ok",
        error: Optional[str] = None,
    ):
        self.path = path
        self.filename = path.name
        self.source = source
        self.sha256 = sha256_hash
        self.tree = tree
        self.status = status
        self.error = error

    @property
    def is_valid(self) -> bool:
        return self.status == "ok" and self.tree is not None


class PythonASTParser:
    """Safe, non-executing AST compiler for explicitly supplied Python files."""

    @staticmethod
    def safe_read_text(path: Path) -> str:
        """Safely read source text with encoding fallbacks."""
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                return path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                return ""
        except Exception:
            return ""

    @staticmethod
    def node_source(node: Optional[ast.AST], source: str) -> str:
        """Extract exact source code segment corresponding to an AST node."""
        if node is None or not source:
            return ""
        try:
            return ast.get_source_segment(source, node) or ""
        except Exception:
            return ""

    @staticmethod
    def literal_value(node: Optional[ast.AST]) -> Any:
        """Extract literal values without code execution."""
        if node is None:
            return None
        try:
            return ast.literal_eval(node)
        except Exception:
            return None

    @classmethod
    def dotted_name(cls, node: Optional[ast.AST]) -> str:
        """Recursively resolve dotted identifier names."""
        if node is None:
            return ""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            left = cls.dotted_name(node.value)
            return f"{left}.{node.attr}" if left else node.attr
        if isinstance(node, ast.Call):
            return cls.dotted_name(node.func)
        return ""

    @staticmethod
    def extract_string_constants(node: Optional[ast.AST]) -> List[str]:
        """Extract string constants from an AST subtree while preserving order."""
        if node is None:
            return []
        values = []
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                val = child.value.strip()
                if val:
                    values.append(val)
        return list(dict.fromkeys(values))

    def parse_file(self, file_path: Union[str, Path]) -> ParsedSourceFile:
        """Parse a single explicitly provided Python file."""
        path = Path(file_path).resolve()
        source = self.safe_read_text(path)
        sha256_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()

        try:
            tree = ast.parse(source, filename=str(path))
            return ParsedSourceFile(
                path=path,
                source=source,
                sha256_hash=sha256_hash,
                tree=tree,
                status="ok",
                error=None,
            )
        except SyntaxError as exc:
            return ParsedSourceFile(
                path=path,
                source=source,
                sha256_hash=sha256_hash,
                tree=None,
                status="syntax_error",
                error=f"{exc.msg} (line {exc.lineno})",
            )
        except Exception as exc:
            return ParsedSourceFile(
                path=path,
                source=source,
                sha256_hash=sha256_hash,
                tree=None,
                status="parse_error",
                error=str(exc),
            )

    def parse_files(self, file_paths: List[Union[str, Path]]) -> List[ParsedSourceFile]:
        """Parse a collection of explicitly supplied Python files."""
        return [self.parse_file(p) for p in file_paths]

