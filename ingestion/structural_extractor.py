"""
Stage 2: Structural Information Extraction.

Extracts imports, classes, functions, variable assignments, object constructors,
calls, control structures, docstrings, and environment variable references.
Preserves original source locations and distinguishes imported symbols from truly unresolved references.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple

from ingestion.python_ast_parser import ParsedSourceFile, PythonASTParser
from schemas.evidence_schema import EvidenceStatus, SourceReference
from schemas.entity_schema import (
    ImportReference,
    FunctionDefinition,
    ClassDefinition,
)

STANDARD_BUILTINS = {
    "print", "len", "range", "round", "abs", "float", "int", "str", "bool",
    "dict", "list", "set", "tuple", "dumps", "loads", "get", "append", "time",
    "format", "strip", "isinstance", "enumerate", "zip", "sum", "min", "max",
    "open", "super", "getattr", "setattr", "hasattr", "Exception", "ValueError",
    "TypeError", "KeyError", "IndexError", "FileNotFoundError", "RuntimeError",
    "None", "True", "False",
}


class FileStructuralContext:
    """Holds all extracted structural artifacts for a single source file."""

    def __init__(self, parsed_file: ParsedSourceFile):
        self.parsed_file = parsed_file
        self.filename = parsed_file.filename
        self.path = parsed_file.path
        self.source = parsed_file.source
        self.imports: List[ImportReference] = []
        self.imported_symbols: Dict[str, str] = {}  # symbol_name -> module_name
        self.external_modules: List[str] = []
        self.classes: List[ClassDefinition] = []
        self.functions: List[FunctionDefinition] = []
        self.top_level_constants: Dict[str, Any] = {}
        self.constructor_calls: List[Dict[str, Any]] = []
        self.function_calls: List[Dict[str, Any]] = []
        self.env_var_references: List[Dict[str, Any]] = []
        self.string_constants: List[Dict[str, Any]] = []
        self.unresolved_references: List[str] = []


class StructuralExtractor:
    """Extracts structural AST constructs deterministically from ParsedSourceFile objects."""

    def __init__(self, parser: Optional[PythonASTParser] = None):
        self.parser = parser or PythonASTParser()

    def extract_structure(
        self,
        parsed_file: ParsedSourceFile,
        known_modules: Optional[Set[str]] = None,
    ) -> FileStructuralContext:
        """Extract all structural elements from a single parsed file."""
        ctx = FileStructuralContext(parsed_file)
        if not parsed_file.is_valid:
            return ctx

        tree = parsed_file.tree
        source = parsed_file.source
        known_mods = known_modules or set()

        # 1. Imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod_name = alias.name
                    as_name = alias.asname or alias.name
                    ctx.imported_symbols[as_name] = mod_name
                    is_known = mod_name in known_mods or mod_name.split(".")[0] in known_mods

                    imp = ImportReference(
                        module=mod_name,
                        names=[mod_name],
                        aliases={mod_name: alias.asname} if alias.asname else {},
                        level=0,
                        line=node.lineno,
                        is_relative=False,
                        status=EvidenceStatus.STATICALLY_RESOLVED,
                    )
                    ctx.imports.append(imp)
                    if not is_known and mod_name not in ["sys", "os", "json", "time", "typing", "pathlib", "ast", "re", "hashlib"]:
                        ctx.external_modules.append(mod_name)

            elif isinstance(node, ast.ImportFrom):
                mod_name = node.module or ""
                names = [a.name for a in node.names]
                aliases = {a.name: a.asname for a in node.names if a.asname}

                for a in node.names:
                    bound_name = a.asname or a.name
                    ctx.imported_symbols[bound_name] = f"{mod_name}.{a.name}" if mod_name else a.name

                is_known = mod_name in known_mods or mod_name.split(".")[0] in known_mods
                imp = ImportReference(
                    module=mod_name,
                    names=names,
                    aliases=aliases,
                    level=node.level,
                    line=node.lineno,
                    is_relative=node.level > 0,
                    status=EvidenceStatus.STATICALLY_RESOLVED,
                )
                ctx.imports.append(imp)
                if not is_known and mod_name not in ["typing", "pathlib", "dataclasses", "collections"]:
                    ctx.external_modules.append(mod_name)

        # 2. Top-level assignments and constants
        for node in tree.body:
            if isinstance(node, ast.Assign):
                val = self.parser.literal_value(node.value)
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        ctx.top_level_constants[target.id] = (
                            val if val is not None else self.parser.node_source(node.value, source)
                        )

        # 3. Environment Variable References (os.environ, os.getenv)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = self.parser.dotted_name(node.func)
                if "getenv" in func_name:
                    env_name = self.parser.literal_value(node.args[0]) if node.args else None
                    ctx.env_var_references.append({
                        "name": env_name,
                        "line": node.lineno,
                        "method": func_name,
                    })
            elif isinstance(node, ast.Subscript):
                val_name = self.parser.dotted_name(node.value)
                if "environ" in val_name:
                    env_key = self.parser.literal_value(getattr(node.slice, "value", node.slice))
                    ctx.env_var_references.append({
                        "name": env_key,
                        "line": node.lineno,
                        "method": "os.environ[]",
                    })

        # 4. Classes
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.append(self._extract_function(item, source, parsed_file.filename, is_method=True))

                cls_def = ClassDefinition(
                    name=node.name,
                    bases=[self.parser.dotted_name(b) for b in node.bases],
                    decorators=[self.parser.dotted_name(d) for d in node.decorator_list],
                    methods=methods,
                    docstring=ast.get_docstring(node),
                    source_ref=SourceReference(
                        file=parsed_file.filename,
                        start_line=node.lineno,
                        end_line=getattr(node, "end_lineno", node.lineno),
                    ),
                )
                ctx.classes.append(cls_def)

        # 5. Functions (Top-level)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                ctx.functions.append(self._extract_function(node, source, parsed_file.filename, is_method=False))

        # 6. Object Constructors and Function Calls
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call_name = self.parser.dotted_name(node.func)
                leaf = call_name.split(".")[-1]
                kwargs = {
                    kw.arg: self.parser.literal_value(kw.value) or self.parser.node_source(kw.value, source)
                    for kw in node.keywords if kw.arg
                }
                call_info = {
                    "call_name": call_name,
                    "leaf": leaf,
                    "line": node.lineno,
                    "end_line": getattr(node, "end_lineno", node.lineno),
                    "args": [self.parser.literal_value(a) or self.parser.node_source(a, source) for a in node.args],
                    "kwargs": kwargs,
                    "node": node,
                }
                ctx.function_calls.append(call_info)
                if leaf and (leaf[0].isupper() or "agent" in leaf.lower() or "task" in leaf.lower() or "crew" in leaf.lower()):
                    ctx.constructor_calls.append(call_info)

        # 7. Significant String Constants
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                val = node.value.strip()
                if len(val) >= 20:
                    ctx.string_constants.append({
                        "value": val,
                        "line": node.lineno,
                    })

        # 8. Check for truly unresolved references in function calls
        local_definitions = set(ctx.top_level_constants.keys())
        local_definitions.update(f.name for f in ctx.functions)
        local_definitions.update(c.name for c in ctx.classes)
        local_definitions.update(ctx.imported_symbols.keys())
        local_definitions.update(STANDARD_BUILTINS)

        for call in ctx.function_calls:
            func_name = call["call_name"]
            base_ident = func_name.split(".")[0]
            if base_ident and base_ident not in local_definitions and not base_ident.startswith("_"):
                ctx.unresolved_references.append(base_ident)

        ctx.external_modules = list(dict.fromkeys(ctx.external_modules))
        ctx.unresolved_references = list(dict.fromkeys(ctx.unresolved_references))
        return ctx

    def _extract_function(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        source: str,
        filename: str,
        is_method: bool = False,
    ) -> FunctionDefinition:
        """Extract structured representation of a function or method."""
        called_funcs = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                name = self.parser.dotted_name(child.func)
                if name:
                    called_funcs.append(name)

        # Returns
        returns = None
        for child in ast.walk(node):
            if isinstance(child, ast.Return) and child.value is not None:
                # Capture clean representation, not framework constructors
                if isinstance(child.value, ast.Call):
                    cname = self.parser.dotted_name(child.value.func)
                    if cname in {"Agent", "Task", "Crew"}:
                        returns = f"<{cname} instance>"
                    else:
                        returns = self.parser.node_source(child.value, source)[:100]
                else:
                    returns = self.parser.node_source(child.value, source)[:100]
                break

        return FunctionDefinition(
            name=node.name,
            args=[a.arg for a in node.args.args],
            defaults=[self.parser.node_source(d, source) for d in node.args.defaults],
            returns=returns,
            decorators=[self.parser.dotted_name(d) for d in node.decorator_list],
            docstring=ast.get_docstring(node),
            source_ref=SourceReference(
                file=filename,
                start_line=node.lineno,
                end_line=getattr(node, "end_lineno", node.lineno),
            ),
            called_functions=list(dict.fromkeys(called_funcs)),
            is_method=is_method,
        )

    def extract_all(self, parsed_files: List[ParsedSourceFile]) -> List[FileStructuralContext]:
        """Extract structures across all provided files, cross-resolving known modules."""
        known_modules = {p.filename.replace(".py", "") for p in parsed_files}
        return [self.extract_structure(p, known_modules=known_modules) for p in parsed_files]
