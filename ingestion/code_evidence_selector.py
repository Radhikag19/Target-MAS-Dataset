"""
Stage 5: AST-Guided Relevant Code Selection.

Selects precise implementation snippets using AST node boundaries (zero arbitrary truncation).
Traces from agent initializations and execution methods to their called helper functions,
recording truly unresolved external references only when an identifier has no local definition,
no import statement, and is not a Python builtin.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple

from ingestion.python_ast_parser import PythonASTParser
from ingestion.structural_extractor import FileStructuralContext, STANDARD_BUILTINS
from schemas.evidence_schema import EvidenceStatus, EvidenceItem


class CodeEvidenceSelector:
    """Selects high-signal, AST-bounded code snippets for retrievable evidence."""

    def __init__(self, parser: Optional[PythonASTParser] = None):
        self.parser = parser or PythonASTParser()

    def select_evidence(
        self,
        ctx: FileStructuralContext,
        agent_id: str,
        all_contexts: Optional[Dict[str, FileStructuralContext]] = None,
    ) -> Tuple[List[EvidenceItem], List[str]]:
        """Select AST-bounded evidence snippets and track truly unresolved identifiers."""
        evidence_items: List[EvidenceItem] = []
        unresolved_deps: List[str] = []
        source = ctx.source
        filename = ctx.filename
        tree = ctx.parsed_file.tree
        if not tree:
            return evidence_items, unresolved_deps

        contexts_by_mod = all_contexts or {}

        # 1. Agent Initialization / Factory Function Snippet
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if any(k in node.name.lower() for k in ["create_agent", "create_", "_agent"]):
                    snippet = self.parser.node_source(node, source)
                    ev_id = f"ev_{agent_id}_agent_init"
                    evidence_items.append(EvidenceItem(
                        evidence_id=ev_id,
                        entity_type="agent_init",
                        entity_id=agent_id,
                        file=filename,
                        start_line=node.lineno,
                        end_line=getattr(node, "end_lineno", node.lineno),
                        status=EvidenceStatus.EXPLICIT,
                        snippet=snippet,
                        description=f"Agent initialization and configuration function '{node.name}'",
                    ))

        # 2. Task Construction / Prompt Execution Function Snippet
        called_helpers: Set[str] = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if any(k in node.name.lower() for k in ["create_task", "run_", "execute", "_task"]):
                    snippet = self.parser.node_source(node, source)
                    ev_id = f"ev_{agent_id}_task_exec_{node.name}"
                    evidence_items.append(EvidenceItem(
                        evidence_id=ev_id,
                        entity_type="task_def",
                        entity_id=agent_id,
                        file=filename,
                        start_line=node.lineno,
                        end_line=getattr(node, "end_lineno", node.lineno),
                        status=EvidenceStatus.EXPLICIT,
                        snippet=snippet,
                        description=f"Task construction, prompt instructions and execution function '{node.name}'",
                    ))

                    # Track called functions inside this execution method
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call):
                            call_name = self.parser.dotted_name(child.func)
                            if call_name:
                                called_helpers.add(call_name.split(".")[-1])

        # 3. Trace and Capture Called Local Helper Functions (e.g. deterministic_arithmetic_check)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in called_helpers and not any(ev.start_line == node.lineno for ev in evidence_items):
                    snippet = self.parser.node_source(node, source)
                    ev_id = f"ev_{agent_id}_helper_{node.name}"
                    evidence_items.append(EvidenceItem(
                        evidence_id=ev_id,
                        entity_type="helper_func",
                        entity_id=agent_id,
                        file=filename,
                        start_line=node.lineno,
                        end_line=getattr(node, "end_lineno", node.lineno),
                        status=EvidenceStatus.STATICALLY_RESOLVED,
                        snippet=snippet,
                        description=f"Direct helper function '{node.name}' called by agent execution",
                    ))

        # 4. Custom Agent Class Snippets
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                if any(k in node.name.lower() for k in ["agent", "specialist", "auditor"]):
                    snippet = self.parser.node_source(node, source)
                    ev_id = f"ev_{agent_id}_class_{node.name}"
                    evidence_items.append(EvidenceItem(
                        evidence_id=ev_id,
                        entity_type="custom_class",
                        entity_id=agent_id,
                        file=filename,
                        start_line=node.lineno,
                        end_line=getattr(node, "end_lineno", node.lineno),
                        status=EvidenceStatus.EXPLICIT,
                        snippet=snippet,
                        description=f"Custom agent class definition '{node.name}'",
                    ))

        # 5. Check for Truly Unresolved Helper References
        # An identifier is only unresolved if it is NOT defined locally, NOT imported from any module, and NOT a builtin
        local_fn_names = {f.name for f in ctx.functions}
        imported_names = set(ctx.imported_symbols.keys())
        top_constants = set(ctx.top_level_constants.keys())

        for helper in called_helpers:
            is_local = (helper in local_fn_names) or (helper in top_constants)
            is_imported = (helper in imported_names)
            is_builtin = (helper in STANDARD_BUILTINS) or helper.startswith("_")

            if not (is_local or is_imported or is_builtin):
                unresolved_deps.append(helper)

        # Deduplicate evidence items
        deduped: List[EvidenceItem] = []
        seen = set()
        for ev in evidence_items:
            if ev.evidence_id not in seen:
                seen.add(ev.evidence_id)
                deduped.append(ev)

        return deduped, list(dict.fromkeys(unresolved_deps))
