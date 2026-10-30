"""
Stage 3 & 4: MAS-Specific Entity Identification & Framework Recognition.

Identifies framework Agent constructors (CrewAI Agent, create_crew_agent),
custom agent classes, and decorated agent functions.
Uses multiple structural indicators to prevent false classifications.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from ingestion.python_ast_parser import PythonASTParser
from ingestion.structural_extractor import FileStructuralContext
from schemas.evidence_schema import EvidenceStatus, SourceReference
from schemas.agent_schema import AgentDefinition


class AgentIdentifier:
    """Identifies and extracts Agent entities from structural AST contexts."""

    def __init__(self, parser: Optional[PythonASTParser] = None):
        self.parser = parser or PythonASTParser()

    def identify_agents(self, ctx: FileStructuralContext) -> List[AgentDefinition]:
        """Identify all verified agent entities in a source file context."""
        agents: List[AgentDefinition] = []
        if not ctx.parsed_file.is_valid:
            return agents

        source = ctx.source
        tree = ctx.parsed_file.tree
        filename = ctx.filename
        top_constants = ctx.top_level_constants

        # Agent ID candidates from top-level constants
        declared_agent_id = top_constants.get("AGENT_ID")

        # 1. Framework-defined Agent(...) constructor calls
        for call_info in ctx.constructor_calls:
            leaf = call_info["leaf"]
            kwargs = call_info["kwargs"]
            node = call_info["node"]

            # Structural indicators for an Agent constructor
            is_crewai_agent = (leaf == "Agent")
            is_factory_crew_agent = (leaf == "create_crew_agent")
            has_role_and_goal = ("role" in kwargs or "goal" in kwargs or "backstory" in kwargs)

            if is_crewai_agent or (is_factory_crew_agent and has_role_and_goal) or (has_role_and_goal and "llm" in kwargs):
                # Resolve role, goal, backstory
                role = self._resolve_constant_or_literal(kwargs.get("role"), top_constants)
                goal = self._resolve_constant_or_literal(kwargs.get("goal"), top_constants)
                backstory = self._resolve_constant_or_literal(kwargs.get("backstory"), top_constants)

                agent_id = declared_agent_id
                if not agent_id:
                    # Statically infer agent ID from factory function name or filename
                    agent_id = self._infer_agent_id_from_context(ctx, call_info)

                agent_name = role or agent_id.replace("_", " ").title()

                # Configuration parameters
                config = {}
                for key in ["verbose", "allow_delegation", "memory", "max_iter"]:
                    if key in kwargs:
                        config[key] = kwargs[key]

                # Relevant methods in the file associated with this agent
                relevant_methods = [f.name for f in ctx.functions]

                agent_def = AgentDefinition(
                    agent_id=str(agent_id),
                    agent_name=str(agent_name),
                    definition_type="constructor_call",
                    framework="CrewAI" if (is_crewai_agent or is_factory_crew_agent) else "Custom",
                    role=role,
                    goal=goal,
                    backstory=backstory,
                    configuration=config,
                    relevant_methods=relevant_methods,
                    source_ref=SourceReference(
                        file=filename,
                        start_line=call_info["line"],
                        end_line=call_info["end_line"],
                    ),
                    extraction_method="crewai_agent_constructor_visitor",
                    evidence_status=EvidenceStatus.EXPLICIT,
                )
                agents.append(agent_def)

        # 2. Custom Agent Classes (Subclasses or decorated classes)
        for cls_def in ctx.classes:
            is_agent_subclass = any("agent" in base.lower() for base in cls_def.bases)
            is_agent_decorated = any("agent" in dec.lower() for dec in cls_def.decorators)
            has_agent_role_attr = False

            if is_agent_subclass or is_agent_decorated or has_agent_role_attr:
                agent_id = declared_agent_id or cls_def.name
                agent_name = cls_def.name
                relevant_methods = [m.name for m in cls_def.methods]

                agent_def = AgentDefinition(
                    agent_id=str(agent_id),
                    agent_name=agent_name,
                    definition_type="custom_class",
                    framework="CrewAI" if is_agent_subclass else "Custom",
                    role=cls_def.docstring,
                    goal=None,
                    backstory=None,
                    configuration={},
                    relevant_methods=relevant_methods,
                    source_ref=cls_def.source_ref,
                    extraction_method="custom_class_inheritance_visitor",
                    evidence_status=EvidenceStatus.STATICALLY_RESOLVED if is_agent_subclass else EvidenceStatus.HEURISTIC,
                )
                agents.append(agent_def)

        # Deduplicate agents by agent_id in the same file
        deduped: List[AgentDefinition] = []
        seen_ids = set()
        for a in agents:
            if a.agent_id not in seen_ids:
                seen_ids.add(a.agent_id)
                deduped.append(a)

        return deduped

    def _resolve_constant_or_literal(self, value: Any, top_constants: Dict[str, Any]) -> Optional[str]:
        """Resolve string literal or top-level variable reference like ROLE or GOAL."""
        if value is None:
            return None
        if isinstance(value, str):
            # Check if this string is the name of a top-level constant (e.g. ROLE)
            if value in top_constants and isinstance(top_constants[value], str):
                return top_constants[value]
            return value
        return str(value)

    def _infer_agent_id_from_context(self, ctx: FileStructuralContext, call_info: Dict[str, Any]) -> str:
        """Infer agent ID based on enclosing function or filename without fabricating information."""
        # Check if caller function has a name like create_extraction_agent
        call_line = call_info["line"]
        for fn in ctx.functions:
            if fn.source_ref and fn.source_ref.start_line <= call_line <= fn.source_ref.end_line:
                if fn.name.startswith("create_") and fn.name.endswith("_agent"):
                    return fn.name.replace("create_", "")
                return fn.name

        # Fallback to filename stem
        return ctx.parsed_file.filename.replace(".py", "")

