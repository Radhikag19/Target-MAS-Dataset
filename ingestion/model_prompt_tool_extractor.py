"""
Stage 3 (Corrected): Model, Prompt, Tool, and I/O Extraction.

Correctly distinguishes:
1. Agent inputs, task expected outputs, and return values (filtering out framework constructor returns).
2. Model initialization, direct LLM invocations, agent execution, workflow execution, and tool execution.
3. Explicitly registered tools (@tool, BaseTool, tools=[...]) from ordinary helper functions.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple

from ingestion.python_ast_parser import PythonASTParser
from ingestion.structural_extractor import FileStructuralContext
from schemas.evidence_schema import EvidenceStatus, SourceReference
from schemas.entity_schema import (
    ModelReference,
    ExecutionPattern,
    PromptTemplate,
    ToolReference,
)


class ModelPromptToolExtractor:
    """Extracts models, prompts, tools, and I/O representations with strict evidence separation."""

    def __init__(self, parser: Optional[PythonASTParser] = None):
        self.parser = parser or PythonASTParser()

    # -------------------------------------------------------------------------
    # 1. Execution Patterns & Model Separation
    # -------------------------------------------------------------------------

    def extract_execution_patterns(self, ctx: FileStructuralContext) -> ExecutionPattern:
        """Categorize execution calls into direct LLM, agent, workflow, and tool calls."""
        direct_llm: List[str] = []
        agent_exec: List[str] = []
        workflow_exec: List[str] = []
        tool_exec: List[str] = []

        for call_info in ctx.function_calls:
            call_name = call_info["call_name"]
            leaf = call_info["leaf"]
            line = call_info["line"]

            # Direct LLM calls: calls directly invoking an LLM client
            if leaf in {"invoke", "generate", "predict", "create"} and any(
                term in call_name.lower() for term in ["llm", "chat", "client.chat", "completions"]
            ):
                direct_llm.append(f"{call_name} (line {line})")

            # Workflow execution: multi-agent / crew level execution
            elif leaf in {"kickoff", "kickoff_async"}:
                workflow_exec.append(f"{call_name} (line {line})")

            # Agent execution: single agent task execution wrappers
            elif leaf in {"run_agent_task", "execute_task"}:
                agent_exec.append(f"{call_name} (line {line})")

            # Tool / Helper execution
            elif leaf.startswith("deterministic_") or leaf in {"check", "validate", "calculate"}:
                tool_exec.append(f"{call_name} (line {line})")

        return ExecutionPattern(
            direct_llm_calls=list(dict.fromkeys(direct_llm)),
            agent_execution_calls=list(dict.fromkeys(agent_exec)),
            workflow_execution_calls=list(dict.fromkeys(workflow_exec)),
            tool_execution_calls=list(dict.fromkeys(tool_exec)),
        )

    def extract_models(self, ctx: FileStructuralContext) -> List[ModelReference]:
        """Extract model client initialization and direct invocation references."""
        models: List[ModelReference] = []
        filename = ctx.filename
        exec_pattern = self.extract_execution_patterns(ctx)

        # Model initializations
        for call_info in ctx.function_calls:
            call_name = call_info["call_name"]
            leaf = call_info["leaf"]
            kwargs = call_info["kwargs"]

            is_model_factory = any(k in call_name.lower() for k in [
                "create_groq_llm", "create_agent_llm", "chatopenai", "chatgroq"
            ])

            if is_model_factory:
                provider = "Groq" if "groq" in call_name.lower() else "OpenAI"
                model_name = None
                if call_info["args"]:
                    model_name = str(call_info["args"][0])
                elif "model" in kwargs:
                    model_name = str(kwargs["model"])
                elif "model_name" in kwargs:
                    model_name = str(kwargs["model_name"])

                model_ref = ModelReference(
                    provider=provider,
                    model_name=model_name,
                    parameters=kwargs,
                    init_call=call_name,
                    direct_llm_invocations=exec_pattern.direct_llm_calls,
                    source_ref=SourceReference(
                        file=filename,
                        start_line=call_info["line"],
                        end_line=call_info["end_line"],
                    ),
                    status=EvidenceStatus.STATICALLY_RESOLVED,
                )
                models.append(model_ref)

        # Fallback when llm is passed as parameter to create_agent(llm=None)
        if not models:
            for fn in ctx.functions:
                if "llm" in fn.args and fn.name.startswith("create_"):
                    models.append(ModelReference(
                        provider="Configurable via Parameter",
                        model_name="llm",
                        parameters={"parameter_name": "llm"},
                        init_call=f"{fn.name}(llm=...)",
                        direct_llm_invocations=exec_pattern.direct_llm_calls,
                        source_ref=fn.source_ref,
                        status=EvidenceStatus.EXPLICIT,
                    ))

        return models

    # -------------------------------------------------------------------------
    # 2. Prompts
    # -------------------------------------------------------------------------

    def extract_prompts(self, ctx: FileStructuralContext) -> List[PromptTemplate]:
        """Extract prompt templates, task descriptions, and dynamic placeholders."""
        prompts: List[PromptTemplate] = []
        source = ctx.source
        filename = ctx.filename
        tree = ctx.parsed_file.tree
        if not tree:
            return prompts

        # Local variable assignments mapping
        local_strings: Dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        val_str = self.parser.literal_value(node.value)
                        if isinstance(val_str, str):
                            local_strings[t.id] = val_str

        # A. Prompts inside Task(description=...)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call_name = self.parser.dotted_name(node.func)
                if call_name.split(".")[-1] == "Task":
                    kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg}
                    desc_node = kwargs.get("description")
                    if desc_node:
                        desc_src = self.parser.node_source(desc_node, source)
                        # If description was a variable reference, check local string
                        if isinstance(desc_node, ast.Name) and desc_node.id in local_strings:
                            desc_src = local_strings[desc_node.id]

                        placeholders = self._extract_placeholders(desc_src)
                        prompts.append(PromptTemplate(
                            prompt_type="task_instruction",
                            template_text=desc_src[:1500],
                            placeholders=placeholders,
                            source_ref=SourceReference(
                                file=filename,
                                start_line=node.lineno,
                                end_line=getattr(node, "end_lineno", node.lineno),
                            ),
                            status=EvidenceStatus.EXPLICIT,
                        ))

        # B. Explicit prompt variables (e.g. description = ..., prompt = ...)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        name_lower = t.id.lower()
                        if any(k in name_lower for k in ["prompt", "task_desc", "instruction", "system_msg"]):
                            val_src = self.parser.node_source(node.value, source)
                            if len(val_src.strip()) >= 30:
                                placeholders = self._extract_placeholders(val_src)
                                prompts.append(PromptTemplate(
                                    prompt_type="template",
                                    template_text=val_src[:1500],
                                    placeholders=placeholders,
                                    source_ref=SourceReference(
                                        file=filename,
                                        start_line=node.lineno,
                                        end_line=getattr(node, "end_lineno", node.lineno),
                                    ),
                                    status=EvidenceStatus.STATICALLY_RESOLVED,
                                ))

        # Deduplicate
        deduped: List[PromptTemplate] = []
        seen = set()
        for p in prompts:
            key = (p.template_text[:100], p.source_ref.start_line if p.source_ref else 0)
            if key not in seen:
                seen.add(key)
                deduped.append(p)

        return deduped

    def _extract_placeholders(self, text: str) -> List[str]:
        """Extract dynamic curly-brace placeholders like {invoice_data} or {supplier_id}."""
        raw_placeholders = re.findall(r"\{([a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)*)\}", text)
        return list(dict.fromkeys(raw_placeholders))

    # -------------------------------------------------------------------------
    # 3. Tools vs Helper Functions
    # -------------------------------------------------------------------------

    def extract_tools_and_helpers(self, ctx: FileStructuralContext) -> Tuple[List[ToolReference], List[str]]:
        """
        Distinguish explicitly registered tools from ordinary helper functions.
        Registered tools require:
          - @tool decorator
          - BaseTool subclass
          - Explicit registration in Agent(tools=[...]) or Task(tools=[...])
        Ordinary helper functions (e.g. deterministic_arithmetic_check) are returned separately.
        """
        registered_tools: List[ToolReference] = []
        helper_functions: List[str] = []
        filename = ctx.filename

        # A. Inspect functions for @tool decorator
        for fn in ctx.functions:
            is_decorated_tool = any("tool" in dec.lower() for dec in fn.decorators)
            if is_decorated_tool:
                registered_tools.append(ToolReference(
                    name=fn.name,
                    registration_type="decorated_tool",
                    parameters=fn.args,
                    decorator=fn.decorators[0] if fn.decorators else None,
                    docstring=fn.docstring,
                    source_ref=fn.source_ref,
                    status=EvidenceStatus.EXPLICIT,
                ))
            elif fn.name.startswith("deterministic_") or not fn.name.startswith("create_"):
                # Ordinary helper function
                if fn.name not in {"run", "run_workflow"}:
                    helper_functions.append(fn.name)

        # B. Inspect classes for BaseTool inheritance
        for cls_def in ctx.classes:
            if any("basetool" in base.lower() or "tool" == base.lower() for base in cls_def.bases):
                registered_tools.append(ToolReference(
                    name=cls_def.name,
                    registration_type="basetool_subclass",
                    parameters=[],
                    docstring=cls_def.docstring,
                    source_ref=cls_def.source_ref,
                    status=EvidenceStatus.EXPLICIT,
                ))

        # C. Inspect Agent(tools=[...]) and Task(tools=[...]) arguments
        for call_info in ctx.function_calls:
            leaf = call_info["leaf"]
            kwargs = call_info["kwargs"]
            if leaf in {"Agent", "Task"} and "tools" in kwargs:
                tools_val = kwargs["tools"]
                if isinstance(tools_val, list):
                    for item in tools_val:
                        registered_tools.append(ToolReference(
                            name=str(item),
                            registration_type="registered_tool_argument",
                            parameters=[],
                            source_ref=SourceReference(
                                file=filename,
                                start_line=call_info["line"],
                                end_line=call_info["end_line"],
                            ),
                            status=EvidenceStatus.EXPLICIT,
                        ))

        # Deduplicate
        deduped_tools: List[ToolReference] = []
        seen = set()
        for t in registered_tools:
            if t.name not in seen:
                seen.add(t.name)
                deduped_tools.append(t)

        return deduped_tools, list(dict.fromkeys(helper_functions))

    # -------------------------------------------------------------------------
    # 4. Correctly Distinguish Inputs, Task Expected Outputs, and Return Schemas
    # -------------------------------------------------------------------------

    def extract_inputs_and_outputs(
        self, ctx: FileStructuralContext
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Correctly distinguish:
        - agent data inputs (parameters of task/execution methods, excluding self/agent/llm)
        - task expected outputs (string contracts from Task(expected_output=...))
        - return data schemas (excluding Agent/Task/Crew framework constructor instances)
        """
        inputs: List[Dict[str, Any]] = []
        task_expected_outputs: List[Dict[str, Any]] = []
        return_schemas: List[Dict[str, Any]] = []

        source = ctx.source
        tree = ctx.parsed_file.tree

        # Local variable assignment map to resolve variable identifiers like expected_output = "..."
        local_strings: Dict[str, str] = {}
        if tree:
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            val_str = self.parser.literal_value(node.value)
                            if isinstance(val_str, str):
                                local_strings[t.id] = val_str

        # 1. Inputs: Extract business data arguments from task creation or execution functions
        framework_args = {"self", "agent", "llm", "tracker", "verbose", "memory", "kwargs"}
        for fn in ctx.functions:
            if any(k in fn.name.lower() for k in ["create_task", "run_", "execute_task"]):
                for arg in fn.args:
                    if arg not in framework_args:
                        inputs.append({
                            "name": arg,
                            "function": fn.name,
                            "line": fn.source_ref.start_line if fn.source_ref else 0,
                            "status": EvidenceStatus.EXPLICIT.value,
                        })

        # 2. Task Expected Outputs from Task(expected_output=...)
        if tree:
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    call_name = self.parser.dotted_name(node.func)
                    if call_name.split(".")[-1] == "Task":
                        kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg}
                        exp_node = kwargs.get("expected_output")
                        if exp_node:
                            resolved_output = None
                            # Literal string
                            lit = self.parser.literal_value(exp_node)
                            if isinstance(lit, str):
                                resolved_output = lit
                            # Local variable reference
                            elif isinstance(exp_node, ast.Name) and exp_node.id in local_strings:
                                resolved_output = local_strings[exp_node.id]
                            # Variable expression
                            elif isinstance(exp_node, ast.Name):
                                resolved_output = f"variable: {exp_node.id}"

                            if resolved_output:
                                task_expected_outputs.append({
                                    "type": "task_expected_output",
                                    "description": resolved_output,
                                    "line": node.lineno,
                                    "status": EvidenceStatus.EXPLICIT.value,
                                })

        # 3. Return Data Schemas (Strictly exclude framework constructors like Agent/Task/Crew)
        for fn in ctx.functions:
            # Skip factory functions whose sole purpose is instantiating Agent/Task/Crew objects
            if fn.returns in {"<Agent instance>", "<Task instance>", "<Crew instance>"}:
                continue
            if fn.name.startswith("create_"):
                continue

            if fn.returns:
                return_schemas.append({
                    "function": fn.name,
                    "return_type": fn.returns,
                    "line": fn.source_ref.end_line if fn.source_ref else 0,
                    "status": EvidenceStatus.STATICALLY_RESOLVED.value,
                })

        # Deduplicate
        dedup_inputs = []
        seen_in = set()
        for i in inputs:
            if i["name"] not in seen_in:
                seen_in.add(i["name"])
                dedup_inputs.append(i)

        dedup_outputs = []
        seen_out = set()
        for o in task_expected_outputs:
            if o["description"] not in seen_out:
                seen_out.add(o["description"])
                dedup_outputs.append(o)

        return dedup_inputs, dedup_outputs, return_schemas
