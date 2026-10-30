"""
Entity schemas for structural, model, prompt, tool, and execution representations.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from schemas.evidence_schema import EvidenceStatus, SourceReference


@dataclass
class ImportReference:
    """Import statement extracted from source AST."""
    module: str
    names: List[str] = field(default_factory=list)
    aliases: Dict[str, str] = field(default_factory=dict)
    level: int = 0
    line: int = 0
    is_relative: bool = False
    status: EvidenceStatus = EvidenceStatus.STATICALLY_RESOLVED

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class FunctionDefinition:
    """Function or method definition extracted from AST."""
    name: str
    args: List[str] = field(default_factory=list)
    defaults: List[str] = field(default_factory=list)
    returns: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    source_ref: Optional[SourceReference] = None
    called_functions: List[str] = field(default_factory=list)
    is_method: bool = False
    evidence_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClassDefinition:
    """Class definition extracted from AST."""
    name: str
    bases: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    methods: List[FunctionDefinition] = field(default_factory=list)
    docstring: Optional[str] = None
    source_ref: Optional[SourceReference] = None
    evidence_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelReference:
    """Model client initialization and direct invocation reference."""
    provider: Optional[str] = None
    model_name: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    init_call: Optional[str] = None
    direct_llm_invocations: List[str] = field(default_factory=list)  # e.g., llm.invoke(), client.chat.completions.create
    source_ref: Optional[SourceReference] = None
    status: EvidenceStatus = EvidenceStatus.STATICALLY_RESOLVED

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class ExecutionPattern:
    """Distinct execution categories separating LLM, agent, workflow, and tool calls."""
    direct_llm_calls: List[str] = field(default_factory=list)        # llm.invoke, llm.generate, etc.
    agent_execution_calls: List[str] = field(default_factory=list)   # run_agent_task, agent.execute_task
    workflow_execution_calls: List[str] = field(default_factory=list)# crew.kickoff, workflow.run
    tool_execution_calls: List[str] = field(default_factory=list)    # deterministic_arithmetic_check, tool.run

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PromptTemplate:
    """Prompt template, task instruction, or system prompt."""
    prompt_type: str                   # "task_instruction", "system_prompt", "user_prompt", "template"
    template_text: str
    placeholders: List[str] = field(default_factory=list)
    source_ref: Optional[SourceReference] = None
    status: EvidenceStatus = EvidenceStatus.EXPLICIT
    evidence_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class ToolReference:
    """Explicitly registered tool reference (decorated with @tool, BaseTool subclass, or passed in tools=[...])."""
    name: str
    registration_type: str             # "decorated_tool", "basetool_subclass", "registered_tool_argument"
    parameters: List[str] = field(default_factory=list)
    decorator: Optional[str] = None
    docstring: Optional[str] = None
    source_ref: Optional[SourceReference] = None
    status: EvidenceStatus = EvidenceStatus.EXPLICIT
    evidence_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d
