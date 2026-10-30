"""
Agent schema for detailed, normalized representations.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from schemas.evidence_schema import EvidenceStatus, SourceReference
from schemas.entity_schema import (
    ModelReference,
    PromptTemplate,
    ToolReference,
    ExecutionPattern,
)


@dataclass
class AgentDefinition:
    """Detailed structural representation of an identified agent."""
    agent_id: str
    agent_name: str
    definition_type: str               # "constructor_call", "custom_class", "decorated_function"
    framework: Optional[str] = "CrewAI"# "CrewAI", "Custom", or None
    role: Optional[str] = None
    goal: Optional[str] = None
    backstory: Optional[str] = None
    configuration: Dict[str, Any] = field(default_factory=dict)
    relevant_methods: List[str] = field(default_factory=list)
    model_references: List[ModelReference] = field(default_factory=list)
    prompt_references: List[PromptTemplate] = field(default_factory=list)
    tool_references: List[ToolReference] = field(default_factory=list)  # Strictly registered tools only
    helper_functions: List[str] = field(default_factory=list)          # Ordinary helper functions like deterministic_*
    execution_pattern: Optional[ExecutionPattern] = None               # Categorized execution calls
    inputs: List[Dict[str, Any]] = field(default_factory=list)         # Agent/Task data inputs
    task_expected_outputs: List[Dict[str, Any]] = field(default_factory=list) # Expected task output schemas
    return_schemas: List[Dict[str, Any]] = field(default_factory=list) # Explicit return data schemas
    source_ref: Optional[SourceReference] = None
    evidence_ids: List[str] = field(default_factory=list)
    extraction_method: str = "ast_visitor_pattern"
    evidence_status: EvidenceStatus = EvidenceStatus.EXPLICIT
    unresolved_references: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["evidence_status"] = self.evidence_status.value
        d["model_references"] = [m.to_dict() for m in self.model_references]
        d["prompt_references"] = [p.to_dict() for p in self.prompt_references]
        d["tool_references"] = [t.to_dict() for t in self.tool_references]
        if self.execution_pattern:
            d["execution_pattern"] = self.execution_pattern.to_dict()
        return d
