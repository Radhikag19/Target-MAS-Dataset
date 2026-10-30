"""
Evidence classification and source provenance schemas for TraceProfiler.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Dict, Any


class EvidenceStatus(str, Enum):
    """Explicit evidence status levels ensuring zero hallucination."""
    EXPLICIT = "EXPLICIT"                      # Directly declared in the supplied source
    STATICALLY_RESOLVED = "STATICALLY_RESOLVED"# Statically traced via AST code references
    HEURISTIC = "HEURISTIC"                    # Pattern-based identification with multiple indicators
    UNRESOLVED = "UNRESOLVED"                  # Referenced in code, but implementation was not supplied
    NOT_FOUND = "NOT_FOUND"                    # Searched for using supported rules, but not identified


@dataclass
class SourceReference:
    """Exact source code reference for provenance."""
    file: str
    start_line: int
    end_line: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceItem:
    """A retrievable code snippet supporting an extracted entity."""
    evidence_id: str
    entity_type: str                   # "agent_init", "task_def", "tool_func", "prompt_template", "helper_func"
    entity_id: str                     # Associated agent_id or entity identifier
    file: str
    start_line: int
    end_line: int
    status: EvidenceStatus
    snippet: str                       # Exact AST node source snippet (never truncated arbitrarily)
    description: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

