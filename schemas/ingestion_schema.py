"""
Ingestion-level output schemas for Level 1 (Index) and Summary reporting.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


@dataclass
class CompactAgentEntry:
    """Level 1: Ultra-compact agent index entry designed for low token context."""
    agent_id: str
    role: Optional[str]
    goal: Optional[str]
    framework: Optional[str]
    model: Optional[str]
    registered_tools: List[str]
    inputs: List[str]
    expected_outputs: List[str]
    helper_functions: List[str]
    source_file: str
    line_range: str
    evidence_ids: List[str]
    unresolved_references: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TokenMetrics:
    """Token calculation metrics using tokenizer where supported with explicit fallback labeling."""
    original_source_tokens: int
    compact_index_tokens: int
    agent_details_tokens: int
    code_evidence_tokens: int
    combined_tokens: int
    token_reduction_count: int
    percentage_reduction: float
    token_measurement_method: str
    is_actual_tokenizer: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IngestionSummary:
    """Extraction summary recording files, agents, unresolved items, and metrics."""
    files_analyzed: List[str]
    agents_identified: int
    evidence_items_count: int
    parsing_errors: List[Dict[str, str]]
    unresolved_references: List[str]
    external_dependencies: List[str]
    token_metrics: TokenMetrics

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["token_metrics"] = self.token_metrics.to_dict()
        return d
