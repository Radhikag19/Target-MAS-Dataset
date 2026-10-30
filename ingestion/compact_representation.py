"""
Stage 6 (Corrected): Compact Representation Generation and Tokenizer Measurement.

1. Eliminates duplication across agent_index.json, agent_details.json, and code_evidence.json.
2. Measures tokens using actual tiktoken if available, with explicitly labeled BPE fallback.
3. Keeps Level 1 compact index minimal for low token consumption.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Any, Tuple, Optional
from schemas.agent_schema import AgentDefinition
from schemas.evidence_schema import EvidenceItem
from schemas.ingestion_schema import CompactAgentEntry, TokenMetrics

# Standard GPT subword BPE tokenization regex pattern
_GPT_BPE_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
)


class TokenCounter:
    """Measures tokens using tiktoken if available, falling back to regex BPE with explicit labeling."""

    def __init__(self):
        self._tiktoken_enc = None
        self._is_actual = False
        self._method_label = "fallback_bpe_regex_tokenizer (tiktoken not installed)"

        try:
            import tiktoken
            self._tiktoken_enc = tiktoken.get_encoding("cl100k_base")
            self._is_actual = True
            self._method_label = "actual_tiktoken_cl100k_base"
        except ImportError:
            self._tiktoken_enc = None
            self._is_actual = False
            self._method_label = "fallback_bpe_regex_tokenizer (tiktoken not installed)"

    @property
    def is_actual(self) -> bool:
        return self._is_actual

    @property
    def method_label(self) -> str:
        return self._method_label

    def count_tokens(self, text: str) -> int:
        """Count tokens accurately or via labeled BPE subword regex approximation."""
        if not text:
            return 0
        if self._tiktoken_enc is not None:
            return len(self._tiktoken_enc.encode(text))
        # Deterministic BPE regex split matching subwords and punctuation tokens
        tokens = _GPT_BPE_PATTERN.findall(text)
        return max(1, len(tokens))


class CompactRepresentationBuilder:
    """Builds clean, non-duplicated two-level representations and measures token metrics."""

    def __init__(self):
        self.token_counter = TokenCounter()

    def build_compact_index(self, agents: List[AgentDefinition]) -> List[CompactAgentEntry]:
        """
        Generate Level 1: Ultra-compact agent index.
        Minimizes initial prompt context: excludes code snippets, AST dumps, and duplication.
        """
        index: List[CompactAgentEntry] = []
        for a in agents:
            model_summary = None
            if a.model_references:
                m = a.model_references[0]
                model_summary = f"{m.provider or 'LLM'}:{m.model_name or m.init_call or 'default'}"

            # Only explicitly registered tools
            registered_tools = [t.name for t in a.tool_references]

            # Clean business inputs (parameter names)
            clean_inputs = [str(i["name"]) for i in a.inputs if "name" in i]

            # Clean expected outputs (task output contracts)
            clean_outputs = [str(o["description"]) for o in a.task_expected_outputs if "description" in o]

            line_str = f"L{a.source_ref.start_line}-L{a.source_ref.end_line}" if a.source_ref else "N/A"
            source_file = a.source_ref.file if a.source_ref else "unknown"

            entry = CompactAgentEntry(
                agent_id=a.agent_id,
                role=a.role,
                goal=a.goal,
                framework=a.framework,
                model=model_summary,
                registered_tools=registered_tools,
                inputs=clean_inputs,
                expected_outputs=clean_outputs,
                helper_functions=a.helper_functions,
                source_file=source_file,
                line_range=line_str,
                evidence_ids=a.evidence_ids,
                unresolved_references=a.unresolved_references,
            )
            index.append(entry)
        return index

    def build_agent_details(self, agents: List[AgentDefinition]) -> List[Dict[str, Any]]:
        """
        Generate Level 2: Normalized structural records.
        Contains detailed configuration, prompts, models, and tools without raw code snippets.
        """
        return [a.to_dict() for a in agents]

    def build_code_evidence_store(self, evidence_items: List[EvidenceItem]) -> Dict[str, Dict[str, Any]]:
        """Generate Level 2: Retrievable source snippet repository keyed by evidence_id."""
        store: Dict[str, Dict[str, Any]] = {}
        for ev in evidence_items:
            store[ev.evidence_id] = ev.to_dict()
        return store

    def calculate_token_metrics(
        self,
        original_sources: Dict[str, str],
        compact_index_data: Any,
        agent_details_data: Any,
        code_evidence_data: Any,
    ) -> TokenMetrics:
        """Measure token counts and reduction using tokenizer with explicit method labeling."""
        combined_original_text = "".join(original_sources.values())
        orig_tokens = self.token_counter.count_tokens(combined_original_text)

        compact_str = json.dumps(compact_index_data, indent=2, default=str)
        details_str = json.dumps(agent_details_data, indent=2, default=str)
        evidence_str = json.dumps(code_evidence_data, indent=2, default=str)

        compact_tokens = self.token_counter.count_tokens(compact_str)
        details_tokens = self.token_counter.count_tokens(details_str)
        evidence_tokens = self.token_counter.count_tokens(evidence_str)
        combined_extracted_tokens = compact_tokens + details_tokens + evidence_tokens

        reduction_count = max(0, orig_tokens - compact_tokens)
        pct_reduction = round((reduction_count / orig_tokens * 100), 2) if orig_tokens > 0 else 0.0

        return TokenMetrics(
            original_source_tokens=orig_tokens,
            compact_index_tokens=compact_tokens,
            agent_details_tokens=details_tokens,
            code_evidence_tokens=evidence_tokens,
            combined_tokens=combined_extracted_tokens,
            token_reduction_count=reduction_count,
            percentage_reduction=pct_reduction,
            token_measurement_method=self.token_counter.method_label,
            is_actual_tokenizer=self.token_counter.is_actual,
        )
