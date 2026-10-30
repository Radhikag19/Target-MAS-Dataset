"""
Stage 7: Context Preparation & Deterministic Evidence Retrieval API.

Provides deterministic Python lookup functions for TraceProfiler to consume
Level 1 compact index initially, and selectively retrieve Level 2 detailed evidence
without vector databases, embeddings, or LLM-based search.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union


class TraceProfilerContextPreparer:
    """In-memory deterministic evidence retrieval engine for TraceProfiler."""

    def __init__(
        self,
        agent_index: List[Dict[str, Any]],
        agent_details: List[Dict[str, Any]],
        code_evidence: Dict[str, Dict[str, Any]],
    ):
        self._index = agent_index
        self._details_by_id: Dict[str, Dict[str, Any]] = {
            a["agent_id"]: a for a in agent_details
        }
        self._evidence_by_id: Dict[str, Dict[str, Any]] = code_evidence

        # Secondary deterministic lookup indices
        self._evidence_by_agent: Dict[str, List[Dict[str, Any]]] = {}
        self._evidence_by_file: Dict[str, List[Dict[str, Any]]] = {}
        self._evidence_by_func: Dict[str, List[Dict[str, Any]]] = {}

        for ev_id, ev in self._evidence_by_id.items():
            agent_id = ev.get("entity_id", "")
            file_name = ev.get("file", "")

            self._evidence_by_agent.setdefault(agent_id, []).append(ev)
            self._evidence_by_file.setdefault(file_name, []).append(ev)

            desc = ev.get("description", "")
            if "'" in desc:
                # Extract quoted function name from description
                parts = desc.split("'")
                if len(parts) >= 2:
                    fn_name = parts[1]
                    self._evidence_by_func.setdefault(fn_name, []).append(ev)

    # -------------------------------------------------------------------------
    # Deterministic Lookup API (No Vector DB / No Embeddings)
    # -------------------------------------------------------------------------

    def get_agent_index(self) -> List[Dict[str, Any]]:
        """Retrieve Level 1 compact agent index for initial prompt context."""
        return self._index

    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve compact entry for a specific agent by ID."""
        for entry in self._index:
            if entry.get("agent_id") == agent_id:
                return entry
        return None

    def get_agent_details(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve Level 2 normalized structural record for an agent."""
        return self._details_by_id.get(agent_id)

    def get_evidence(self, evidence_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific code snippet by its unique evidence ID."""
        return self._evidence_by_id.get(evidence_id)

    def get_evidence_by_agent(self, agent_id: str) -> List[Dict[str, Any]]:
        """Retrieve all code evidence snippets supporting an agent."""
        return self._evidence_by_agent.get(agent_id, [])

    def get_evidence_by_file(self, filename: str) -> List[Dict[str, Any]]:
        """Retrieve all code evidence snippets originating from a specific source file."""
        return self._evidence_by_file.get(filename, [])

    def get_evidence_by_function(self, function_name: str) -> List[Dict[str, Any]]:
        """Retrieve code evidence snippets for a specific helper or execution function."""
        return self._evidence_by_func.get(function_name, [])

    def get_unresolved_references(self) -> List[str]:
        """Retrieve all unresolved external dependencies detected during ingestion."""
        unresolved = []
        for a in self._details_by_id.values():
            unresolved.extend(a.get("unresolved_references", []))
        return list(dict.fromkeys(unresolved))

    @classmethod
    def load_from_directory(cls, output_dir: Union[str, Path]) -> TraceProfilerContextPreparer:
        """Instantiate ContextPreparer by reading exported JSON artifacts from directory."""
        path = Path(output_dir)
        index_file = path / "agent_index.json"
        details_file = path / "agent_details.json"
        evidence_file = path / "code_evidence.json"

        with open(index_file, "r", encoding="utf-8") as f:
            agent_index = json.load(f)
        with open(details_file, "r", encoding="utf-8") as f:
            agent_details = json.load(f)
        with open(evidence_file, "r", encoding="utf-8") as f:
            code_evidence = json.load(f)

        return cls(agent_index, agent_details, code_evidence)

