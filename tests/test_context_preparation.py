"""
Unit tests for Context Preparation, Two-Level Architecture, and Token Measurement.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from ingestion.ingestion_orchestrator import IngestionOrchestrator
from ingestion.context_preparer import TraceProfilerContextPreparer
from schemas.agent_schema import AgentDefinition
from schemas.evidence_schema import EvidenceItem, EvidenceStatus, SourceReference
from schemas.entity_schema import ModelReference, ToolReference


class TestContextPreparation(unittest.TestCase):
    def test_context_preparer_lookups(self):
        """Verify deterministic Python lookup methods without vector DBs or embeddings."""
        agent_index = [
            {
                "agent_id": "test_agent_1",
                "role": "Extractor",
                "goal": "Extract invoice fields",
                "definition_type": "constructor_call",
                "framework": "CrewAI",
                "model_reference": "Groq:openai/gpt-oss-120b",
                "tool_references": ["math_check"],
                "inputs": ["invoice_data"],
                "outputs": ["extracted_invoice"],
                "relevant_methods": ["create_agent", "create_task"],
                "source_file": "agent_1.py",
                "line_range": "L10-L30",
                "evidence_ids": ["ev_1", "ev_2"],
                "unresolved_references": ["external_lib"],
            }
        ]

        agent_details = [
            {
                "agent_id": "test_agent_1",
                "agent_name": "Extractor",
                "definition_type": "constructor_call",
                "framework": "CrewAI",
                "role": "Extractor",
                "goal": "Extract invoice fields",
                "configuration": {"verbose": False},
                "unresolved_references": ["external_lib"],
            }
        ]

        code_evidence = {
            "ev_1": {
                "evidence_id": "ev_1",
                "entity_type": "agent_init",
                "entity_id": "test_agent_1",
                "file": "agent_1.py",
                "start_line": 10,
                "end_line": 20,
                "status": "EXPLICIT",
                "snippet": "def create_agent(): ...",
                "description": "Agent initialization function 'create_agent'",
            },
            "ev_2": {
                "evidence_id": "ev_2",
                "entity_type": "helper_func",
                "entity_id": "test_agent_1",
                "file": "agent_1.py",
                "start_line": 22,
                "end_line": 30,
                "status": "STATICALLY_RESOLVED",
                "snippet": "def math_check(): ...",
                "description": "Direct helper function 'math_check'",
            },
        }

        preparer = TraceProfilerContextPreparer(agent_index, agent_details, code_evidence)

        # Lookup agent in Level 1
        agent_entry = preparer.get_agent("test_agent_1")
        self.assertIsNotNone(agent_entry)
        self.assertEqual(agent_entry["role"], "Extractor")

        # Lookup Level 2 details
        details = preparer.get_agent_details("test_agent_1")
        self.assertIsNotNone(details)
        self.assertEqual(details["configuration"], {"verbose": False})

        # Lookup evidence by ID
        ev1 = preparer.get_evidence("ev_1")
        self.assertIsNotNone(ev1)
        self.assertEqual(ev1["start_line"], 10)

        # Lookup evidence by agent
        agent_evs = preparer.get_evidence_by_agent("test_agent_1")
        self.assertEqual(len(agent_evs), 2)

        # Lookup evidence by function
        fn_evs = preparer.get_evidence_by_function("math_check")
        self.assertEqual(len(fn_evs), 1)

        # Lookup unresolved references
        unres = preparer.get_unresolved_references()
        self.assertIn("external_lib", unres)

    def test_end_to_end_orchestration_on_real_agent_files(self):
        """Verify pipeline execution on real repository agent files."""
        repo_root = Path(__file__).resolve().parent.parent
        target_files = [
            repo_root / "Agents" / "invoice_processing" / "invoice_extraction_agent.py",
            repo_root / "Agents" / "invoice_processing" / "invoice_validation_agent.py",
        ]

        # Verify files exist
        for f in target_files:
            self.assertTrue(f.exists(), f"Target agent file not found: {f}")

        temp_dir = Path(tempfile.mkdtemp())
        try:
            orchestrator = IngestionOrchestrator()
            preparer, summary = orchestrator.process_files(
                agent_files=target_files,
                output_dir=temp_dir,
            )

            # Check files created
            self.assertTrue((temp_dir / "agent_index.json").exists())
            self.assertTrue((temp_dir / "agent_details.json").exists())
            self.assertTrue((temp_dir / "code_evidence.json").exists())
            self.assertTrue((temp_dir / "extraction_summary.json").exists())

            # Check agents identified
            self.assertEqual(summary["agents_identified"], 2)
            agent_ids = [a["agent_id"] for a in summary["token_metrics"].__class__.__name__ and preparer.get_agent_index()]
            self.assertIn("invoice_extraction_agent", agent_ids)
            self.assertIn("invoice_validation_agent", agent_ids)

            # Check token reduction metrics
            tm = summary["token_metrics"]
            self.assertGreater(tm["original_source_tokens"], 0)
            self.assertGreater(tm["compact_index_tokens"], 0)
            self.assertGreater(tm["token_reduction_count"], 0)
            self.assertGreater(tm["percentage_reduction"], 0.0)
            self.assertIsNotNone(tm["token_measurement_method"])

            # Check that helper function deterministic_arithmetic_check was captured in evidence
            helper_ev = preparer.get_evidence_by_function("deterministic_arithmetic_check")
            self.assertGreater(len(helper_ev), 0)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

