"""
Unit tests for CodeEvidenceSelector and AST boundary preservation.
"""

import tempfile
import unittest
from pathlib import Path

from ingestion.python_ast_parser import PythonASTParser
from ingestion.structural_extractor import StructuralExtractor
from ingestion.code_evidence_selector import CodeEvidenceSelector
from schemas.evidence_schema import EvidenceStatus


class TestCodeEvidenceSelector(unittest.TestCase):
    def setUp(self):
        self.parser = PythonASTParser()
        self.extractor = StructuralExtractor(self.parser)
        self.evidence_selector = CodeEvidenceSelector(self.parser)

    def test_evidence_selection_and_helper_tracing(self):
        """Verify AST boundary preservation, helper function tracing, and unresolved dependencies."""
        code = """
from crewai import Agent, Task
from external_tools import unknown_formatter

AGENT_ID = "invoice_validation_agent"

def deterministic_math_check(data):
    # Important arithmetic calculation
    return data.get("qty", 0) * data.get("price", 0)

def create_agent(llm=None):
    return Agent(role="Auditor", goal="Audit math", backstory="Accountant", llm=llm)

def create_task(agent, invoice_data):
    math_result = deterministic_math_check(invoice_data)
    formatted = unimported_missing_helper(math_result)
    return Task(description=f"Validate {math_result}", expected_output="JSON", agent=agent)
"""
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".py") as f:
            f.write(code)
            f_path = Path(f.name)

        try:
            parsed = self.parser.parse_file(f_path)
            ctx = self.extractor.extract_structure(parsed)
            evidence_items, unresolved = self.evidence_selector.select_evidence(
                ctx, agent_id="invoice_validation_agent"
            )

            # Check that agent_init was captured
            init_items = [ev for ev in evidence_items if ev.entity_type == "agent_init"]
            self.assertEqual(len(init_items), 1)
            self.assertIn("create_agent", init_items[0].snippet)

            # Check that task_def was captured
            task_items = [ev for ev in evidence_items if ev.entity_type == "task_def"]
            self.assertEqual(len(task_items), 1)
            self.assertIn("create_task", task_items[0].snippet)

            # Check that called helper deterministic_math_check was statically traced and captured!
            helper_items = [ev for ev in evidence_items if ev.entity_type == "helper_func"]
            self.assertEqual(len(helper_items), 1)
            self.assertEqual(helper_items[0].status, EvidenceStatus.STATICALLY_RESOLVED)
            self.assertIn("deterministic_math_check", helper_items[0].snippet)

            # Check that unimported helper 'unimported_missing_helper' was flagged as unresolved
            self.assertIn("unimported_missing_helper", unresolved)
        finally:
            f_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()

