"""
Unit tests for AgentIdentifier and framework pattern recognition.
"""

import tempfile
import unittest
from pathlib import Path

from ingestion.python_ast_parser import PythonASTParser
from ingestion.structural_extractor import StructuralExtractor
from ingestion.agent_identifier import AgentIdentifier
from schemas.evidence_schema import EvidenceStatus


class TestAgentIdentification(unittest.TestCase):
    def setUp(self):
        self.parser = PythonASTParser()
        self.extractor = StructuralExtractor(self.parser)
        self.identifier = AgentIdentifier(self.parser)

    def test_crewai_agent_constructor_detection(self):
        """Verify CrewAI Agent(...) constructor extraction with role, goal, backstory."""
        code = """
from crewai import Agent

AGENT_ID = "invoice_extraction_agent"

def create_agent(llm=None):
    return Agent(
        role="Senior Extraction Specialist",
        goal="Accurately extract all structured invoice fields.",
        backstory="An expert document auditor parsing accounts payable.",
        llm=llm,
        verbose=False,
        allow_delegation=False
    )
"""
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".py") as f:
            f.write(code)
            f_path = Path(f.name)

        try:
            parsed = self.parser.parse_file(f_path)
            ctx = self.extractor.extract_structure(parsed)
            agents = self.identifier.identify_agents(ctx)

            self.assertEqual(len(agents), 1)
            agent = agents[0]
            self.assertEqual(agent.agent_id, "invoice_extraction_agent")
            self.assertEqual(agent.role, "Senior Extraction Specialist")
            self.assertIn("structured invoice", agent.goal)
            self.assertIn("document auditor", agent.backstory)
            self.assertEqual(agent.framework, "CrewAI")
            self.assertEqual(agent.definition_type, "constructor_call")
            self.assertEqual(agent.evidence_status, EvidenceStatus.EXPLICIT)
        finally:
            f_path.unlink(missing_ok=True)

    def test_no_false_positive_agents(self):
        """Verify helper classes or functions with 'agent' in their name are not falsely classified as agents."""
        code = """
class AgentDatabaseHelper:
    '''Utility for querying agent logs.'''
    def get_agent_record(self, agent_id):
        return {"id": agent_id, "status": "active"}

def run_agent_monitor():
    return "Monitoring active agents"
"""
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".py") as f:
            f.write(code)
            f_path = Path(f.name)

        try:
            parsed = self.parser.parse_file(f_path)
            ctx = self.extractor.extract_structure(parsed)
            agents = self.identifier.identify_agents(ctx)

            # Neither AgentDatabaseHelper nor run_agent_monitor construct an agent or subclass Agent
            self.assertEqual(len(agents), 0, "Non-agent classes/functions must not be classified as agents")
        finally:
            f_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()

