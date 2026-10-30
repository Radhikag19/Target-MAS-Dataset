"""
Targeted Unit Tests for AST Ingestion Pipeline Corrections.

Validates:
1. Distinction of agent inputs, task expected outputs, and return values (no AST constructor leaks).
2. Separation of model initialization, direct LLM invocations, agent execution, workflow execution, and tool execution.
3. Distinction of explicitly registered tools from ordinary helper functions (deterministic_* is not a tool).
4. Accurate unresolved reference classification (imported symbols like Task are not unresolved).
5. Tokenizer-based measurements with clear fallback labeling.
6. Elimination of duplication across agent_index.json, agent_details.json, and code_evidence.json.
"""

import ast
import json
import tempfile
import unittest
from pathlib import Path

from ingestion.python_ast_parser import PythonASTParser
from ingestion.structural_extractor import StructuralExtractor
from ingestion.agent_identifier import AgentIdentifier
from ingestion.model_prompt_tool_extractor import ModelPromptToolExtractor
from ingestion.code_evidence_selector import CodeEvidenceSelector
from ingestion.compact_representation import CompactRepresentationBuilder, TokenCounter
from ingestion.ingestion_orchestrator import IngestionOrchestrator
from schemas.evidence_schema import EvidenceStatus


class TestTargetedCorrections(unittest.TestCase):
    def setUp(self):
        self.parser = PythonASTParser()
        self.structural_extractor = StructuralExtractor(self.parser)
        self.mpt_extractor = ModelPromptToolExtractor(self.parser)
        self.evidence_selector = CodeEvidenceSelector(self.parser)
        self.builder = CompactRepresentationBuilder()

    def test_1_inputs_outputs_and_intermediate_expressions(self):
        """
        Correction 1: Correctly distinguish inputs, return values, task expected outputs.
        Do not extract entire AST constructor expressions as agent outputs.
        """
        code = """
from crewai import Agent, Task

AGENT_ID = "test_extraction_agent"

def create_agent(llm=None) -> Agent:
    return Agent(role="Extractor", goal="Extract fields", backstory="Auditor", llm=llm)

def create_task(agent: Agent, invoice_data: dict) -> Task:
    description = f"Extract fields from {invoice_data}"
    expected_output = "Strictly valid JSON containing extracted_invoice and extraction_flags."
    return Task(description=description, expected_output=expected_output, agent=agent)
"""
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".py") as f:
            f.write(code)
            f_path = Path(f.name)

        try:
            parsed = self.parser.parse_file(f_path)
            ctx = self.structural_extractor.extract_structure(parsed)
            inputs, task_outputs, return_schemas = self.mpt_extractor.extract_inputs_and_outputs(ctx)

            # 1. Inputs: Must capture 'invoice_data', must NOT capture 'llm' or 'self' or 'agent'
            input_names = [i["name"] for i in inputs]
            self.assertIn("invoice_data", input_names)
            self.assertNotIn("llm", input_names)
            self.assertNotIn("agent", input_names)

            # 2. Task Expected Outputs: Must resolve variable string literal, NOT entire Task() or Agent() AST
            self.assertEqual(len(task_outputs), 1)
            self.assertEqual(task_outputs[0]["description"], "Strictly valid JSON containing extracted_invoice and extraction_flags.")

            # 3. Return schemas: Must NOT contain <Agent instance> or <Task instance>
            return_types = [r.get("return_type", "") for r in return_schemas]
            for rt in return_types:
                self.assertNotIn("Agent(", rt)
                self.assertNotIn("Task(", rt)
        finally:
            f_path.unlink(missing_ok=True)

    def test_2_execution_patterns_separation(self):
        """
        Correction 2: Separate model initialization, direct LLM invocations, agent execution,
        workflow execution, and tool execution. Methods like kickoff() must not be direct LLM calls.
        """
        code = """
from crewai import Agent, Crew
from common.agent_runner import create_groq_llm_for_agent

def run_pipeline():
    llm, _ = create_groq_llm_for_agent("my_agent")
    crew = Crew(agents=[], tasks=[])
    result = crew.kickoff()
    return result
"""
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".py") as f:
            f.write(code)
            f_path = Path(f.name)

        try:
            parsed = self.parser.parse_file(f_path)
            ctx = self.structural_extractor.extract_structure(parsed)
            models = self.mpt_extractor.extract_models(ctx)
            exec_pattern = self.mpt_extractor.extract_execution_patterns(ctx)

            # 1. kickoff() is a workflow execution call, NOT a direct LLM call
            self.assertTrue(any("kickoff" in c for c in exec_pattern.workflow_execution_calls))
            self.assertEqual(len(exec_pattern.direct_llm_calls), 0, "kickoff() must not be classified as direct LLM invocation")

            # 2. Model initialization is captured separately
            self.assertEqual(len(models), 1)
            self.assertEqual(models[0].provider, "Groq")
            self.assertEqual(models[0].direct_llm_invocations, [])
        finally:
            f_path.unlink(missing_ok=True)

    def test_3_tools_vs_helper_functions_distinction(self):
        """
        Correction 3: Distinguish explicitly registered tools from ordinary helper functions.
        A deterministic_* name alone is insufficient evidence of tool registration.
        """
        code = """
from crewai import Agent
from crewai.tools import tool

def deterministic_arithmetic_check(data):
    '''Ordinary deterministic calculation function.'''
    return data.get("qty", 0) * data.get("price", 0)

@tool("Calculator")
def registered_calculator_tool(query: str) -> str:
    '''Explicitly registered tool.'''
    return "calculated"
"""
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".py") as f:
            f.write(code)
            f_path = Path(f.name)

        try:
            parsed = self.parser.parse_file(f_path)
            ctx = self.structural_extractor.extract_structure(parsed)
            reg_tools, helpers = self.mpt_extractor.extract_tools_and_helpers(ctx)

            # 1. Registered tools must only contain decorated tool
            reg_tool_names = [t.name for t in reg_tools]
            self.assertIn("registered_calculator_tool", reg_tool_names)
            self.assertNotIn("deterministic_arithmetic_check", reg_tool_names, "deterministic_* is not a registered tool")

            # 2. Helper functions must contain deterministic_arithmetic_check
            self.assertIn("deterministic_arithmetic_check", helpers)
        finally:
            f_path.unlink(missing_ok=True)

    def test_4_unresolved_reference_classification(self):
        """
        Correction 4: An imported symbol is resolved by name and module even when
        its internal implementation source has not been supplied.
        """
        code = """
from crewai import Agent, Task
import json

def create_task(agent, data):
    # Task and json are imported, truly_undefined_symbol is not!
    result = json.dumps(data)
    truly_undefined_symbol(result)
    return Task(description="desc", expected_output="out", agent=agent)
"""
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".py") as f:
            f.write(code)
            f_path = Path(f.name)

        try:
            parsed = self.parser.parse_file(f_path)
            ctx = self.structural_extractor.extract_structure(parsed)

            # Imported symbols are mapped
            self.assertEqual(ctx.imported_symbols.get("Task"), "crewai.Task")
            self.assertEqual(ctx.imported_symbols.get("Agent"), "crewai.Agent")
            self.assertEqual(ctx.imported_symbols.get("json"), "json")

            # In unresolved references, 'Task' and 'json' must NOT be present!
            self.assertNotIn("Task", ctx.unresolved_references)
            self.assertNotIn("json", ctx.unresolved_references)

            # Truly undefined symbol MUST be identified as unresolved
            self.assertIn("truly_undefined_symbol", ctx.unresolved_references)
        finally:
            f_path.unlink(missing_ok=True)

    def test_5_token_counter_measurement_and_labeling(self):
        """
        Correction 5: Replace character estimates with tokenizer measurements wherever supported,
        and clearly label any fallback estimates.
        """
        counter = TokenCounter()
        sample_text = "def create_agent(llm=None):\n    return Agent(role='Auditor')\n"
        tokens = counter.count_tokens(sample_text)

        self.assertGreater(tokens, 0)
        self.assertIsNotNone(counter.method_label)
        # Must explicitly label whether actual tiktoken or labeled fallback
        if counter.is_actual:
            self.assertIn("tiktoken", counter.method_label)
        else:
            self.assertIn("fallback", counter.method_label)
            self.assertIn("bpe", counter.method_label)

    def test_6_no_duplication_and_compact_index_minimalism(self):
        """
        Correction 6: Review duplication across agent_index.json, agent_details.json,
        and code_evidence.json. Keep compact index minimal and evidence separately retrievable.
        """
        repo_root = Path(__file__).resolve().parent.parent
        target_file = repo_root / "Agents" / "invoice_processing" / "invoice_extraction_agent.py"
        self.assertTrue(target_file.exists())

        temp_dir = Path(tempfile.mkdtemp())
        try:
            orchestrator = IngestionOrchestrator()
            preparer, summary = orchestrator.process_files(
                agent_files=[target_file],
                output_dir=temp_dir,
            )

            with open(temp_dir / "agent_index.json", "r", encoding="utf-8") as f:
                index = json.load(f)

            agent_entry = index[0]
            # Verify compact index contains NO code snippet bodies
            index_str = json.dumps(agent_entry)
            self.assertNotIn("def create_agent", index_str)
            self.assertNotIn("def create_task", index_str)
            self.assertNotIn("Agent(\n", index_str)

            # Verify expected_outputs is clean
            for exp in agent_entry.get("expected_outputs", []):
                self.assertNotIn("Agent(", exp)
                self.assertNotIn("Task(", exp)

            # Verify evidence IDs point to code_evidence.json
            with open(temp_dir / "code_evidence.json", "r", encoding="utf-8") as f:
                evidence = json.load(f)

            for ev_id in agent_entry["evidence_ids"]:
                self.assertIn(ev_id, evidence)
                # Code snippet is stored in code_evidence.json only
                self.assertIn("snippet", evidence[ev_id])
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

