"""
Unit tests for PythonASTParser and StructuralExtractor.
"""

import ast
import tempfile
import unittest
from pathlib import Path

from ingestion.python_ast_parser import PythonASTParser
from ingestion.structural_extractor import StructuralExtractor
from schemas.evidence_schema import EvidenceStatus


class TestStructuralExtractor(unittest.TestCase):
    def setUp(self):
        self.parser = PythonASTParser()
        self.extractor = StructuralExtractor(self.parser)

    def test_safe_read_and_sha256(self):
        """Verify safe text reading and SHA256 computation."""
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".py") as f:
            f.write("x = 42\nprint(x)\n")
            f_path = Path(f.name)

        try:
            parsed = self.parser.parse_file(f_path)
            self.assertTrue(parsed.is_valid)
            self.assertEqual(parsed.status, "ok")
            self.assertIsNotNone(parsed.sha256)
            self.assertEqual(len(parsed.sha256), 64)
        finally:
            f_path.unlink(missing_ok=True)

    def test_syntax_error_resilience(self):
        """Verify syntax errors are recorded without halting the pipeline."""
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".py") as f:
            f.write("def malformed_syntax(\n  # incomplete\n")
            f_path = Path(f.name)

        try:
            parsed = self.parser.parse_file(f_path)
            self.assertFalse(parsed.is_valid)
            self.assertEqual(parsed.status, "syntax_error")
            self.assertIn("line", parsed.error or "")
        finally:
            f_path.unlink(missing_ok=True)

    def test_structural_extraction_elements(self):
        """Verify imports, classes, functions, and top constants are properly extracted."""
        code = """
import os
from common.settings import MANDATORY_MODEL
from external_unknown import mystery_fn

AGENT_ID = "test_sample_agent"

def deterministic_math(a, b=10):
    '''Calculate total sum.'''
    return a + b

class SampleProcessor:
    def process(self, data):
        return mystery_fn(data)
"""
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".py") as f:
            f.write(code)
            f_path = Path(f.name)

        try:
            parsed = self.parser.parse_file(f_path)
            ctx = self.extractor.extract_structure(parsed, known_modules={"common.settings"})

            # Check constants
            self.assertEqual(ctx.top_level_constants.get("AGENT_ID"), "test_sample_agent")

            # Check imports and external module status
            external = ctx.external_modules
            self.assertIn("external_unknown", external)

            # Check functions
            fn_names = [fn.name for fn in ctx.functions]
            self.assertIn("deterministic_math", fn_names)
            math_fn = next(fn for fn in ctx.functions if fn.name == "deterministic_math")
            self.assertEqual(math_fn.args, ["a", "b"])
            self.assertEqual(math_fn.docstring, "Calculate total sum.")

            # Check classes
            self.assertEqual(len(ctx.classes), 1)
            self.assertEqual(ctx.classes[0].name, "SampleProcessor")
            self.assertEqual(len(ctx.classes[0].methods), 1)
        finally:
            f_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()

