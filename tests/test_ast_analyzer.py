"""
Tests for TraceProfiler AST Ingestion & Multi-Agent Workflow Analyzer.

Validates that static AST analysis covers all 10 workflows (78 functional agents),
achieves 100% agent-to-source implementation resolution, detects execution orders
and architectural discrepancies, and exports report artifacts properly.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from common.settings import PROJECT_ROOT
from tools.ast_workflow_analyzer import ASTWorkflowAnalyzer

EXPECTED_WORKFLOW_AGENT_COUNTS = {
    "invoice_processing": 8,
    "customer_support": 6,
    "procurement_supplier_selection": 9,
    "sales_lead_qualification": 7,
    "recruitment_screening": 8,
    "inventory_replenishment": 6,
    "contract_review": 10,
    "incident_triage": 7,
    "expense_reimbursement": 8,
    "market_research": 9,
}


class TestASTWorkflowAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analyzer = ASTWorkflowAnalyzer(root_dir=PROJECT_ROOT)
        cls.report = cls.analyzer.run_full_analysis()

    def test_analyzer_root_and_python_scan(self):
        """Verify repository root and that all Python files are parsed without syntax errors."""
        self.assertEqual(self.analyzer.root_dir, PROJECT_ROOT)
        self.assertGreaterEqual(len(self.analyzer.python_inventory), 78, "Should parse at least 78 agent files")
        syntax_errors = [p for p in self.analyzer.python_inventory if p["parse_status"] != "ok"]
        self.assertEqual(len(syntax_errors), 0, f"Found Python files with syntax errors: {syntax_errors}")

    def test_ingest_all_10_workflow_manifests(self):
        """Verify exactly 10 workflow manifests are discovered and normalized."""
        self.assertEqual(len(self.analyzer.manifest_records), 10, f"Expected 10 manifests, got {len(self.analyzer.manifest_records)}")
        manifest_names = {m["workflow_name"] for m in self.analyzer.manifest_records}
        self.assertEqual(manifest_names, set(EXPECTED_WORKFLOW_AGENT_COUNTS.keys()))

        total_agents = sum(len(m["agents"]) for m in self.analyzer.manifest_records)
        self.assertEqual(total_agents, 78, f"Expected 78 total agents, got {total_agents}")

    def test_100_percent_agent_implementation_resolution(self):
        """Verify all 78 declared agents map to a concrete Python source implementation file."""
        for inv in self.analyzer.agent_inventories:
            wf_name = inv["workflow_name"]
            expected_count = EXPECTED_WORKFLOW_AGENT_COUNTS[wf_name]
            self.assertEqual(len(inv["agents"]), expected_count, f"{wf_name} has {len(inv['agents'])} agents, expected {expected_count}")

            unmatched = [a["id"] for a in inv["agents"] if not a["implementation_found"]]
            self.assertEqual(len(unmatched), 0, f"Workflow '{wf_name}' has unmatched agents: {unmatched}")

    def test_observed_execution_orders(self):
        """Verify observed execution orders in workflow.py match declared execution orders in workflow.json."""
        for comp in self.analyzer.comparisons:
            wf = comp["workflow"]
            declared = comp["declared_order"]
            observed = comp["source_observed_order"]

            self.assertEqual(len(declared), EXPECTED_WORKFLOW_AGENT_COUNTS[wf], f"Declared count mismatch in {wf}")
            self.assertEqual(len(observed), EXPECTED_WORKFLOW_AGENT_COUNTS[wf], f"Observed count mismatch in {wf}")
            self.assertEqual(declared, observed, f"Execution order mismatch in {wf}: declared={declared}, observed={observed}")

    def test_discrepancies_detected(self):
        """Verify architectural discrepancies (e.g. multiple kickoffs, declared parallelism) are flagged."""
        discrepancy_types = set()
        for comp in self.analyzer.comparisons:
            for f in comp["findings"]:
                discrepancy_types.add(f["type"])

        self.assertIn("MULTIPLE_KICKOFF_CALLS", discrepancy_types, "Should flag workflows using multiple sequential Crew kickoffs")
        self.assertIn("DECLARED_PARALLELIZABLE", discrepancy_types, "Should flag agents declared parallelizable in manifests")

    def test_export_reports_and_csvs(self):
        """Verify export_reports generates all JSON and CSV artifacts with non-empty content."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            analyzer = ASTWorkflowAnalyzer(root_dir=PROJECT_ROOT)
            out_dir = analyzer.export_reports(output_dir=temp_dir)

            expected_files = [
                "target_mas_ingestion_report.json",
                "agent_inventory.csv",
                "python_file_inventory.csv",
                "workflow_summary.csv",
                "discrepancies.csv",
            ]

            for fname in expected_files:
                fpath = out_dir / fname
                self.assertTrue(fpath.exists(), f"Missing exported artifact: {fname}")
                self.assertGreater(fpath.stat().st_size, 0, f"Exported artifact is empty: {fname}")

            with open(out_dir / "target_mas_ingestion_report.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["metrics"]["total_workflows"], 10)
            self.assertEqual(data["metrics"]["total_declared_agents"], 78)
            self.assertEqual(data["metrics"]["matched_implemented_agents"], 78)
            self.assertEqual(data["metrics"]["agent_match_rate"], "100.0%")

            # Verify separated per-workflow reports exist for all 10 workflows
            wf_expected_files = [
                "ingestion_report.json",
                "agent_inventory.csv",
                "python_file_inventory.csv",
                "workflow_summary.csv",
                "discrepancies.csv",
                "workflow_summary.md",
            ]
            for wf in EXPECTED_WORKFLOW_AGENT_COUNTS.keys():
                wf_dir = out_dir / wf
                self.assertTrue(wf_dir.exists(), f"Missing workflow report directory: {wf}")
                self.assertTrue(wf_dir.is_dir())
                for wfname in wf_expected_files:
                    wf_file = wf_dir / wfname
                    self.assertTrue(wf_file.exists(), f"Missing report in {wf}: {wfname}")
                    self.assertGreater(wf_file.stat().st_size, 0, f"Empty report in {wf}: {wfname}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
