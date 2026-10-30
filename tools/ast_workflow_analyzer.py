"""
TraceProfiler: Step 1 - AST-Based Multi-Agent System (MAS) Ingestion & Analyzer.

Performs static, non-executing AST analysis across all 10 agentic workflows
in the repository (78 functional agents). Identifies:
  - Agent definitions, roles, goals, backstories, and implementation locations
  - Workflow entry points, CrewAI primitives (Crew/Task/Agent), and execution calls
  - Variable handoffs and data flow dependencies
  - Declared workflow metadata (workflow.json) vs implemented reality (workflow.py)
  - Run history and telemetry artifacts in runs/
  - Generates consolidated JSON ingestion reports and tabular CSV summaries.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class ASTWorkflowAnalyzer:
    """Static AST-based analyzer for agentic workflows."""

    def __init__(self, root_dir: Optional[Path] = None):
        self.root_dir = Path(root_dir).resolve() if root_dir else Path.cwd().resolve()
        self.agents_dir = self.root_dir / "Agents"
        self.runs_dir = self.root_dir / "runs"
        self.python_inventory: List[Dict[str, Any]] = []
        self.manifest_records: List[Dict[str, Any]] = []
        self.agent_inventories: List[Dict[str, Any]] = []
        self.workflow_analyses: List[Dict[str, Any]] = []
        self.comparisons: List[Dict[str, Any]] = []
        self.run_inventory: List[Dict[str, Any]] = []
        self.python_files_by_path: Dict[str, Dict[str, Any]] = {}

    # -------------------------------------------------------------------------
    # Helper Utilities
    # -------------------------------------------------------------------------

    @staticmethod
    def safe_read_text(path: Path) -> str:
        """Safely read text with encoding fallbacks without raising exceptions."""
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                return path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                return ""
        except Exception:
            return ""

    @staticmethod
    def node_source(node: Optional[ast.AST], source: str) -> str:
        """Extract source code segment corresponding to an AST node."""
        if node is None:
            return ""
        try:
            return ast.get_source_segment(source, node) or ""
        except Exception:
            return ""

    @staticmethod
    def literal_value(node: Optional[ast.AST]) -> Any:
        """Extract values that can be safely evaluated as Python literals."""
        if node is None:
            return None
        try:
            return ast.literal_eval(node)
        except Exception:
            return None

    @classmethod
    def dotted_name(cls, node: Optional[ast.AST]) -> str:
        """Recursively resolve dotted identifier names."""
        if node is None:
            return ""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            left = cls.dotted_name(node.value)
            return f"{left}.{node.attr}" if left else node.attr
        if isinstance(node, ast.Call):
            return cls.dotted_name(node.func)
        return ""

    @staticmethod
    def extract_string_constants(node: Optional[ast.AST]) -> List[str]:
        """Extract string constants from an AST subtree while preserving order."""
        if node is None:
            return []
        values = []
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                val = child.value.strip()
                if val:
                    values.append(val)
        return list(dict.fromkeys(values))

    def relpath(self, path: Path) -> str:
        """Return relative path to root_dir with forward slashes."""
        try:
            return str(path.relative_to(self.root_dir)).replace("\\", "/")
        except Exception:
            return str(path).replace("\\", "/")

    # -------------------------------------------------------------------------
    # 1. Python File AST Parser
    # -------------------------------------------------------------------------

    def parse_python_file(self, path: Path) -> Dict[str, Any]:
        """Parse a Python source file using AST without executing any code."""
        source = self.safe_read_text(path)
        sha256_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
        rel_path = self.relpath(path)

        result: Dict[str, Any] = {
            "file": rel_path,
            "filename": path.name,
            "stem": path.stem,
            "sha256": sha256_hash,
            "parse_status": "ok",
            "parse_error": None,
            "imports": [],
            "classes": [],
            "functions": [],
            "agent_id_constants": [],
            "agent_definitions": [],
            "task_definitions": [],
            "crew_definitions": [],
            "llm_references": [],
            "tool_references": [],
            "execution_calls": [],
            "string_literals": [],
        }

        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            result["parse_status"] = "syntax_error"
            result["parse_error"] = f"{exc.msg} (line {exc.lineno})"
            return result

        # 1. Imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    result["imports"].append({
                        "module": alias.name,
                        "asname": alias.asname,
                        "line": node.lineno,
                    })
            elif isinstance(node, ast.ImportFrom):
                result["imports"].append({
                    "module": node.module or "",
                    "level": node.level,
                    "names": [a.name for a in node.names],
                    "aliases": {a.name: a.asname for a in node.names if a.asname},
                    "line": node.lineno,
                })

        # 2. Top-level constants (e.g. AGENT_ID = "...")
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "AGENT_ID":
                        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                            result["agent_id_constants"].append(node.value.value)

        # 3. Classes and Functions
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                result["classes"].append({"name": node.name, "line": node.lineno})
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn = {
                    "name": node.name,
                    "line": node.lineno,
                    "end_line": getattr(node, "end_lineno", None),
                    "docstring": ast.get_docstring(node),
                    "args": [a.arg for a in node.args.args],
                    "decorators": [self.dotted_name(d) for d in node.decorator_list],
                }
                result["functions"].append(fn)

                name_lower = node.name.lower()
                if any(k in name_lower for k in [
                    "tool", "query", "search", "calculate", "validate",
                    "lookup", "check", "deterministic_", "run_",
                ]):
                    result["tool_references"].append({
                        "name": node.name,
                        "line": node.lineno,
                        "classification": "function_candidate",
                        "docstring": fn["docstring"],
                    })

        # 4. CrewAI Constructions & Execution Calls
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call_name = self.dotted_name(node.func)
                leaf = call_name.split(".")[-1]

                if leaf == "Agent":
                    kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg}
                    result["agent_definitions"].append({
                        "line": node.lineno,
                        "constructor": call_name,
                        "role": self.literal_value(kwargs.get("role")),
                        "goal": self.literal_value(kwargs.get("goal")),
                        "backstory": self.literal_value(kwargs.get("backstory")),
                        "llm_expression": self.node_source(kwargs.get("llm"), source),
                        "verbose": self.literal_value(kwargs.get("verbose")),
                        "allow_delegation": self.literal_value(kwargs.get("allow_delegation")),
                    })

                elif leaf == "Task":
                    kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg}
                    desc_node = kwargs.get("description")
                    result["task_definitions"].append({
                        "line": node.lineno,
                        "description_expression": self.node_source(desc_node, source)[:800],
                        "description_literals": self.extract_string_constants(desc_node),
                        "expected_output": self.literal_value(kwargs.get("expected_output")),
                        "agent_expression": self.node_source(kwargs.get("agent"), source),
                    })

                elif leaf == "Crew":
                    kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg}
                    result["crew_definitions"].append({
                        "line": node.lineno,
                        "agents_expression": self.node_source(kwargs.get("agents"), source),
                        "tasks_expression": self.node_source(kwargs.get("tasks"), source),
                        "process_expression": self.node_source(kwargs.get("process"), source),
                    })

                # LLM / Model calls
                if any(term in call_name.lower() for term in [
                    "llm", "chat", "model", "completion", "invoke", "create_groq_llm"
                ]):
                    result["llm_references"].append({
                        "call": call_name,
                        "line": node.lineno,
                        "arguments": [self.node_source(a, source)[:200] for a in node.args[:3]],
                    })

                # Execution calls
                if leaf in {"kickoff", "kickoff_async", "execute", "run", "log_agent_execution", "agent_span"}:
                    result["execution_calls"].append({
                        "call": call_name,
                        "line": node.lineno,
                    })

        # 5. Prominent String Literals (Prompts / Rubrics >= 35 chars)
        result["string_literals"] = [
            {"value": node.value[:1000], "line": node.lineno}
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and len(node.value.strip()) >= 35
        ]

        return result

    def scan_all_python_files(self) -> List[Dict[str, Any]]:
        """Scan and parse all Python files in the repository."""
        py_files = sorted(self.root_dir.rglob("*.py"))
        self.python_inventory = []
        self.python_files_by_path = {}

        for p in py_files:
            # Skip virtual environments or cache folders if any
            if any(part in p.parts for part in [".venv", "venv", "__pycache__", ".pytest_cache"]):
                continue
            parsed = self.parse_python_file(p)
            self.python_inventory.append(parsed)
            self.python_files_by_path[parsed["file"]] = parsed

        return self.python_inventory

    # -------------------------------------------------------------------------
    # 2. Workflow Manifest Parser (workflow.json)
    # -------------------------------------------------------------------------

    def scan_manifests(self) -> List[Dict[str, Any]]:
        """Discover and normalize all workflow manifests in the repository."""
        # Focus on workflow manifest files in Agents/ or root, ignoring runs/reports/data/
        self.manifest_records = []
        candidate_files = []

        # First priority: explicit workflow.json files
        for p in sorted(self.root_dir.rglob("workflow.json")):
            if not any(part in p.parts for part in [".venv", "venv", "__pycache__", ".pytest_cache", "runs", "reports"]):
                candidate_files.append(p)

        # Also search for any manifest-like json files if no workflow.json found
        if not candidate_files:
            for p in sorted(self.root_dir.rglob("*.json")):
                if any(part in p.parts for part in [".venv", "venv", "__pycache__", ".pytest_cache", "runs", "reports", "data"]):
                    continue
                candidate_files.append(p)

        for p in candidate_files:
            content = self.safe_read_text(p)
            try:
                data = json.loads(content)
            except Exception:
                continue

            # Must contain workflow definition keys
            if (
                isinstance(data, dict)
                and "agents" in data
                and isinstance(data["agents"], list)
                and ("execution_order" in data or "workflow_name" in data)
            ):
                normalized_agents = []
                for a in data.get("agents", []):
                    if not isinstance(a, dict):
                        continue
                    agent_id = a.get("id") or a.get("agent_id")
                    if not agent_id:
                        continue
                    name = a.get("name") or a.get("agent_name") or a.get("role")
                    role = a.get("role") or a.get("name")
                    declared_file = a.get("file")
                    deps = a.get("dependencies", [])
                    inputs = a.get("inputs") or a.get("input_fields") or []
                    outputs = a.get("outputs") or a.get("output_fields") or []
                    assigned_key = a.get("assigned_key") or a.get("assigned_key_env")
                    parallelizable = a.get("parallelizable", False)
                    parallel_group = a.get("parallel_group")
                    pruning_candidate = a.get("pruning_candidate") or a.get("optimization_note")

                    normalized_agents.append({
                        "id": agent_id,
                        "name": name,
                        "role": role,
                        "declared_file": declared_file,
                        "dependencies": deps,
                        "inputs": inputs,
                        "outputs": outputs,
                        "assigned_key": assigned_key,
                        "parallelizable": parallelizable,
                        "parallel_group": parallel_group,
                        "pruning_candidate": pruning_candidate,
                        "raw": a,
                    })

                record = {
                    "file": self.relpath(p),
                    "dir": self.relpath(p.parent),
                    "workflow_name": data.get("workflow_name", p.parent.name),
                    "version": data.get("version", "1.0.0"),
                    "description": data.get("description", ""),
                    "fixed_model": data.get("fixed_model", ""),
                    "total_agents": data.get("total_agents", len(normalized_agents)),
                    "execution_mode": data.get("execution_mode", "sequential"),
                    "execution_order": data.get("execution_order", []),
                    "optimization_opportunities": data.get("optimization_opportunities", []),
                    "agents": normalized_agents,
                }
                self.manifest_records.append(record)

        return self.manifest_records

    # -------------------------------------------------------------------------
    # 3. Agent Alignment (Manifest -> AST Code Implementation)
    # -------------------------------------------------------------------------

    def build_agent_inventory(self, manifest_record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Map declared manifest agents to exact Python AST files and constructs."""
        manifest_dir = manifest_record["dir"]
        # Limit candidate python files primarily to the workflow folder
        wf_py_files = [
            p for p in self.python_inventory
            if p["file"].startswith(manifest_dir + "/")
            and not p["filename"].startswith("workflow")
            and not p["filename"].startswith("__")
        ]

        # In fallback, consider all parsed files
        all_py_files = self.python_inventory

        agents_out = []
        for agent in manifest_record["agents"]:
            aid = agent["id"]
            role = agent["role"] or agent["name"]
            declared_file = agent["declared_file"]

            matched: Optional[Dict[str, Any]] = None

            # Strategy 1: Explicit manifest file attribute
            if declared_file:
                target_name = Path(declared_file).name
                for p in wf_py_files:
                    if p["filename"] == target_name:
                        matched = p
                        break

            # Strategy 2: AGENT_ID constant inside file matches aid exactly
            if not matched:
                for p in wf_py_files:
                    if aid in p.get("agent_id_constants", []):
                        matched = p
                        break

            # Strategy 3: Filename stem exact or stripped match
            if not matched:
                for p in wf_py_files:
                    stem = p["stem"]
                    if stem == aid or stem.replace("_agent", "") == aid.replace("_agent", ""):
                        matched = p
                        break

            # Strategy 4: Substring/token match
            if not matched:
                aid_clean = aid.replace("agent", "").replace("_", "")
                for p in wf_py_files:
                    stem_clean = p["stem"].replace("agent", "").replace("_", "")
                    if stem_clean and (stem_clean in aid_clean or aid_clean in stem_clean):
                        matched = p
                        break

            # Strategy 5: Role matching in AST Agent(role=...)
            if not matched and role:
                role_lower = role.lower()
                for p in wf_py_files:
                    for ad in p.get("agent_definitions", []):
                        r = str(ad.get("role") or "").lower()
                        if r and (role_lower in r or r in role_lower):
                            matched = p
                            break
                    if matched:
                        break

            # Strategy 6: Global fallback across entire repo if needed
            if not matched and declared_file:
                for p in all_py_files:
                    if p["filename"] == Path(declared_file).name:
                        matched = p
                        break

            agents_out.append({
                "id": aid,
                "name": agent["name"],
                "role": role,
                "workflow": manifest_record["workflow_name"],
                "declared_file": declared_file,
                "assigned_key": agent["assigned_key"],
                "dependencies": agent["dependencies"],
                "inputs": agent["inputs"],
                "outputs": agent["outputs"],
                "parallelizable": agent["parallelizable"],
                "parallel_group": agent["parallel_group"],
                "pruning_candidate": agent["pruning_candidate"],
                "implementation_found": matched is not None,
                "implementation_file": matched["file"] if matched else None,
                "agent_constructors": matched["agent_definitions"] if matched else [],
                "task_constructors": matched["task_definitions"] if matched else [],
                "functions": matched["functions"] if matched else [],
                "imports": matched["imports"] if matched else [],
                "tool_references": matched["tool_references"] if matched else [],
            })

        return agents_out

    # -------------------------------------------------------------------------
    # 4. Workflow Source & Execution Order Analysis
    # -------------------------------------------------------------------------

    def analyze_workflow_source(self, path: Path) -> Dict[str, Any]:
        """Analyze workflow orchestrator (workflow.py) for entry points, stages, and handoffs."""
        source = self.safe_read_text(path)
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            return {
                "file": self.relpath(path),
                "error": str(exc),
                "entry_points": [],
                "crew_constructions": [],
                "assignments": [],
                "task_handoffs": [],
                "observed_order": [],
            }

        entry_points = []
        classes = []
        crew_constructions = []
        assignments = []
        task_handoffs = []

        # Map import aliases
        aliases: Dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for a in node.names:
                    if a.asname:
                        aliases[a.asname] = a.name

        # Detect entry points and classes
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append({"name": node.name, "line": node.lineno})
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name_lower = node.name.lower()
                if any(k in name_lower for k in ["run", "workflow", "execute", "main"]):
                    entry_points.append({
                        "name": node.name,
                        "line": node.lineno,
                        "args": [a.arg for a in node.args.args],
                        "docstring": ast.get_docstring(node),
                    })

            # Crew(...) calls
            if isinstance(node, ast.Call) and self.dotted_name(node.func).split(".")[-1] == "Crew":
                crew_constructions.append({
                    "line": node.lineno,
                    "source": self.node_source(node, source)[:1000],
                })

            # Assignments for data flow
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                target_names = [t.id for t in targets if isinstance(t, ast.Name)]
                val = node.value
                if target_names and val is not None:
                    value_names = sorted({n.id for n in ast.walk(val) if isinstance(n, ast.Name)})
                    assignments.append({
                        "line": node.lineno,
                        "targets": target_names,
                        "value_names": value_names,
                        "expression": self.node_source(val, source)[:300],
                    })

            # Task(...) definitions and variable references
            if isinstance(node, ast.Call) and self.dotted_name(node.func).split(".")[-1] == "Task":
                kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg}
                desc = kwargs.get("description")
                ref_names = sorted({n.id for n in ast.walk(desc) if isinstance(n, ast.Name)}) if desc else []
                task_handoffs.append({
                    "line": node.lineno,
                    "referenced_variables": ref_names,
                    "description_literals": self.extract_string_constants(desc) if desc else [],
                    "agent_expression": self.node_source(kwargs.get("agent"), source),
                })

        # Observed execution order detection
        observed_order = self._detect_observed_order(tree, source, path.parent)

        return {
            "file": self.relpath(path),
            "classes": classes,
            "entry_points": entry_points,
            "crew_constructions": crew_constructions,
            "assignments": assignments,
            "task_handoffs": task_handoffs,
            "observed_order": observed_order,
            "sequential_kickoffs": len(re.findall(r"\.kickoff\s*\(", source)),
        }

    def _detect_observed_order(self, tree: ast.AST, source: str, parent_dir: Path) -> List[str]:
        """Detect agent execution order across multiple architectural conventions."""
        observed: List[str] = []

        # 1. Look for agent_span("agent_name", ...)
        for match in re.finditer(r'agent_span\(\s*["\']([^"\']+)["\']', source):
            aid = match.group(1)
            if aid not in observed:
                observed.append(aid)

        if observed:
            return observed

        # 2. Look for tracker.log_agent_execution(...)
        # Build local mapping of module/agent constants in parent_dir
        agent_id_by_module = {}
        for p in parent_dir.glob("*.py"):
            if p.name == "workflow.py":
                continue
            try:
                t = ast.parse(self.safe_read_text(p))
                for node in ast.walk(t):
                    if isinstance(node, ast.Assign):
                        for tgt in node.targets:
                            if isinstance(tgt, ast.Name) and tgt.id == "AGENT_ID":
                                if isinstance(node.value, ast.Constant):
                                    agent_id_by_module[p.stem] = node.value.value
            except Exception:
                pass

        var_to_aid: Dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod_stem = Path(node.module.replace(".", "/")).name if node.module else ""
                for alias in node.names:
                    var_name = alias.asname or alias.name
                    if alias.name == "AGENT_ID" and mod_stem in agent_id_by_module:
                        var_to_aid[var_name] = agent_id_by_module[mod_stem]
                    elif mod_stem in agent_id_by_module:
                        var_to_aid[var_name] = agent_id_by_module[mod_stem]

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call_name = ""
                if isinstance(node.func, ast.Attribute):
                    call_name = f"{getattr(node.func.value, 'id', '')}.{node.func.attr}"
                if "log_agent_execution" in call_name and node.args:
                    first = node.args[0]
                    resolved = None
                    if isinstance(first, ast.Name) and first.id in var_to_aid:
                        resolved = var_to_aid[first.id]
                    elif isinstance(first, ast.Constant):
                        resolved = first.value
                    if resolved and resolved not in observed:
                        observed.append(resolved)

        if observed:
            return observed

        # 3. Look for module stage execution calls (e.g. intake_agent.run_contract_intake)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                mod_name = getattr(node.func.value, "id", "")
                if mod_name in agent_id_by_module:
                    aid = agent_id_by_module[mod_name]
                    if aid not in observed:
                        observed.append(aid)

        return observed

    # -------------------------------------------------------------------------
    # 5. Manifest vs Implementation Discrepancy Detector
    # -------------------------------------------------------------------------

    def compare_manifest_to_source(
        self,
        manifest_record: Dict[str, Any],
        workflow_analysis: Optional[Dict[str, Any]],
        agent_inventory: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Detect architectural mismatches between manifest declarations and source code."""
        findings: List[Dict[str, Any]] = []
        declared_order = manifest_record.get("execution_order", [])
        observed_order = workflow_analysis.get("observed_order", []) if workflow_analysis else []

        # 1. Execution Order Mismatch
        if declared_order and observed_order and declared_order != observed_order:
            findings.append({
                "type": "ORDER_MISMATCH",
                "severity": "HIGH",
                "declared": declared_order,
                "source_observed": observed_order,
                "note": "Declared execution order differs from source code orchestration.",
            })

        # 2. Parallelizability Discrepancy
        # If marked parallel in manifest but executed sequentially in source
        parallel_agents = [a["id"] for a in agent_inventory if a.get("parallelizable") is True]
        if parallel_agents:
            findings.append({
                "type": "DECLARED_PARALLELIZABLE",
                "severity": "INFO",
                "agents": parallel_agents,
                "note": "Agents declared parallelizable in manifest; confirm concurrency in runtime traces.",
            })

        # 3. Sequential Multiple Kickoffs
        kickoffs = workflow_analysis.get("sequential_kickoffs", 0) if workflow_analysis else 0
        if kickoffs > 1:
            findings.append({
                "type": "MULTIPLE_KICKOFF_CALLS",
                "severity": "WARNING",
                "count": kickoffs,
                "note": f"Workflow invokes {kickoffs} sequential Crew kickoff calls instead of single execution graph.",
            })

        # 4. Missing Implementation Detection
        missing_impl = [a["id"] for a in agent_inventory if not a.get("implementation_found")]
        if missing_impl:
            findings.append({
                "type": "MISSING_IMPLEMENTATION",
                "severity": "CRITICAL",
                "agents": missing_impl,
                "note": "Declared agents have no matching Python source implementation file.",
            })

        return {
            "workflow": manifest_record["workflow_name"],
            "manifest_file": manifest_record["file"],
            "workflow_source": workflow_analysis["file"] if workflow_analysis else None,
            "declared_order": declared_order,
            "source_observed_order": observed_order,
            "declared_agent_count": len(declared_order),
            "observed_agent_count": len(observed_order),
            "discrepancy_count": len(findings),
            "findings": findings,
        }

    # -------------------------------------------------------------------------
    # 6. Telemetry & Execution Artifact Ingestion
    # -------------------------------------------------------------------------

    def scan_telemetry(self) -> List[Dict[str, Any]]:
        """Discover and summarize execution history files in runs/."""
        if not self.runs_dir.exists():
            return []

        run_files = []
        for p in self.runs_dir.rglob("*"):
            if p.is_file() and (p.suffix.lower() in {".json", ".jsonl", ".ndjson"}):
                run_files.append(p)

        self.run_inventory = []
        for p in sorted(run_files):
            rec: Dict[str, Any] = {
                "file": self.relpath(p),
                "filename": p.name,
                "suffix": p.suffix.lower(),
            }
            try:
                if p.suffix.lower() == ".json":
                    data = json.loads(self.safe_read_text(p))
                    if isinstance(data, dict):
                        rec["workflow_name"] = data.get("workflow_name") or data.get("workflow")
                        rec["run_id"] = data.get("run_id")
                        rec["status"] = data.get("status")
                        rec["total_duration_seconds"] = data.get("total_duration_seconds") or data.get("duration_seconds")
                        rec["agent_count"] = data.get("agent_count")
                        rec["total_tokens"] = data.get("total_tokens")
                        rec["estimated_cost_usd"] = data.get("estimated_cost_usd")
                elif p.suffix.lower() in {".jsonl", ".ndjson"}:
                    lines = [line for line in self.safe_read_text(p).splitlines() if line.strip()]
                    rec["record_count"] = len(lines)
                    if lines:
                        try:
                            first = json.loads(lines[0])
                            if isinstance(first, dict):
                                rec["first_record_keys"] = list(first.keys())
                                rec["workflow_name"] = first.get("workflow_name") or first.get("workflow")
                        except Exception:
                            pass
            except Exception as exc:
                rec["read_error"] = str(exc)

            self.run_inventory.append(rec)

        return self.run_inventory

    # -------------------------------------------------------------------------
    # 7. Complete Pipeline Orchestrator & Report Generation
    # -------------------------------------------------------------------------

    def run_full_analysis(self) -> Dict[str, Any]:
        """Execute complete AST ingestion and analysis across all workflows."""
        self.scan_all_python_files()
        self.scan_manifests()

        self.agent_inventories = []
        self.workflow_analyses = []
        self.comparisons = []

        # Find workflow.py files
        wf_files_by_dir: Dict[str, Path] = {}
        for p in sorted(self.agents_dir.rglob("workflow.py")):
            wf_files_by_dir[self.relpath(p.parent)] = p

        for manifest in self.manifest_records:
            agents = self.build_agent_inventory(manifest)
            self.agent_inventories.append({
                "workflow_name": manifest["workflow_name"],
                "manifest_file": manifest["file"],
                "agents": agents,
            })

            wf_path = wf_files_by_dir.get(manifest["dir"])
            wf_analysis = self.analyze_workflow_source(wf_path) if wf_path else None
            if wf_analysis:
                self.workflow_analyses.append(wf_analysis)

            comp = self.compare_manifest_to_source(manifest, wf_analysis, agents)
            self.comparisons.append(comp)

        self.scan_telemetry()

        total_agents = sum(len(x["agents"]) for x in self.agent_inventories)
        matched_agents = sum(
            sum(1 for a in x["agents"] if a["implementation_found"])
            for x in self.agent_inventories
        )

        report = {
            "report_type": "traceprofiler_target_mas_ingestion",
            "analysis_mode": "static_ast_non_executing",
            "repository_root": str(self.root_dir),
            "metrics": {
                "total_workflows": len(self.manifest_records),
                "total_declared_agents": total_agents,
                "matched_implemented_agents": matched_agents,
                "agent_match_rate": f"{(matched_agents / total_agents * 100):.1f}%" if total_agents else "0%",
                "total_python_files": len(self.python_inventory),
                "total_telemetry_artifacts": len(self.run_inventory),
            },
            "manifests": self.manifest_records,
            "agent_inventories": self.agent_inventories,
            "workflow_analyses": self.workflow_analyses,
            "comparisons": self.comparisons,
            "python_files": self.python_inventory,
            "run_inventory": self.run_inventory,
            "limitations": [
                "AST analysis does not execute code or resolve arbitrary runtime values.",
                "Function and variable references are evidence, not proof of semantic data dependencies.",
                "Dynamic imports, decorators, generated code and framework internals may require trace corroboration.",
                "Manifest parallelizable flags are declarations and do not prove runtime concurrency.",
                "Runtime traces are required to validate observed execution order and timing.",
            ],
        }

        return report

    def export_reports(self, output_dir: Optional[Path] = None) -> Path:
        """Export JSON report and CSV summaries to output_dir."""
        out_dir = Path(output_dir) if output_dir else self.root_dir / "reports" / "traceprofiler"
        out_dir.mkdir(parents=True, exist_ok=True)

        report = self.run_full_analysis()

        # 1. target_mas_ingestion_report.json
        json_path = out_dir / "target_mas_ingestion_report.json"
        json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

        # 2. agent_inventory.csv
        agent_rows = []
        for inv in self.agent_inventories:
            for a in inv["agents"]:
                agent_rows.append({
                    "workflow": a["workflow"],
                    "agent_id": a["id"],
                    "role": a["role"],
                    "assigned_key": a["assigned_key"],
                    "declared_file": a["declared_file"] or "",
                    "implementation_file": a["implementation_file"] or "NOT_FOUND",
                    "implementation_found": a["implementation_found"],
                    "dependencies": "; ".join(a["dependencies"]),
                    "inputs": "; ".join(a["inputs"]),
                    "outputs": "; ".join(a["outputs"]),
                    "parallelizable": a["parallelizable"],
                    "pruning_candidate": a["pruning_candidate"] or "",
                })

        if agent_rows:
            csv_path = out_dir / "agent_inventory.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(agent_rows[0].keys()))
                writer.writeheader()
                writer.writerows(agent_rows)

        # 3. python_file_inventory.csv
        file_rows = [
            {
                "file": p["file"],
                "parse_status": p["parse_status"],
                "imports": len(p["imports"]),
                "classes": len(p["classes"]),
                "functions": len(p["functions"]),
                "agent_constructors": len(p["agent_definitions"]),
                "task_constructors": len(p["task_definitions"]),
                "crew_constructors": len(p["crew_definitions"]),
                "execution_calls": len(p["execution_calls"]),
                "sha256": p["sha256"],
            }
            for p in self.python_inventory
        ]

        if file_rows:
            csv_path = out_dir / "python_file_inventory.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(file_rows[0].keys()))
                writer.writeheader()
                writer.writerows(file_rows)

        # 4. workflow_summary.csv
        wf_rows = []
        for comp in self.comparisons:
            wf_rows.append({
                "workflow": comp["workflow"],
                "manifest_file": comp["manifest_file"],
                "workflow_source": comp["workflow_source"] or "NONE",
                "declared_agents": comp["declared_agent_count"],
                "observed_agents": comp["observed_agent_count"],
                "discrepancies": comp["discrepancy_count"],
            })

        if wf_rows:
            csv_path = out_dir / "workflow_summary.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(wf_rows[0].keys()))
                writer.writeheader()
                writer.writerows(wf_rows)

        # 5. discrepancies.csv
        disc_rows = []
        for comp in self.comparisons:
            for finding in comp["findings"]:
                disc_rows.append({
                    "workflow": comp["workflow"],
                    "finding_type": finding["type"],
                    "severity": finding.get("severity", "INFO"),
                    "details": json.dumps(finding, default=str),
                })

        if disc_rows:
            csv_path = out_dir / "discrepancies.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(disc_rows[0].keys()))
                writer.writeheader()
                writer.writerows(disc_rows)

        # 6. Separate per-workflow reports for all workflows
        for comp in self.comparisons:
            wf_name = comp["workflow"]
            wf_target_dir = out_dir / wf_name
            self.export_workflow_report(wf_name, wf_target_dir)

        return out_dir

    def export_workflow_report(self, wf_name: str, target_dir: Path) -> Dict[str, Any]:
        """Export dedicated JSON, CSV, and Markdown reports for a specific workflow."""
        target_dir.mkdir(parents=True, exist_ok=True)

        manifest = next((m for m in self.manifest_records if m["workflow_name"] == wf_name), {})
        agent_inv = next((inv["agents"] for inv in self.agent_inventories if inv["workflow_name"] == wf_name), [])
        wf_analysis = next(
            (w for w in self.workflow_analyses if f"/{wf_name}/" in w["file"] or w["file"].startswith(f"Agents/{wf_name}/")),
            None
        )
        comp = next((c for c in self.comparisons if c["workflow"] == wf_name), {})
        wf_py_files = [
            p for p in self.python_inventory
            if f"/{wf_name}/" in p["file"] or p["file"].startswith(f"Agents/{wf_name}/")
        ]
        wf_runs = [
            r for r in self.run_inventory
            if r.get("workflow_name") == wf_name or f"runs/{wf_name}/" in r.get("file", "")
        ]

        wf_report = {
            "workflow_name": wf_name,
            "report_type": "traceprofiler_single_workflow_ingestion",
            "analysis_mode": "static_ast_non_executing",
            "metrics": {
                "declared_agents": len(agent_inv),
                "implemented_agents": sum(1 for a in agent_inv if a.get("implementation_found")),
                "python_files": len(wf_py_files),
                "telemetry_runs": len(wf_runs),
                "discrepancy_count": comp.get("discrepancy_count", 0),
            },
            "manifest": manifest,
            "agent_inventory": agent_inv,
            "workflow_analysis": wf_analysis,
            "comparison": comp,
            "python_files": wf_py_files,
            "runs": wf_runs,
        }

        # 1. Dedicated JSON reports
        (target_dir / "ingestion_report.json").write_text(json.dumps(wf_report, indent=2, default=str), encoding="utf-8")
        (target_dir / f"{wf_name}_ingestion_report.json").write_text(json.dumps(wf_report, indent=2, default=str), encoding="utf-8")

        # 2. Dedicated agent_inventory.csv
        agent_rows = [
            {
                "workflow": a["workflow"],
                "agent_id": a["id"],
                "role": a["role"],
                "assigned_key": a["assigned_key"],
                "declared_file": a["declared_file"] or "",
                "implementation_file": a["implementation_file"] or "NOT_FOUND",
                "implementation_found": a["implementation_found"],
                "dependencies": "; ".join(a["dependencies"]),
                "inputs": "; ".join(a["inputs"]),
                "outputs": "; ".join(a["outputs"]),
                "parallelizable": a["parallelizable"],
                "pruning_candidate": a["pruning_candidate"] or "",
            }
            for a in agent_inv
        ]
        if agent_rows:
            with open(target_dir / "agent_inventory.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(agent_rows[0].keys()))
                writer.writeheader()
                writer.writerows(agent_rows)

        # 3. Dedicated python_file_inventory.csv
        file_rows = [
            {
                "file": p["file"],
                "parse_status": p["parse_status"],
                "imports": len(p["imports"]),
                "classes": len(p["classes"]),
                "functions": len(p["functions"]),
                "agent_constructors": len(p["agent_definitions"]),
                "task_constructors": len(p["task_definitions"]),
                "crew_constructors": len(p["crew_definitions"]),
                "execution_calls": len(p["execution_calls"]),
                "sha256": p["sha256"],
            }
            for p in wf_py_files
        ]
        if file_rows:
            with open(target_dir / "python_file_inventory.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(file_rows[0].keys()))
                writer.writeheader()
                writer.writerows(file_rows)

        # 4. Dedicated workflow_summary.csv
        wf_summary_rows = [
            {
                "workflow": wf_name,
                "manifest_file": manifest.get("file", ""),
                "workflow_source": wf_analysis.get("file", "") if wf_analysis else "NONE",
                "declared_agents": comp.get("declared_agent_count", len(agent_inv)),
                "observed_agents": comp.get("observed_agent_count", 0),
                "discrepancies": comp.get("discrepancy_count", 0),
            }
        ]
        with open(target_dir / "workflow_summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(wf_summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(wf_summary_rows)

        # 5. Dedicated discrepancies.csv
        disc_rows = [
            {
                "workflow": wf_name,
                "finding_type": f["type"],
                "severity": f.get("severity", "INFO"),
                "details": json.dumps(f, default=str),
            }
            for f in comp.get("findings", [])
        ]
        with open(target_dir / "discrepancies.csv", "w", newline="", encoding="utf-8") as f:
            fieldnames = ["workflow", "finding_type", "severity", "details"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            if disc_rows:
                writer.writerows(disc_rows)

        # 6. Dedicated workflow_summary.md (rich human-readable documentation)
        md_lines = [
            f"# TraceProfiler Ingestion Report: `{wf_name}`",
            "",
            f"**Workflow Name:** `{wf_name}`  ",
            f"**Description:** {manifest.get('description', 'N/A')}  ",
            f"**Assigned Key:** `{manifest.get('agents', [{}])[0].get('assigned_key', 'N/A')}`  ",
            f"**Fixed Model:** `{manifest.get('fixed_model', 'openai/gpt-oss-120b')}`  ",
            f"**Total Agents:** {len(agent_inv)} (Implemented: {sum(1 for a in agent_inv if a.get('implementation_found'))}/{len(agent_inv)})  ",
            "",
            "## 1. Execution Order",
            "",
            "| Step | Agent ID | Status | Implementation File |",
            "|---|---|---|---|",
        ]
        for idx, aid in enumerate(manifest.get("execution_order", []), 1):
            agent_data = next((a for a in agent_inv if a["id"] == aid), {})
            status = "Implemented" if agent_data.get("implementation_found") else "Missing"
            file_link = agent_data.get("implementation_file", "NOT_FOUND")
            md_lines.append(f"| {idx} | `{aid}` | {status} | `{file_link}` |")

        md_lines.extend([
            "",
            "## 2. Architectural Findings & Discrepancies",
            "",
        ])
        if comp.get("findings"):
            for f in comp["findings"]:
                sev = f.get("severity", "INFO")
                md_lines.append(f"- **[{sev}] {f['type']}**: {f.get('note', '')}")
        else:
            md_lines.append("- No discrepancies detected between declared manifest and source code orchestration.")

        md_lines.extend([
            "",
            "## 3. Optimization Hypotheses & Opportunities",
            "",
        ])
        opt_opps = manifest.get("optimization_opportunities", [])
        if opt_opps:
            for opp in opt_opps:
                md_lines.append(f"- {opp}")
        else:
            md_lines.append("- None explicitly declared in manifest.")

        md_lines.extend([
            "",
            "## 4. Telemetry History Artifacts",
            "",
            f"Found **{len(wf_runs)}** run artifacts recorded in `runs/{wf_name}/`.",
            "",
        ])
        for r in wf_runs[:10]:
            run_id = r.get("run_id", "N/A")
            status = r.get("status", "N/A")
            dur = f"{r.get('total_duration_seconds', 0):.2f}s" if r.get('total_duration_seconds') is not None else "N/A"
            md_lines.append(f"- `{r.get('file', '')}` (Run ID: `{run_id}`, Status: `{status}`, Duration: `{dur}`)")

        (target_dir / "workflow_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

        return wf_report


def print_cli_summary(report: Dict[str, Any]):
    """Format and print an executive summary to console."""
    metrics = report["metrics"]
    print("\n" + "=" * 78)
    print(" TRACEPROFILER: AST INGESTION & WORKFLOW REPOSITORY ANALYSIS")
    print("=" * 78)
    print(f" Repository Root        : {report['repository_root']}")
    print(f" Workflows Ingested     : {metrics['total_workflows']}")
    print(f" Total Functional Agents: {metrics['total_declared_agents']}")
    print(f" Implemented Agents     : {metrics['matched_implemented_agents']} ({metrics['agent_match_rate']})")
    print(f" Python Files Parsed    : {metrics['total_python_files']}")
    print(f" Telemetry Artifacts    : {metrics['total_telemetry_artifacts']}")
    print("-" * 78)
    print(f" {'WORKFLOW':<32} {'DECLARED':<10} {'IMPLEMENTED':<12} {'DISCREPANCIES'}")
    print("-" * 78)

    for comp in report["comparisons"]:
        wf = comp["workflow"]
        dec = comp["declared_agent_count"]
        obs = comp["observed_agent_count"]
        disc = comp["discrepancy_count"]
        print(f" {wf:<32} {dec:<10} {obs:<12} {disc}")

    print("-" * 78)
    print(" Architectural Discrepancies & Optimization Findings:")
    has_disc = False
    for comp in report["comparisons"]:
        for f in comp["findings"]:
            has_disc = True
            sev = f.get("severity", "INFO")
            print(f"  [{sev}] {comp['workflow']}: {f['type']} - {f.get('note', '')}")

    if not has_disc:
        print("  None. All declared workflows match implementation execution graphs perfectly.")

    print("=" * 78 + "\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="TraceProfiler AST Multi-Agent Workflow Ingestion")
    parser.add_argument("--root", default=str(Path.cwd()), help="Root directory of repository")
    parser.add_argument("--output-dir", default=None, help="Directory to export JSON and CSV artifacts")
    parser.add_argument("--json", action="store_true", help="Print raw JSON report to stdout")
    args = parser.parse_args()

    analyzer = ASTWorkflowAnalyzer(root_dir=Path(args.root))
    out_dir = analyzer.export_reports(output_dir=Path(args.output_dir) if args.output_dir else None)

    report_path = out_dir / "target_mas_ingestion_report.json"
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_cli_summary(report)
        print(f"Reports saved to: {out_dir}")
        for p in sorted(out_dir.iterdir()):
            if p.is_file():
                print(f"  - {p.name} ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
