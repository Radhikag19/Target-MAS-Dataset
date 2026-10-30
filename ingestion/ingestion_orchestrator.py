"""
Central Ingestion Pipeline Orchestrator for TraceProfiler.

Coordinates Stages 1 through 7 over explicitly supplied Python files.
Generates:
  1. agent_index.json
  2. agent_details.json
  3. code_evidence.json
  4. extraction_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

from ingestion.python_ast_parser import PythonASTParser, ParsedSourceFile
from ingestion.structural_extractor import StructuralExtractor, FileStructuralContext
from ingestion.agent_identifier import AgentIdentifier
from ingestion.model_prompt_tool_extractor import ModelPromptToolExtractor
from ingestion.code_evidence_selector import CodeEvidenceSelector
from ingestion.compact_representation import CompactRepresentationBuilder
from ingestion.context_preparer import TraceProfilerContextPreparer
from schemas.agent_schema import AgentDefinition
from schemas.evidence_schema import EvidenceItem
from schemas.ingestion_schema import IngestionSummary


class IngestionOrchestrator:
    """Orchestrates evidence-based, token-efficient ingestion over explicitly supplied files."""

    def __init__(self):
        self.parser = PythonASTParser()
        self.structural_extractor = StructuralExtractor(self.parser)
        self.agent_identifier = AgentIdentifier(self.parser)
        self.mpt_extractor = ModelPromptToolExtractor(self.parser)
        self.evidence_selector = CodeEvidenceSelector(self.parser)
        self.representation_builder = CompactRepresentationBuilder()

    def process_files(
        self,
        agent_files: List[Union[str, Path]],
        supporting_files: Optional[List[Union[str, Path]]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Tuple[TraceProfilerContextPreparer, Dict[str, Any]]:
        """Run the complete ingestion pipeline over explicitly supplied files."""
        # 1. Resolve and validate files
        agent_paths = [Path(p).resolve() for p in agent_files]
        supp_paths = [Path(p).resolve() for p in (supporting_files or [])]
        all_paths = agent_paths + supp_paths

        missing = [str(p) for p in all_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"The following explicitly supplied files were not found: {missing}")

        # 2. Stage 1: Safe AST Parsing
        parsed_files = self.parser.parse_files(all_paths)
        parsing_errors = [
            {"file": p.filename, "error": p.error}
            for p in parsed_files
            if not p.is_valid and p.error
        ]

        # 3. Stage 2: Structural Extraction
        contexts = self.structural_extractor.extract_all(parsed_files)
        contexts_by_name = {c.filename.replace(".py", ""): c for c in contexts}

        original_sources = {c.filename: c.source for c in contexts}

        # 4. Stage 3 & 4: MAS Agent Identification
        identified_agents: List[AgentDefinition] = []
        all_evidence_items: List[EvidenceItem] = []
        all_unresolved: List[str] = []

        agent_filenames = {p.name for p in agent_paths}

        for ctx in contexts:
            if ctx.filename in agent_filenames:
                agents_in_file = self.agent_identifier.identify_agents(ctx)

                file_models = self.mpt_extractor.extract_models(ctx)
                file_prompts = self.mpt_extractor.extract_prompts(ctx)
                file_tools, file_helpers = self.mpt_extractor.extract_tools_and_helpers(ctx)
                file_inputs, file_task_outputs, file_return_schemas = self.mpt_extractor.extract_inputs_and_outputs(ctx)
                file_exec_pattern = self.mpt_extractor.extract_execution_patterns(ctx)

                for agent in agents_in_file:
                    agent.model_references = file_models
                    agent.prompt_references = file_prompts
                    agent.tool_references = file_tools          # Strictly registered tools
                    agent.helper_functions = file_helpers       # Ordinary helper functions
                    agent.execution_pattern = file_exec_pattern # Direct LLM vs workflow vs tool calls
                    agent.inputs = file_inputs
                    agent.task_expected_outputs = file_task_outputs
                    agent.return_schemas = file_return_schemas

                    # Stage 5: AST-Guided Evidence Selection
                    ev_items, unresolved = self.evidence_selector.select_evidence(
                        ctx, agent.agent_id, all_contexts=contexts_by_name
                    )
                    agent.evidence_ids = [ev.evidence_id for ev in ev_items]
                    agent.unresolved_references = unresolved

                    all_evidence_items.extend(ev_items)
                    all_unresolved.extend(unresolved)
                    identified_agents.append(agent)

        # 5. Stage 6: Compact Representation & Token Efficiency
        compact_index = self.representation_builder.build_compact_index(identified_agents)
        compact_index_data = [e.to_dict() for e in compact_index]

        agent_details_data = self.representation_builder.build_agent_details(identified_agents)
        code_evidence_data = self.representation_builder.build_code_evidence_store(all_evidence_items)

        token_metrics = self.representation_builder.calculate_token_metrics(
            original_sources=original_sources,
            compact_index_data=compact_index_data,
            agent_details_data=agent_details_data,
            code_evidence_data=code_evidence_data,
        )

        external_dependencies = list(dict.fromkeys([
            mod for ctx in contexts for mod in ctx.external_modules
        ]))

        true_unresolved = list(dict.fromkeys(
            [u for ctx in contexts for u in ctx.unresolved_references] + all_unresolved
        ))

        summary = IngestionSummary(
            files_analyzed=[p.name for p in all_paths],
            agents_identified=len(identified_agents),
            evidence_items_count=len(all_evidence_items),
            parsing_errors=parsing_errors,
            unresolved_references=true_unresolved,
            external_dependencies=external_dependencies,
            token_metrics=token_metrics,
        )
        summary_data = summary.to_dict()

        # 6. Stage 7: Context Preparer Engine
        preparer = TraceProfilerContextPreparer(
            agent_index=compact_index_data,
            agent_details=agent_details_data,
            code_evidence=code_evidence_data,
        )

        # 7. Write Expected Outputs if output_dir is provided
        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            (out_path / "agent_index.json").write_text(
                json.dumps(compact_index_data, indent=2, default=str), encoding="utf-8"
            )
            (out_path / "agent_details.json").write_text(
                json.dumps(agent_details_data, indent=2, default=str), encoding="utf-8"
            )
            (out_path / "code_evidence.json").write_text(
                json.dumps(code_evidence_data, indent=2, default=str), encoding="utf-8"
            )
            (out_path / "extraction_summary.json").write_text(
                json.dumps(summary_data, indent=2, default=str), encoding="utf-8"
            )

        return preparer, summary_data


def main():
    parser = argparse.ArgumentParser(description="TraceProfiler Evidence-Based AST Ingestion Pipeline")
    parser.add_argument("--files", nargs="+", required=True, help="Explicit list of agent Python files")
    parser.add_argument("--supporting-files", nargs="*", default=[], help="Optional list of supporting tool/utility/workflow files")
    parser.add_argument("--output-dir", default="reports/traceprofiler_evidence", help="Output directory for generated JSON files")
    args = parser.parse_args()

    orchestrator = IngestionOrchestrator()
    preparer, summary = orchestrator.process_files(
        agent_files=args.files,
        supporting_files=args.supporting_files,
        output_dir=args.output_dir,
    )

    tm = summary["token_metrics"]
    print("\n" + "=" * 70)
    print(" TRACEPROFILER AST INGESTION COMPLETE")
    print("=" * 70)
    print(f" Files Analyzed        : {len(summary['files_analyzed'])}")
    print(f" Agents Identified     : {summary['agents_identified']}")
    print(f" Evidence Snippets     : {summary['evidence_items_count']}")
    print(f" External Dependencies : {len(summary['external_dependencies'])} {summary['external_dependencies']}")
    print(f" Unresolved References : {len(summary['unresolved_references'])} {summary['unresolved_references']}")
    print("-" * 70)
    print(f" TOKEN MEASUREMENT ({'ACTUAL TOKENIZER' if tm['is_actual_tokenizer'] else 'FALLBACK ESTIMATE'}):")
    print(f"  Method               : {tm['token_measurement_method']}")
    print(f"  Original Source Code : {tm['original_source_tokens']:,} tokens")
    print(f"  Compact Agent Index  : {tm['compact_index_tokens']:,} tokens")
    print(f"  Token Reduction      : {tm['token_reduction_count']:,} tokens ({tm['percentage_reduction']}%)")
    print("-" * 70)
    print(f" Outputs saved to: {args.output_dir}")
    print("  - agent_index.json      (Level 1 Initial Context)")
    print("  - agent_details.json    (Level 2 Detailed Structural Records)")
    print("  - code_evidence.json    (Level 2 Code Snippet Store)")
    print("  - extraction_summary.json")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
