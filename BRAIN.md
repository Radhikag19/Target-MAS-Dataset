# Master Architectural & Technical Blueprint (`BRAIN.md`)
## Multi-Agent Baseline Systems for Agentic Workflow Optimizer (AWO)

---

## 1. Executive Summary & Architectural Vision

This repository provides **seven fully functional, production-grade multi-agent baseline workflows** comprising **exactly 54 unique agents** designed to serve as the ground-truth benchmark, testing harness, and mutation substrate for the **Agentic Workflow Optimizer (AWO)**.

The primary objective is to evaluate how an automated meta-agent optimizer can analyze, prune, consolidate, parallelize, and optimize multi-agent baseline systems across complex business domains without sacrificing task accuracy, compliance, or decision quality.

### Core Architectural Principles
1. **Strict Model Uniformity:** Every agent across all seven workflows uses Groq's `openai/gpt-oss-120b` model exclusively. Dynamic model switching, fallbacks, or heterogeneous models are prohibited.
2. **Dedicated Workflow Credential Isolation:** Each workflow is assigned exactly one dedicated Groq API key (`GROQ_API_KEY_1` through `GROQ_API_KEY_7`), ensuring clean credential isolation, independent rate limiting, and zero cross-workflow contagion.
3. **Deterministic Core + Cognitive Reasoning:** All arithmetic, 3-way matching, policy thresholds, cost comparisons, weighted scoring matrices, EOQ/MOQ rounding, and qualification rubrics are implemented in pure, deterministic Python. The LLM is reserved for semantic understanding, classification, synthesis, and decision explanation.
4. **100% Local Synthetic Datasets:** All data is generated programmatically and stored locally under `data/`. Zero external APIs, web scraping, or LLM calls are used for data generation.
5. **Observability via Arize Phoenix:** Native OpenTelemetry spans and OpenInference instrumentation capture trace graphs, execution latencies, agent hierarchies, and token counts at `http://localhost:6006` with zero cloud credentials required.
6. **Canonical Workflow Representation (CWR) Readiness:** Every workflow contains an optimizer-ready `workflow.json` manifest specifying its execution graph, agent dependencies, input/output contracts, and explicit optimization hypotheses.

---

## 2. Complete Project Directory & File Manifest

```text
Agentic_AI_Project/
├── .env                                       # Local environment variables (Groq keys 1-7, Phoenix settings)
├── .env.example                               # Template for required environment variables
├── requirements.txt                           # Pinned dependencies (crewai, litellm, groq, arize-phoenix, etc.)
├── main.py                                    # Central CLI entry point for listing, validating, and running workflows
├── generate_dummy_data.py                     # Deterministic, offline synthetic dataset generator (seed=42)
├── README.md                                  # User-facing setup, quickstart, and operation guide
├── BRAIN.md                                   # Master architectural & technical documentation (This file)
│
├── common/                                    # Shared infrastructure and utilities
│   ├── settings.py                            # Mandatory model enforcement, paths, and ConfigurationError
│   ├── api_key_manager.py                     # Workflow Groq API key isolation, validation, and safe masking
│   ├── agent_runner.py                        # CrewAI LLM factory, role-to-ID resolver, and task execution engine
│   ├── tracing.py                             # Arize Phoenix tracing integration and OpenInference instrumentation
│   ├── metrics.py                             # MetricsTracker, latency/token recording, and run artifact persistence
│   └── validation.py                          # Pydantic schemas, JSON extraction, and data integrity validators
│
├── Agents/                                    # Baseline multi-agent system implementations (54 agents total)
│   ├── invoice_processing/                    # Workflow 1: 8 Baseline Agents (GROQ_API_KEY_1)
│   │   ├── invoice_extraction_agent.py        # Agent 1: Structured document parser
│   │   ├── invoice_validation_agent.py        # Agent 2: Arithmetic & field completeness auditor
│   │   ├── po_matching_agent.py               # Agent 3: 3-way purchase order reconciliation specialist
│   │   ├── policy_compliance_agent.py         # Agent 4: Corporate AP policy & threshold auditor
│   │   ├── duplicate_detection_agent.py       # Agent 5: Historical payment & duplicate billing analyst
│   │   ├── exception_analysis_agent.py        # Agent 6: Discrepancy & operational risk consolidator
│   │   ├── approval_recommendation_agent.py   # Agent 7: Fiduciary payment decision authority
│   │   ├── final_review_agent.py              # Agent 8: AP audit trail verification & executive reporting
│   │   ├── workflow.py                        # Sequential 8-agent execution pipeline
│   │   └── workflow.json                      # CWR-ready machine-readable workflow manifest
│   │
│   ├── customer_support/                      # Workflow 2: 6 Baseline Agents (GROQ_API_KEY_2)
│   │   ├── ticket_classification_agent.py     # Agent 1: Inquiry categorization & complexity triage
│   │   ├── policy_retrieval_agent.py          # Agent 2: Support SLA & troubleshooting SOP retriever
│   │   ├── issue_investigation_agent.py       # Agent 3: Customer history & technical root cause investigator
│   │   ├── resolution_agent.py                # Agent 4: Remediation strategist & action planner
│   │   ├── response_drafting_agent.py         # Agent 5: Customer-facing communications specialist
│   │   ├── resolution_verification_agent.py   # Agent 6: Quality assurance auditor & final reviewer
│   │   ├── workflow.py                        # Sequential 6-agent execution pipeline
│   │   └── workflow.json                      # CWR-ready machine-readable workflow manifest
│   │
│   ├── procurement_supplier_selection/        # Workflow 3: 9 Baseline Agents (GROQ_API_KEY_3)
│   │   ├── quotation_analysis_agent.py        # Agent 1: Commercial quotation normalizer
│   │   ├── supplier_profile_agent.py          # Agent 2: Vendor capability & capacity auditor
│   │   ├── delivery_analysis_agent.py         # Agent 3: Logistics & fulfillment schedule analyst
│   │   ├── supplier_risk_agent.py             # Agent 4: Quality defect & vendor reliability auditor
│   │   ├── contract_analysis_agent.py         # Agent 5: Legal terms, warranty & penalty specialist
│   │   ├── cost_comparison_agent.py           # Agent 6: Landed cost calculator & financial ranker
│   │   ├── weighted_evaluation_agent.py       # Agent 7: Multi-criteria decision analysis (MCDA) evaluator
│   │   ├── selection_recommendation_agent.py  # Agent 8: Sourcing committee award strategist
│   │   ├── final_review_agent.py              # Agent 9: Procurement governance & final reporting lead
│   │   ├── workflow.py                        # Sequential 9-agent execution pipeline
│   │   └── workflow.json                      # CWR-ready machine-readable workflow manifest
│   │
│   ├── sales_lead_qualification/              # Workflow 4: 7 Baseline Agents (GROQ_API_KEY_4)
│   │   ├── lead_enrichment_agent.py           # Agent 1: Firmographic & contact data normalizer
│   │   ├── lead_fit_agent.py                  # Agent 2: Ideal Customer Profile (ICP) match evaluator
│   │   ├── intent_analysis_agent.py           # Agent 3: Behavioral interaction & buying signal analyst
│   │   ├── lead_scoring_agent.py              # Agent 4: 100-point multi-signal scoring model calculator
│   │   ├── opportunity_assessment_agent.py    # Agent 5: Commercial viability & deal feasibility assessor
│   │   ├── next_action_agent.py               # Agent 6: RevOps engagement & pipeline routing strategist
│   │   ├── final_review_agent.py              # Agent 7: Pipeline hygiene & executive audit reviewer
│   │   ├── workflow.py                        # Sequential 7-agent execution pipeline
│   │   └── workflow.json                      # CWR-ready machine-readable workflow manifest
│   │
│   ├── recruitment_screening/                 # Workflow 5: 8 Baseline Agents (GROQ_API_KEY_5)
│   │   ├── intake_agent.py                    # Agent 1: Resume intake & section normalizer
│   │   ├── qualification_agent.py             # Agent 2: Candidate qualification extractor (linked evidence)
│   │   ├── job_criteria_agent.py              # Agent 3: Job criteria analyst & requirements deconstructor
│   │   ├── evidence_mapping_agent.py          # Agent 4: Criteria-to-qualification evidence mapper
│   │   ├── experience_assessment_agent.py     # Agent 5: Career trajectory & tenure relevance evaluator
│   │   ├── missing_evidence_agent.py          # Agent 6: Inconsistency & unverified claim detector
│   │   ├── review_agent.py                    # Agent 7: Screening quality & compliance auditor
│   │   ├── recruiter_summary_agent.py         # Agent 8: Recruiter briefing & deterministic score synthesizer
│   │   ├── workflow.py                        # Sequential 8-agent execution pipeline
│   │   └── workflow.json                      # CWR-ready machine-readable workflow manifest
│   │
│   ├── inventory_replenishment/               # Workflow 6: 6 Baseline Agents (GROQ_API_KEY_6)
│   │   ├── intake_agent.py                    # Agent 1: SKU inventory intake coordinator
│   │   ├── stock_policy_agent.py              # Agent 2: Stock policy & usable inventory level specialist
│   │   ├── demand_forecast_agent.py           # Agent 3: Demand forecasting & sales history analyst
│   │   ├── supplier_lead_time_agent.py        # Agent 4: Supplier lead time, MOQ & pack size specialist
│   │   ├── replenishment_planning_agent.py    # Agent 5: Replenishment quantity & deficit optimization planner
│   │   ├── review_audit_agent.py              # Agent 6: Replenishment integrity & double-counting auditor
│   │   ├── workflow.py                        # Sequential 6-agent execution pipeline
│   │   └── workflow.json                      # CWR-ready machine-readable workflow manifest
│   │
│   └── contract_review/                       # Workflow 7: 10 Baseline Agents (GROQ_API_KEY_7)
│       ├── intake_agent.py                    # Agent 1: Contract intake & document ingestion specialist
│       ├── clause_extraction_agent.py         # Agent 2: Contract clause extraction & section cataloger
│       ├── policy_interpretation_agent.py     # Agent 3: Enterprise compliance policy interpreter
│       ├── clause_mapping_agent.py            # Agent 4: Clause-to-policy alignment & coverage specialist
│       ├── deviation_agent.py                 # Agent 5: Contract compliance deviation & missing term evaluator
│       ├── risk_ambiguity_agent.py            # Agent 6: Contract ambiguity & latent operational risk analyst
│       ├── consistency_agent.py               # Agent 7: Cross-clause conflict & auto-renewal auditor
│       ├── evidence_validation_agent.py       # Agent 8: Legal citation & excerpt validator
│       ├── review_agent.py                    # Agent 9: Independent legal counsel review & severity calibrator
│       ├── final_report_agent.py              # Agent 10: Executive legal compliance report author
│       ├── workflow.py                        # Sequential 10-agent execution pipeline
│       └── workflow.json                      # CWR-ready machine-readable workflow manifest
│
├── data/                                      # Local synthetic dataset repository
│   ├── invoice_processing/                    # Invoices, POs, suppliers, AP policies, payment history
│   ├── customer_support/                      # Tickets, customer profiles, policies, SOPs, resolutions
│   ├── procurement_supplier_selection/        # RFQs, vendor quotes, delivery timelines, contracts, risks
│   ├── sales_lead_qualification/              # Leads, company profiles, ICPs, engagement events, rules
│   ├── recruitment_screening/                 # Resumes, jobs, criteria profiles, candidate evidence
│   ├── inventory_replenishment/               # Inventory snapshots, sales history, supplier constraints, policies
│   └── contract_review/                       # Full contract texts (MSA, NDA, DPA, SLA) and enterprise policies
│
├── runs/                                      # Structured JSON execution logs ({run_id}.json)
└── tests/                                     # Automated test suite (36 unit, logic & mocked integration tests)
    ├── test_structure_and_manifests.py        # Validates 54 unique agents, 7 manifests, and manifest schemas
    ├── test_dataset_generator.py              # Validates reproducibility and zero-LLM generation
    ├── test_api_key_manager.py                # Validates key isolation, safe masking, and error handling
    ├── test_deterministic_logic.py            # Validates deterministic arithmetic, logic, and scoring rubrics
    ├── test_mocked_workflows.py               # Validates end-to-end execution of baseline workflows
    ├── test_recruitment_screening.py          # Validates recruitment screening agent count, logic & mock run
    ├── test_inventory_replenishment.py        # Validates inventory replenishment formulas, MOQ & mock run
    ├── test_contract_review.py                # Validates contract review agent count, schemas & mock run
    └── test_shared_infrastructure.py          # Validates model enforcement, JSON parser, and Phoenix tracing
```

---

## 3. Technology Stack & Runtime Configuration

### 3.1 Frameworks and Libraries
- **Language:** Python 3.12 (CPython x86_64)
- **Multi-Agent Framework:** CrewAI (`v1.15.28`)
- **LLM Routing Engine:** LiteLLM (`v1.106.0`) interfacing with Groq API
- **Inference Model:** `openai/gpt-oss-120b` (Groq hosted)
- **Observability:** Arize Phoenix (`v20.23.0`) + OpenInference CrewAI Instrumentor (`v1.1.21`)
- **Validation & Data Structures:** Pydantic (`v2.12.5`), Python standard library (`json`, `dataclasses`)
- **Test Harness:** Pytest (`v9.1.1`)

### 3.2 Groq API Parameter Normalization
CrewAI `1.15.28` introduces prompt caching mechanisms that inject internal `cache_breakpoint` metadata into message payloads. While standard Anthropic endpoints accept cache breakpoints, Groq's OpenAI-compatible endpoint strictly validates input dictionaries against standard OpenAI chat message schemas and rejects any dictionary containing `cache_breakpoint`.

To guarantee 100% API compatibility with Groq without altering CrewAI core internals, [`common/agent_runner.py`](file:///c:/Users/radhi/Desktop/Agentic_AI_Project/common/agent_runner.py) and [`common/settings.py`](file:///c:/Users/radhi/Desktop/Agentic_AI_Project/common/settings.py) apply two strategic interventions:
1. `litellm.drop_params = True`: Directs LiteLLM to strip unsupported provider parameters before dispatching requests.
2. Neutralizes `crewai.llms.cache.mark_cache_breakpoint = lambda msg: msg`: Ensures message dictionaries constructed by agent executors remain pristine, standard OpenAI-compatible objects.

---

## 4. Master Workflow Catalog & Architecture Matrix

| # | Workflow Name | CLI Alias | Assigned Groq Key | Agent Count | Primary Input Data | Final Deliverable |
| :-: | :--- | :---: | :--- | :---: | :--- | :--- |
| **1** | **Invoice Processing & AP** | `invoice` | `GROQ_API_KEY_1` | **8** | Invoices, Purchase Orders, AP Policies | Fiduciary payment recommendation & audit trail |
| **2** | **Customer Support Resolution** | `support` | `GROQ_API_KEY_2` | **6** | Support Tickets, Customer History, SOPs | Customer response draft & QA verification |
| **3** | **Procurement & Sourcing** | `procurement` | `GROQ_API_KEY_3` | **9** | RFQ, Supplier Quotes, Delivery Terms | Weighted MCDA award recommendation & contract risk |
| **4** | **Sales Lead Qualification** | `sales` | `GROQ_API_KEY_4` | **7** | Inbound Leads, Company Profiles, ICPs | 100-point lead grade & RevOps routing decision |
| **5** | **Recruitment Screening** | `recruitment` | `GROQ_API_KEY_5` | **8** | Resumes, Job Criteria, Evidence IDs | Recruiter summary report & deterministic match score |
| **6** | **Inventory Replenishment** | `inventory` | `GROQ_API_KEY_6` | **6** | SKU Records, Demand History, Supplier Terms | Recommended reorder quantity & stockout risk |
| **7** | **Contract Review & Compliance** | `contract` | `GROQ_API_KEY_7` | **10** | Contracts (MSA/NDA/SLA), Compliance Policies | Comprehensive legal compliance audit & deviation report |

**Grand Total: 7 Workflows, 54 Unique Agents, 7 Isolated API Keys, 1 Uniform Model (`openai/gpt-oss-120b`).**

---

## 5. Detailed Agent Specifications: Workflow 1 - Invoice Processing (8 Agents)

- **Workflow Name:** `invoice_processing`
- **Assigned Credential:** `GROQ_API_KEY_1`
- **Input Data:** `data/invoice_processing/`

### Agent Breakdown:
1. **Invoice Extraction Agent (`invoice_extraction_agent`):** Extracts vendor identifiers, line items, net totals, tax amounts, and payment terms from unstructured invoice records.
2. **Invoice Validation Agent (`invoice_validation_agent`):** Verifies arithmetic integrity: checks that `sum(line_items) + tax_amount == total_amount` within 1 cent tolerance. Flags fatal calculation errors.
3. **PO Matching Agent (`po_matching_agent`):** Performs deterministic 3-way reconciliation against `purchase_orders.json`. Detects quantity overruns, price inflation, and missing POs.
4. **Policy Compliance Agent (`policy_compliance_agent`):** Audits invoice against corporate AP policies: payment threshold limits, restricted vendor status (`SUP-109`), and authorization requirements.
5. **Duplicate Detection Agent (`duplicate_detection_agent`):** Evaluates historical payment logs in `payment_history.json` to prevent duplicate billing.
6. **Exception Analysis Agent (`exception_analysis_agent`):** Consolidates discrepancies, calculates dispute severity, and synthesizes root-cause summaries.
7. **Approval Recommendation Agent (`approval_recommendation_agent`):** Deterministic fiduciary payment authority recommending `APPROVE`, `REJECT`, or `NEEDS_REVIEW`.
8. **Final Review Agent (`final_review_agent_invoice`):** Audits complete pipeline execution, verifies reasoning integrity, and compiles the final executive AP report.

---

## 6. Detailed Agent Specifications: Workflow 2 - Customer Support (6 Agents)

- **Workflow Name:** `customer_support`
- **Assigned Credential:** `GROQ_API_KEY_2`
- **Input Data:** `data/customer_support/`

### Agent Breakdown:
1. **Ticket Classification Agent (`ticket_classification_agent`):** Ingests customer support tickets, classifies category (`billing`, `technical`, `account_access`, `feature_request`), and assigns urgency.
2. **Policy Retrieval Agent (`policy_retrieval_agent`):** Retrieves applicable support policies, SLA targets, and troubleshooting Standard Operating Procedures (SOPs).
3. **Issue Investigation Agent (`issue_investigation_agent`):** Ingests customer tier profile, subscription history, and past tickets to identify technical or operational root causes.
4. **Resolution Agent (`resolution_agent`):** Formulates precise, policy-compliant remediation plans (refunds, password resets, configuration fixes).
5. **Response Drafting Agent (`response_drafting_agent`):** Composes professional, empathetic, customer-facing response emails with clear action steps.
6. **Resolution Verification Agent (`resolution_verification_agent`):** Audits drafted response against SLA policies, ensures safety and clarity, and signs off on ticket closure.

---

## 7. Detailed Agent Specifications: Workflow 3 - Procurement & Sourcing (9 Agents)

- **Workflow Name:** `procurement_supplier_selection`
- **Assigned Credential:** `GROQ_API_KEY_3`
- **Input Data:** `data/procurement_supplier_selection/`

### Agent Breakdown:
1. **Quotation Analysis Agent (`quotation_analysis_agent`):** Ingests supplier quotations, normalizes unit pricing, tax rates, and currency terms.
2. **Supplier Profile Agent (`supplier_profile_agent`):** Audits vendor manufacturing capacity, industry certifications (ISO 9001), and historical performance.
3. **Delivery Analysis Agent (`delivery_analysis_agent`):** Analyzes proposed delivery schedules, shipping modes, lead times, and on-time reliability metrics.
4. **Supplier Risk Agent (`supplier_risk_agent`):** Evaluates supplier financial stability, geopolitical risk, and component defect rates.
5. **Contract Analysis Agent (`contract_analysis_agent`):** Reviews vendor contract proposals, warranty terms, liability caps, and cancellation penalties.
6. **Cost Comparison Agent (`cost_comparison_agent`):** Computes Total Landed Cost (TLC) incorporating base price, shipping, tariffs, and payment discounts.
7. **Weighted Evaluation Agent (`weighted_evaluation_agent`):** Executes deterministic Multi-Criteria Decision Analysis (MCDA) across cost, quality, delivery, and risk dimensions.
8. **Selection Recommendation Agent (`selection_recommendation_agent`):** Synthesizes MCDA rankings into formal procurement award justifications.
9. **Final Review Agent (`final_review_agent_procurement`):** Conducts governance review, verifies procurement policy adherence, and compiles the final award dossier.

---

## 8. Detailed Agent Specifications: Workflow 4 - Sales Lead Qualification (7 Agents)

- **Workflow Name:** `sales_lead_qualification`
- **Assigned Credential:** `GROQ_API_KEY_4`
- **Input Data:** `data/sales_lead_qualification/`

### Agent Breakdown:
1. **Lead Enrichment Agent (`lead_enrichment_agent`):** Ingests inbound sales leads and enriches contact details with company firmographics (revenue, employees, industry).
2. **Lead Fit Agent (`lead_fit_agent`):** Evaluates firmographics against Ideal Customer Profile (ICP) criteria; immediately flags disqualified industries.
3. **Intent Analysis Agent (`intent_analysis_agent`):** Analyzes digital behavioral engagement (pricing visits, demo requests, whitepaper downloads) to measure buyer intent.
4. **Lead Scoring Agent (`lead_scoring_agent`):** Computes deterministic 100-point qualification score across fit, seniority, intent, and budget.
5. **Opportunity Assessment Agent (`opportunity_assessment_agent`):** Synthesizes fit, intent, and scoring into commercial opportunity size and viability ratings.
6. **Next Action Agent (`next_action_agent`):** Recommends optimal next sales step: executive demo, SDR outreach, nurture drip, or disqualification.
7. **Final Review Agent (`final_review_agent_sales`):** Validates lead qualification pipeline, verifies handoff validity, and produces executive sales summary.

---

## 9. Detailed Agent Specifications: Workflow 5 - Recruitment Screening (8 Agents)

- **Workflow Name:** `recruitment_screening`
- **Assigned Credential:** `GROQ_API_KEY_5`
- **Input Data:** `data/recruitment_screening/`

### Agent Breakdown:
1. **Resume Intake Agent (`resume_intake_agent`):**
   - **File:** [`Agents/recruitment_screening/intake_agent.py`](file:///c:/Users/radhi/Desktop/Agentic_AI_Project/Agents/recruitment_screening/intake_agent.py)
   - **Role:** Resume Intake Specialist
   - **Responsibilities:** Normalizes raw resume text, catalogs candidate sections (education, experience, skills), and verifies document completeness.
   - **Inputs:** `application_id`, `candidate_id`, `sections`, `source_evidence_ids`.
   - **Outputs:** `application_id`, `candidate_id`, `candidate_name`, `sections_identified`, `missing_sections`, `raw_sections`.

2. **Qualification Extraction Agent (`qualification_agent`):**
   - **File:** [`Agents/recruitment_screening/qualification_agent.py`](file:///c:/Users/radhi/Desktop/Agentic_AI_Project/Agents/recruitment_screening/qualification_agent.py)
   - **Role:** Candidate Qualification Extractor
   - **Responsibilities:** Extracts technical skills, academic degrees, certifications, and experience records strictly tied to ground-truth evidence IDs (`EVD-*`). Never hallucinates qualifications.
   - **Inputs:** `candidate_id`, `raw_sections`, `source_evidence_ids`.
   - **Outputs:** `technical_skills`, `education`, `certifications`, `experience_records`, `projects`.

3. **Job Criteria Agent (`job_criteria_agent`):**
   - **File:** [`Agents/recruitment_screening/job_criteria_agent.py`](file:///c:/Users/radhi/Desktop/Agentic_AI_Project/Agents/recruitment_screening/job_criteria_agent.py)
   - **Role:** Job Criteria Analyst
   - **Responsibilities:** Deconstructs job postings into discrete mandatory requirements (must-have) and preferred criteria (nice-to-have) with stable criterion IDs.
   - **Inputs:** `job_id`, `title`, `mandatory_criteria`, `preferred_criteria`.
   - **Outputs:** `job_id`, `job_title`, `mandatory_criteria`, `preferred_criteria`, `total_criteria_count`.

4. **Evidence Mapping Agent (`evidence_mapping_agent`):**
   - **File:** [`Agents/recruitment_screening/evidence_mapping_agent.py`](file:///c:/Users/radhi/Desktop/Agentic_AI_Project/Agents/recruitment_screening/evidence_mapping_agent.py)
   - **Role:** Candidate Evidence Mapping Specialist
   - **Responsibilities:** Aligns extracted candidate qualifications against job criteria, classifying each status as `supported`, `partially_supported`, or `unsupported` while preserving citation links.
   - **Inputs:** `technical_skills`, `education`, `experience_records`, `mandatory_criteria`, `preferred_criteria`.
   - **Outputs:** `mappings`, `supported_count`, `unsupported_count`, `partially_supported_count`.

5. **Experience Assessment Agent (`experience_assessment_agent`):**
   - **File:** [`Agents/recruitment_screening/experience_assessment_agent.py`](file:///c:/Users/radhi/Desktop/Agentic_AI_Project/Agents/recruitment_screening/experience_assessment_agent.py)
   - **Role:** Experience Assessment Evaluator
   - **Responsibilities:** Audits employment duration, role seniority progression, domain relevance, and identifies transferable skills and resume tenure gaps.
   - **Inputs:** `experience_records`, `mappings`.
   - **Outputs:** `relevance_summary`, `strengths`, `experience_gaps`, `transferable_skills`.

6. **Missing Evidence and Consistency Agent (`missing_evidence_agent`):**
   - **File:** [`Agents/recruitment_screening/missing_evidence_agent.py`](file:///c:/Users/radhi/Desktop/Agentic_AI_Project/Agents/recruitment_screening/missing_evidence_agent.py)
   - **Role:** Recruitment Consistency and Verification Auditor
   - **Responsibilities:** Flags timeline contradictions, unsupported buzzwords, missing evidence citations, and claims lacking verifiable tenure backing.
   - **Inputs:** `raw_sections`, `mappings`.
   - **Outputs:** `missing_evidence_items`, `unreliable_requirements`, `inconsistencies`.

7. **Screening Review Agent (`screening_review_agent`):**
   - **File:** [`Agents/recruitment_screening/review_agent.py`](file:///c:/Users/radhi/Desktop/Agentic_AI_Project/Agents/recruitment_screening/review_agent.py)
   - **Role:** Screening Quality and Integrity Reviewer
   - **Responsibilities:** Performs independent quality review: verifies that mandatory criteria are verified, flags internal audit contradictions, and isolates topics requiring human recruiter follow-up.
   - **Inputs:** `mappings`, `strengths`, `experience_gaps`, `inconsistencies`.
   - **Outputs:** `review_passed`, `contradictions_detected`, `traceability_verified`, `mandatory_criteria_met`, `issues_requiring_human_review`.

8. **Recruiter Summary Agent (`recruiter_summary_agent`):**
   - **File:** [`Agents/recruitment_screening/recruiter_summary_agent.py`](file:///c:/Users/radhi/Desktop/Agentic_AI_Project/Agents/recruitment_screening/recruiter_summary_agent.py)
   - **Role:** Executive Recruiter Summary Synthesizer
   - **Responsibilities:** Generates the final candidate screening summary, recommended interview follow-up questions, and executes `calculate_deterministic_match_score` (weighted ratio of supported criteria). Emits mandatory non-autonomous decision disclaimer.
   - **Inputs:** Upstream candidate overview, criteria matrix, strengths, gaps, review issues.
   - **Outputs:** `run_id`, `application_id`, `candidate_name`, `criteria_matrix`, `deterministic_match_score`, `human_review_required`.

---

## 10. Detailed Agent Specifications: Workflow 6 - Inventory Replenishment (6 Agents)

- **Workflow Name:** `inventory_replenishment`
- **Assigned Credential:** `GROQ_API_KEY_6`
- **Input Data:** `data/inventory_replenishment/`

### Agent Breakdown:
1. **Inventory Intake Agent (`inventory_intake_agent`):**
   - **File:** [`Agents/inventory_replenishment/intake_agent.py`](file:///c:/Users/radhi/Desktop/Agentic_AI_Project/Agents/inventory_replenishment/intake_agent.py)
   - **Role:** Inventory Intake Coordinator
   - **Responsibilities:** Validates SKU identifiers, normalizes stock records, and aggregates warehouse snapshots, sales velocity, and vendor records.
   - **Inputs:** `sku_record`, `supplier_record`, `policy_record`, `sales_record`.
   - **Outputs:** `sku_id`, `warehouse_id`, `source_record_ids`, `raw_inventory_record`, `raw_supplier_record`.

2. **Stock and Reorder Policy Agent (`stock_policy_agent`):**
   - **File:** [`Agents/inventory_replenishment/stock_policy_agent.py`](file:///c:/Users/radhi/Desktop/Agentic_AI_Project/Agents/inventory_replenishment/stock_policy_agent.py)
   - **Role:** Stock Policy and Inventory Level Specialist
   - **Responsibilities:** Calculates usable inventory via deterministic arithmetic: `usable_inventory = current_stock - reserved_stock + incoming_stock`. Evaluates against reorder threshold and safety stock targets. Prevents incoming stock double-counting.
   - **Inputs:** `raw_inventory_record`, `warehouse_id`.
   - **Outputs:** `current_stock`, `reserved_stock`, `incoming_stock`, `usable_inventory`, `reorder_threshold`, `safety_stock`, `replenishment_needed`.

3. **Demand Forecast Agent (`demand_forecast_agent`):**
   - **File:** [`Agents/inventory_replenishment/demand_forecast_agent.py`](file:///c:/Users/radhi/Desktop/Agentic_AI_Project/Agents/inventory_replenishment/demand_forecast_agent.py)
   - **Role:** Demand Forecasting and Sales History Specialist
   - **Responsibilities:** Analyzes daily sales records, computes historical average daily demand and standard deviation, and projects projected demand across the forecast horizon.
   - **Inputs:** `sales_record`, `forecast_horizon_days`.
   - **Outputs:** `historical_avg_daily_demand`, `forecast_demand_units`, `volatility_metric`, `demand_spikes_detected`.

4. **Supplier and Lead-Time Agent (`supplier_agent`):**
   - **File:** [`Agents/inventory_replenishment/supplier_lead_time_agent.py`](file:///c:/Users/radhi/Desktop/Agentic_AI_Project/Agents/inventory_replenishment/supplier_lead_time_agent.py)
   - **Role:** Supplier Lead-Time and Constraints Specialist
   - **Responsibilities:** Verifies vendor lead time, minimum order quantity (MOQ), pack size increments, and computes expected order arrival dates.
   - **Inputs:** `raw_supplier_record`, `planning_date`.
   - **Outputs:** `lead_time_days`, `minimum_order_quantity`, `pack_size`, `expected_arrival_date`.

5. **Replenishment Planning Agent (`replenishment_planning_agent`):**
   - **File:** [`Agents/inventory_replenishment/replenishment_planning_agent.py`](file:///c:/Users/radhi/Desktop/Agentic_AI_Project/Agents/inventory_replenishment/replenishment_planning_agent.py)
   - **Role:** Replenishment Optimization Planner
   - **Responsibilities:** Calculates gross inventory deficit: `gross_requirement = (avg_daily_demand * lead_time) + safety_stock`. Applies pack-size ceiling rounding and MOQ enforcement: `recommended_order_quantity = max(MOQ, ceil(deficit / pack_size) * pack_size)`.
   - **Inputs:** `usable_inventory`, `historical_avg_daily_demand`, `lead_time_days`, `safety_stock`, `pack_size`, `minimum_order_quantity`.
   - **Outputs:** `base_replenishment_quantity`, `recommended_order_quantity`, `calculation_breakdown`, `stockout_risk`.

6. **Validation and Final Review Agent (`inventory_review_agent`):**
   - **File:** [`Agents/inventory_replenishment/review_audit_agent.py`](file:///c:/Users/radhi/Desktop/Agentic_AI_Project/Agents/inventory_replenishment/review_audit_agent.py)
   - **Role:** Replenishment Integrity and Compliance Auditor
   - **Responsibilities:** Verifies replenishment mathematics against double-counting, audits stockout risk classifications, attaches planning-tool disclaimers, and compiles final purchase requisition plan.
   - **Inputs:** `base_replenishment_quantity`, `recommended_order_quantity`, `stockout_risk`, `stock_policy_data`.
   - **Outputs:** `run_id`, `planning_date`, `warehouse_id`, `sku_id`, `available_stock`, `recommended_order_quantity`, `expected_arrival_date`, `stockout_risk`, `disclaimer`, `human_review_required`.

---

## 11. Detailed Agent Specifications: Workflow 7 - Contract Review & Compliance (10 Agents)

- **Workflow Name:** `contract_review`
- **Assigned Credential:** `GROQ_API_KEY_7`
- **Input Data:** `data/contract_review/`

### Agent Breakdown:
1. **Contract Intake Agent (`contract_intake_agent`):**
   - **File:** [`Agents/contract_review/intake_agent.py`](file:///c:/Users/radhi/Desktop/Agentic_AI_Project/Agents/contract_review/intake_agent.py)
   - **Role:** Contract Intake and Ingestion Specialist
   - **Responsibilities:** Ingests raw contract text, identifies document type (MSA, NDA, DPA, SLA), catalogs governing law and parties, and validates completeness.
   - **Inputs:** `contract_id`, `title`, `contract_type`, `version`, `clauses`, `policy_id`.
   - **Outputs:** `document_id`, `contract_id`, `contract_type`, `document_complete`, `section_count`.

2. **Clause Extraction Agent (`clause_extraction_agent`):**
   - **File:** [`Agents/contract_review/clause_extraction_agent.py`](file:///c:/Users/radhi/Desktop/Agentic_AI_Project/Agents/contract_review/clause_extraction_agent.py)
   - **Role:** Contract Clause Extraction Specialist
   - **Responsibilities:** Parses text into discrete, stable clause units with IDs (`CLS-*`), section numbers, headings, and normalized excerpts.
   - **Inputs:** `clauses`, `contract_id`.
   - **Outputs:** `clauses`, `absent_clause_types`, `extraction_notes`.

3. **Policy Interpretation Agent (`policy_interpretation_agent`):**
   - **File:** [`Agents/contract_review/policy_interpretation_agent.py`](file:///c:/Users/radhi/Desktop/Agentic_AI_Project/Agents/contract_review/policy_interpretation_agent.py)
   - **Role:** Policy Interpretation Specialist
   - **Responsibilities:** Deconstructs enterprise compliance policies into discrete mandatory requirements (e.g. data breach notification $\le 72$ hours, unlimited confidentiality) and recommended thresholds.
   - **Inputs:** `policy_id`, `policy_name`, `requirements`.
   - **Outputs:** `policy_id`, `policy_version`, `mandatory_requirements`, `recommended_requirements`.

4. **Clause-to-Policy Mapping Agent (`clause_mapping_agent`):**
   - **File:** [`Agents/contract_review/clause_mapping_agent.py`](file:///c:/Users/radhi/Desktop/Agentic_AI_Project/Agents/contract_review/clause_mapping_agent.py)
   - **Role:** Clause-to-Policy Mapping Specialist
   - **Responsibilities:** Aligns extracted contract clauses to policy requirements, classifying alignment status (`fully_covered`, `partially_covered`, `missing`).
   - **Inputs:** `clauses`, `mandatory_requirements`, `recommended_requirements`.
   - **Outputs:** `mappings`, `coverage_percentage`.

5. **Compliance Deviation Agent (`compliance_deviation_agent`):**
   - **File:** [`Agents/contract_review/deviation_agent.py`](file:///c:/Users/radhi/Desktop/Agentic_AI_Project/Agents/contract_review/deviation_agent.py)
   - **Role:** Contract Compliance Deviation Evaluator
   - **Responsibilities:** Identifies non-compliant terms, missing mandatory protections, and unfavorable clauses. Assigns severity (`critical`, `high`, `medium`, `low`).
   - **Inputs:** `mappings`, `mandatory_requirements`.
   - **Outputs:** `deviations`, `has_critical_deviations`.

6. **Risk and Ambiguity Agent (`risk_ambiguity_agent`):**
   - **File:** [`Agents/contract_review/risk_ambiguity_agent.py`](file:///c:/Users/radhi/Desktop/Agentic_AI_Project/Agents/contract_review/risk_ambiguity_agent.py)
   - **Role:** Contract Risk and Ambiguity Analyst
   - **Responsibilities:** Scans contract language for vague standards ("reasonable efforts"), open-ended indemnity, unilateral amendment rights, and undefined timelines.
   - **Inputs:** `clauses`.
   - **Outputs:** `risk_findings`.

7. **Cross-Clause Consistency Agent (`cross_clause_consistency_agent`):**
   - **File:** [`Agents/contract_review/consistency_agent.py`](file:///c:/Users/radhi/Desktop/Agentic_AI_Project/Agents/contract_review/consistency_agent.py)
   - **Role:** Contract Inter-Clause Consistency Auditor
   - **Responsibilities:** Detects internal contractual contradictions (e.g., immediate termination for convenience vs. 3-year auto-renewal lock-in without opt-out).
   - **Inputs:** `clauses`, `effective_date`, `expiration_date`.
   - **Outputs:** `inconsistencies`.

8. **Evidence and Citation Validation Agent (`evidence_validation_agent`):**
   - **File:** [`Agents/contract_review/evidence_validation_agent.py`](file:///c:/Users/radhi/Desktop/Agentic_AI_Project/Agents/contract_review/evidence_validation_agent.py)
   - **Role:** Legal Evidence and Citation Validator
   - **Responsibilities:** Verifies that every finding, risk, and deviation references a genuine clause ID and that quoted text matches the extracted clause verbatim.
   - **Inputs:** `deviations`, `risk_findings`, `inconsistencies`, `clauses`, `requirements`.
   - **Outputs:** `all_citations_valid`, `verified_finding_ids`, `invalid_or_unsupported_findings`.

9. **Independent Review Agent (`contract_independent_review_agent`):**
   - **File:** [`Agents/contract_review/review_agent.py`](file:///c:/Users/radhi/Desktop/Agentic_AI_Project/Agents/contract_review/review_agent.py)
   - **Role:** Senior Contract Compliance Reviewer
   - **Responsibilities:** Deduplicates overlapping findings across agents, calibrates severity ratings, and isolates legal matters requiring human attorney review.
   - **Inputs:** `deviations`, `risk_findings`, `inconsistencies`, `all_citations_valid`.
   - **Outputs:** `review_status`, `duplicate_findings_merged`, `severity_adjustments`, `unresolved_matters`.

10. **Final Compliance Report Agent (`final_compliance_report_agent`):**
    - **File:** [`Agents/contract_review/final_report_agent.py`](file:///c:/Users/radhi/Desktop/Agentic_AI_Project/Agents/contract_review/final_report_agent.py)
    - **Role:** Lead Contract Compliance Report Author
    - **Responsibilities:** Synthesizes the master contract review report: executive summary, policy coverage score, clause inventory, findings matrix, and mandatory legal disclaimer.
    - **Inputs:** All upstream findings, deviations, risk assessments, and reviews.
    - **Outputs:** `run_id`, `contract_id`, `executive_summary`, `compliance_findings`, `risk_findings`, `legal_disclaimer`, `human_review_required`.

---

## 12. Common Infrastructure & Execution Architecture

### 12.1 [`common/settings.py`](file:///c:/Users/radhi/Desktop/Agentic_AI_Project/common/settings.py)
- **Mandatory Model Check:** Defines `MANDATORY_MODEL = "openai/gpt-oss-120b"`. Reads `GROQ_MODEL` from environment and raises `ConfigurationError` if any deviation or alternative model is specified.
- **Exception Class:** Defines `ConfigurationError(ValueError)` for unified configuration failure handling.
- **Helper Functions:** Exposes `get_groq_model()`, `get_groq_api_key()`, `get_phoenix_endpoint()`, `get_phoenix_project_name()`, and `validate_environment()`.

### 12.2 [`common/api_key_manager.py`](file:///c:/Users/radhi/Desktop/Agentic_AI_Project/common/api_key_manager.py)
- **Workflow Key Map:** Maps all seven workflows to their dedicated environment variables:
  ```python
  WORKFLOW_KEY_ENV_MAP = {
      "invoice_processing": ("GROQ_API_KEY_1", "GROQ_API_KEY_INVOICE_PROCESSING"),
      "customer_support": ("GROQ_API_KEY_2", "GROQ_API_KEY_CUSTOMER_SUPPORT"),
      "procurement_supplier_selection": ("GROQ_API_KEY_3", "GROQ_API_KEY_PROCUREMENT"),
      "sales_lead_qualification": ("GROQ_API_KEY_4", "GROQ_API_KEY_SALES_LEAD"),
      "recruitment_screening": ("GROQ_API_KEY_5", "GROQ_API_KEY_RECRUITMENT"),
      "inventory_replenishment": ("GROQ_API_KEY_6", "GROQ_API_KEY_INVENTORY"),
      "contract_review": ("GROQ_API_KEY_7", "GROQ_API_KEY_CONTRACT"),
  }
  ```
- **Agent Registry (`AGENT_WORKFLOW_MAP`):** Authoritative lookup mapping each of the 54 unique agent IDs to its parent workflow.
- **Safe Masking:** `mask_key(key)` ensures raw secrets are never leaked into logs, traces, or terminal outputs (e.g. `gsk_...BPdW`).
- **Validation Engine:** `validate_all_workflow_keys()` checks that all seven keys are configured and valid.

### 12.3 [`common/agent_runner.py`](file:///c:/Users/radhi/Desktop/Agentic_AI_Project/common/agent_runner.py)
- **Role-to-Agent Mapping (`ROLE_TO_AGENT_ID`):** Automatically maps agent roles to their designated unique `agent_id` so that factories instantiating agents by role receive their correct workflow credentials.
- **CrewAI Agent Factory:** `create_crew_agent(role, goal, backstory, ...)` creates a CrewAI agent bound to the mandatory `openai/gpt-oss-120b` Groq LLM instance.
- **Task Runner:** `run_agent_task(...)` executes a single agent task with structured Pydantic schema validation, automatic token usage extraction, and Arize Phoenix tracing instrumentation.
- **Unit Testing Mock Handler:** `set_mock_task_handler(handler)` allows unit tests to mock LLM task responses cleanly without touching the network.

### 12.4 [`common/tracing.py`](file:///c:/Users/radhi/Desktop/Agentic_AI_Project/common/tracing.py)
- **Arize Phoenix Integrator:** `init_phoenix(project_name, verbose=True)` initializes the local Phoenix dashboard (`http://localhost:6006`) and registers `CrewAIInstrumentor().instrument(skip_dep_check=True)`.
- **Token Extraction:** `extract_token_usage(crew_output, crew)` inspects CrewAI execution objects to capture accurate prompt and completion token counts.

### 12.5 [`common/metrics.py`](file:///c:/Users/radhi/Desktop/Agentic_AI_Project/common/metrics.py)
- **MetricsTracker:** Records agent execution steps, latencies, token counts, cost estimations, and validation warnings.
- **Persistence:** Serializes comprehensive run records into `runs/{workflow_name}_{run_id}.json` and appends high-level summaries into `runs/{workflow_name}/summary.jsonl`.

### 12.6 [`common/validation.py`](file:///c:/Users/radhi/Desktop/Agentic_AI_Project/common/validation.py)
- **Pydantic Schemas:** Strong typing for recruitment, inventory, and contract review outputs (`ResumeIntakeOutput`, `StockPolicyOutput`, `ContractFinalReportOutput`, etc.).
- **JSON Parsing:** Robust extractor `extract_json_from_text` handles raw JSON, markdown-fenced blocks (```json ... ```), and noisy LLM output.

---

## 13. Synthetic Datasets & Ground Truth Files

All datasets are stored locally under `data/`:
1. **`data/invoice_processing/`**: Invoices (including arithmetic discrepancy `INV-2026-0010`, duplicate `INV-2026-0020`, missing PO `INV-2026-0030`, restricted vendor `INV-2026-0040`), POs, and AP policies.
2. **`data/customer_support/`**: Customer tickets, SLA policies, and technical troubleshooting SOPs.
3. **`data/procurement_supplier_selection/`**: RFQ requests, vendor bids, delivery schedules, risk records, contracts, and evaluation weights.
4. **`data/sales_lead_qualification/`**: Inbound leads (high-fit `LEAD-2026-0001`, disqualified bakery `LEAD-2026-0002`), company profiles, ICP definitions, and qualification rubrics.
5. **`data/recruitment_screening/`**: Job descriptions, resumes with ground-truth evidence IDs (`EVD-*`), and criteria profiles.
6. **`data/inventory_replenishment/`**: SKU records, warehouse inventory, historical daily demand, and supplier lead-time constraints.
7. **`data/contract_review/`**: Full contracts (MSA, NDA, DPA, SLA) and enterprise compliance policies with explicit requirements.

---

## 14. Verification & Automated Testing Suite

The project includes **36 automated unit and integration tests** executing in **~6.5 seconds** with **zero API token consumption**:

```bash
pytest tests/ -v -k "not smoke"
```

### Test Coverage Breakdown:
- **`tests/test_structure_and_manifests.py`**:
  - Asserts exactly 54 unique agent IDs across all seven workflows.
  - Asserts exact agent distribution: 8, 6, 9, 7, 8, 6, 10.
  - Asserts that all seven `workflow.json` manifests exist, match required keys, and enforce `openai/gpt-oss-120b`.
- **`tests/test_dataset_generator.py`**:
  - Asserts reproducibility and zero-LLM data generation.
- **`tests/test_api_key_manager.py`**:
  - Asserts key isolation, safe masking, and error handling for missing keys.
- **`tests/test_deterministic_logic.py`**:
  - Validates deterministic math, 3-way matching, risk scoring, MCDA formulas, and ICP rubrics.
- **`tests/test_mocked_workflows.py`**:
  - Validates end-to-end execution of baseline workflows with mocked LLM calls.
- **`tests/test_recruitment_screening.py`**:
  - Validates 8 recruitment agents, deterministic match score formula, and mock pipeline run.
- **`tests/test_inventory_replenishment.py`**:
  - Validates usable stock formula, MOQ rounding, demand projection, and mock pipeline run.
- **`tests/test_contract_review.py`**:
  - Validates 10 contract agents, evidence linking, and mock pipeline run.
- **`tests/test_shared_infrastructure.py`**:
  - Validates model enforcement, JSON extraction, metrics recording, and Phoenix tracing initialization.

---

## 15. Agentic Workflow Optimizer (AWO) Mutation Opportunities

Each workflow's `workflow.json` manifest is engineered with explicit optimization entry points:

### 1. Agent Pruning Candidates
- **Invoice Processing:** Remove `exception_analysis_agent` when validation already establishes fatal rejections.
- **Customer Support:** Prune `issue_investigation_agent` on simple password reset or FAQ inquiries.
- **Sales Lead:** Prune `opportunity_assessment_agent` on leads scoring $<45$ points.
- **Recruitment Screening:** Prune `screening_review_agent` when all mandatory criteria are supported with zero inconsistencies.
- **Inventory Replenishment:** Prune `inventory_review_agent` on routine replenishments with zero stockout risk.
- **Contract Review:** Prune `evidence_validation_agent` or `review_agent` on standard bilateral NDAs without custom IP terms.

### 2. Agent Consolidation Candidates
- **Invoice Processing:** Merge `invoice_validation_agent` and `exception_analysis_agent`.
- **Procurement:** Merge `cost_comparison_agent` directly into `weighted_evaluation_agent`.
- **Recruitment Screening:** Consolidate `resume_intake_agent` and `qualification_agent`.
- **Inventory Replenishment:** Consolidate `stock_policy_agent` and `replenishment_planning_agent`.
- **Contract Review:** Consolidate `clause_mapping_agent` with `compliance_deviation_agent`, and `risk_ambiguity_agent` with `cross_clause_consistency_agent`.

### 3. Graph Parallelization Opportunities
- **Invoice Processing:** Parallelize `invoice_validation_agent`, `po_matching_agent`, `policy_compliance_agent`, and `duplicate_detection_agent`.
- **Procurement:** Execute quotation analysis, supplier profiling, delivery analysis, risk assessment, and contract analysis concurrently across 5 worker threads.
- **Sales Lead:** Run `lead_fit_agent` and `intent_analysis_agent` concurrently.
- **Recruitment Screening:** Execute `qualification_agent` and `job_criteria_agent` concurrently; parallelize `experience_assessment_agent` and `missing_evidence_agent`.
- **Inventory Replenishment:** Concurrently execute `stock_policy_agent`, `demand_forecast_agent`, and `supplier_agent`.
- **Contract Review:** Concurrently execute `compliance_deviation_agent`, `risk_ambiguity_agent`, and `cross_clause_consistency_agent`.

### 4. Dynamic Routing & Early Stopping
- **Customer Support:** Route simple tickets directly from Policy Retrieval to Response Drafting.
- **Sales Lead:** Early-stop and disqualify immediately after `lead_fit_agent` for prohibited industries.
- **Recruitment Screening:** Early-stop candidates who fail mandatory education or licensure thresholds before running downstream experience assessment.
- **Contract Review:** Standard NDA bypass to skip cross-clause deep auditing when no restrictive covenants exist.
