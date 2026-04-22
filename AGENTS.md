# Antigravity AGENTS.md

## Project Overview
This repository (`Init-DataOps`) is a DataOps infrastructure project managed dynamically via Antigravity, an advanced agentic AI coding assistant. The primary goal is to maintain, optimize, and orchestrate automated data pipelines, observability, FinOps, and incident response.

## Core Agent Behavior & Interaction
- **Progressive Disclosure:** Do not assume context. When interacting with this repository, always rely on the specialized skills stored in `.agents/skills/`. Read the `SKILL.md` of the relevant skill before executing complex tasks.
- **Modularity:** Treat agent skills as code. Maintain separation of concerns between deterministic tasks (e.g., executing Python scripts) and LLM-driven reasoning.

## Executable Commands (with RTK Compression)
When running terminal commands to analyze the workspace, you MUST prefix them with `rtk` to reduce token consumption by 60-90%. 
- **Environment:** Use the local workspace configuration.
- **Python/ETL Tasks:** Use `rtk python <script.py>` or `rtk pytest`.
- **DAG Testing:** Always run `rtk python <dag_file.py>` before finalizing orchestration changes.
- **Git & Logs:** Use `rtk git status`, `rtk git diff`, `rtk read <file>`, `rtk grep <pattern>`.

## Coding Style & Standards
- **SQL:** Favor CTEs (Common Table Expressions) over nested subqueries for readability. Optimize for cloud data warehouse execution (push filters down early).
- **Python/ETL:** Enforce strict type hinting, modular functions, and robust `try/except` fallback logic for network/API calls.
- **Idempotency:** All data tasks and orchestration DAGs must be idempotent.

## "Don't Touch" Boundaries
- **Credentials/Secrets:** NEVER commit or hardcode secrets. Always use environment variables or secret managers.
- **Production Data:** Never execute destructive commands (`DROP`, `DELETE`, `TRUNCATE`) without explicit human-in-the-loop approval.

## Active DataOps Skills
When asked to perform DataOps tasks, use the `view_file` tool to load instructions from the corresponding skill:
- **Orchestration (non-Airflow):** `.agents/skills/pipeline-orchestration/SKILL.md` — Dagster, Prefect, custom schedulers only
- **Quality & Observability:** `.agents/skills/data-observability/SKILL.md` — Great Expectations, Soda, schema drift (NOT dbt tests)
- **Query Writing:** `.agents/skills/sql-queries/SKILL.md` — all SQL dialects
- **Cost/FinOps:** `.agents/skills/cloud-finops/SKILL.md` — infrastructure-level costs (NOT query-level)
- **Incident Response:** `.agents/skills/rca-diagnostics/SKILL.md` — cross-system failures only (NOT single DAG/dbt errors)
- **Governance:** `.agents/skills/data-governance/SKILL.md` — RBAC, data contracts, compliance (NOT dbt documentation)

## dbt & Snowflake Skills (AltimateAI)
When working with dbt models or Snowflake queries, use the `view_file` tool to load instructions from these benchmark-proven skills:
- **Creating Models:** `.agents/skills/creating-dbt-models/SKILL.md`
- **Debugging Errors:** `.agents/skills/debugging-dbt-errors/SKILL.md`
- **Incremental Models:** `.agents/skills/developing-incremental-models/SKILL.md`
- **Documenting Models:** `.agents/skills/documenting-dbt-models/SKILL.md`
- **SQL-to-dbt Migration:** `.agents/skills/migrating-sql-to-dbt/SKILL.md`
- **Refactoring Models:** `.agents/skills/refactoring-dbt-models/SKILL.md`
- **Testing Models:** `.agents/skills/testing-dbt-models/SKILL.md`
- **Finding Expensive Queries:** `.agents/skills/finding-expensive-queries/SKILL.md`
- **Optimizing Query by ID:** `.agents/skills/optimizing-query-by-id/SKILL.md`
- **Optimizing Query Text:** `.agents/skills/optimizing-query-text/SKILL.md`

## Software Engineering & CI/CD Skills
When dealing with general software development, CI/CD, and code quality, use the `view_file` tool to load instructions from these skills:
- **Python Development:** `.agents/skills/python-expert/SKILL.md`
- **Package Management:** `.agents/skills/uv-package-manager/SKILL.md`
- **CI/CD Pipelines:** `.agents/skills/ci-cd-pipeline-builder/SKILL.md`
- **Tech Debt & Clean Code:** `.agents/skills/tech-debt-tracker/SKILL.md`, `.agents/skills/clean-code/SKILL.md`
- **Dependency Auditing:** `.agents/skills/dependency-auditor/SKILL.md`
- **Performance Profiling:** `.agents/skills/performance-profiler/SKILL.md`
- **Git & Worktrees:** `.agents/skills/git-workflow/SKILL.md`, `.agents/skills/worktree-manager/SKILL.md`

## Airflow & Astronomer Skills
When working with Apache Airflow or Astronomer Astro, use the `view_file` tool to load instructions:
- **Airflow CLI & Ops:** `.agents/skills/airflow/SKILL.md` — querying, managing, troubleshooting via `af` CLI
- **Writing DAGs:** `.agents/skills/authoring-dags/SKILL.md` — best practices for new DAG creation
- **Debugging DAGs:** `.agents/skills/debugging-dags/SKILL.md` — complex DAG failure diagnosis and RCA
- **Testing DAGs:** `.agents/skills/testing-dags/SKILL.md` — iterative test-debug-fix cycles
- **Deploying Airflow:** `.agents/skills/deploying-airflow/SKILL.md` — CI/CD and production deployment
- **Airflow 2→3 Migration:** `.agents/skills/migrating-airflow-2-to-3/SKILL.md` — upgrade guide
- **HITL Workflows:** `.agents/skills/airflow-hitl/SKILL.md` — human-in-the-loop approval/branching
- **Airflow Plugins:** `.agents/skills/airflow-plugins/SKILL.md` — FastAPI, React, custom UI in Airflow 3.1+
- **Blueprint/YAML DAGs:** `.agents/skills/blueprint/SKILL.md` — reusable task group templates from YAML
- **Astro Local Env:** `.agents/skills/managing-astro-local-env/SKILL.md` — start/stop/restart local Airflow
- **Astro Deployments:** `.agents/skills/managing-astro-deployments/skill.md` — production deployment management
- **Astro Troubleshooting:** `.agents/skills/troubleshooting-astro-deployments/skill.md` — production issue diagnosis
- **Astro Project Setup:** `.agents/skills/setting-up-astro-project/SKILL.md` — initialize new Astro projects

## Cosmos & dbt-Airflow Integration
- **Cosmos + dbt Core:** `.agents/skills/cosmos-dbt-core/SKILL.md` — run dbt Core projects as Airflow DAGs
- **Cosmos + dbt Fusion:** `.agents/skills/cosmos-dbt-fusion/SKILL.md` — run dbt Fusion on Snowflake/Databricks

## Data Lineage & Observability
- **Upstream Lineage:** `.agents/skills/tracing-upstream-lineage/SKILL.md` — where does data come from?
- **Downstream Lineage:** `.agents/skills/tracing-downstream-lineage/SKILL.md` — what depends on this data?
- **Task Lineage Annotation:** `.agents/skills/annotating-task-lineage/SKILL.md` — inlets/outlets metadata
- **OpenLineage Extractors:** `.agents/skills/creating-openlineage-extractors/SKILL.md` — custom extractors
- **Data Freshness Check:** `.agents/skills/checking-freshness/SKILL.md` — is data up to date?
- **Table Profiling:** `.agents/skills/profiling-tables/SKILL.md` — deep-dive data statistics
- **Warehouse Init:** `.agents/skills/warehouse-init/SKILL.md` — schema discovery setup

## Data Analysis & Querying
- **Analyzing Data:** `.agents/skills/analyzing-data/SKILL.md` — business questions via SQL
- **Statistical Analysis:** `.agents/skills/statistical-analysis/SKILL.md` — descriptive stats, hypothesis testing
- **Data Visualization:** `.agents/skills/data-visualization/SKILL.md` — matplotlib, seaborn, plotly charts
- **Data Storytelling:** `.agents/skills/data-storytelling/SKILL.md` — compelling narratives from data
- **Power BI Modeling:** `.agents/skills/powerbi-modeling/SKILL.md` — DAX, star schemas, RLS

## Financial Analysis Skills
- **Financial Statements:** `.agents/skills/analyzing-financial-statements/SKILL.md` — ratios and metrics
- **Financial Models:** `.agents/skills/creating-financial-models/SKILL.md` — DCF, Monte Carlo, scenarios
- **Variance Analysis:** `.agents/skills/variance-analysis/SKILL.md` — budget vs actual decomposition
- **Reconciliation:** `.agents/skills/reconciliation/SKILL.md` — GL-to-subledger, bank reconciliation

## Document Generation
- **Word Docs:** `.agents/skills/docx/SKILL.md` — create, read, edit .docx files
- **PowerPoint:** `.agents/skills/pptx/SKILL.md` — create, edit slide decks
- **PDF:** `.agents/skills/pdf/SKILL.md` — create, read, merge, split PDFs
- **Excel:** `.agents/skills/xlsx/SKILL.md` — create, read, edit spreadsheets
- **Excalidraw:** `.agents/skills/excalidraw-diagram-generator/SKILL.md` — visual diagrams
- **Canvas Design:** `.agents/skills/canvas-design/SKILL.md` — visual art and posters
- **PRD:** `.agents/skills/prd/SKILL.md` — Product Requirements Documents

## Research & Knowledge Skills
- **Deep Research:** `.agents/skills/deep-research/SKILL.md` — multi-source synthesis with citations
- **Agentic Eval:** `.agents/skills/agentic-eval/SKILL.md` — self-critique and quality loops

## Deprecated/Redirect Skills
These skills redirect to more specialized alternatives:
- **SQL Optimization:** `.agents/skills/sql-optimization/SKILL.md` — **DEPRECATED**, use `optimizing-query-text` or `optimizing-query-by-id`

## Background Knowledge Skills (non-invocable)
These provide context but are not directly triggered:
- `senior-data-engineer` — general data engineering architecture
- `knowledge-synthesis` — cross-source result merging (used by deep-research)
- `search-strategy` — query decomposition (used by deep-research)
- `sql-queries` — SQL dialect reference (used by analyzing-data)
- `data-visualization` — chart reference (used by data-storytelling)
- `statistical-analysis` — methods reference (used by analyzing-data)
