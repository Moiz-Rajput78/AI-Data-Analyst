# AI Data Analyst Agent — Technical Report

## 1. Introduction

The AI Data Analyst Agent is an agentic data-analysis system designed to answer business questions from a structured CSV dataset using a language model, analytical tools, Python, Pandas, and visualization.

The main objective is not to build a fixed dashboard or predefined analytics pipeline. Instead, the language model acts as an analyst and decides what information is needed, which tools should be called, whether additional analysis is necessary, and when enough evidence has been collected to produce a final answer.

The project demonstrates language-model integration, structured tool calling, dynamic agent loops, CSV analysis, Pandas-based numerical computation, controlled Python execution, chart generation, grounded reasoning, output validation, automated testing, report generation, and Streamlit deployment.

The central design principle is:

```text
Language Model = Analyst / Orchestrator
Python + Pandas = Calculator
Tools = Controlled analytical capabilities
Agent Loop = Dynamic decision-making mechanism
```

## 2. Dataset Design

The project uses `data/sales.csv`, a fictional sales dataset containing 1,337 rows and 13 columns: order_id, order_date, customer_id, product, category, region, quantity, unit_price, discount, revenue, cost, profit, and salesperson.

The dataset contains intentionally designed patterns, including a final-month revenue decline, category-level differences, regional differences, product-level variation, unusually large quantity orders, and missing discount values.

Verified results include:

```text
August 2026 revenue: $157,493.06
September 2026 revenue: $133,662.60
Month-over-month change: -15.13%
```

Electronics shows the strongest category-level decline. The dataset can be regenerated through `data/generate_sales.py`.

## 3. System Architecture

```text
User
  ↓
CLI / Streamlit
  ↓
Agent State + Context
  ↓
Qwen Model
  ↓
Decide Next Action
  ↓
Analytical Tools
  ↓
Tool Results
  ↓
Qwen
  ↓
More Analysis or Final Report
```

The system does not use one universal sequence for every question. A simple ranking question may require only one grouped aggregation, while a complex change-analysis question may require statistics, comparisons, custom Pandas analysis, and visualization.

## 4. Agent Design

The main agent loop is implemented in `agent/agent.py`. Conversation state is managed in `agent/state.py`, prompt behavior in `agent/prompts.py`, and model access in `agent/llm.py`.

Conceptually:

```python
while analysis_is_not_complete:
    response = ask_model(conversation, available_tools)

    if response requests a tool:
        result = execute_tool(tool_name, arguments)
        add_result_to_context(result)
    else:
        validate_final_answer()
        return final_answer
```

A tool-call budget limits excessive or repetitive actions.

## 5. Tool Design

### Dataset Inspection
`inspect_dataset()` returns dataset structure, data types, missing values, duplicates, date ranges, numeric summaries, and sample records.

### Statistical Analysis
`calculate_statistics()` supports mean, median, sum, min, max, standard deviation, count, and percentage change, including month/year/quarter grouping.

### Grouped Aggregation
`group_and_aggregate()` performs grouped comparisons such as revenue by category, revenue by region, revenue by product, and profit by salesperson.

### Filtering
`filter_dataset()` allows focused row-level analysis using comparison and text operators.

### Restricted Python
`run_python()` provides custom Pandas analysis with `df` and `pd` already available. Imports, unrestricted filesystem access, OS access, process execution, file writing, loops, and dangerous operations are blocked.

### Visualization
`create_chart()` generates line, bar, pie, and scatter charts using Matplotlib. Charts are stored in `output/charts/`.

## 6. Prompt Design

The prompt instructs Qwen to act as the analyst and orchestrator, use tools for numerical evidence, dynamically select tools, avoid unsupported business explanations, derive time periods from the dataset, calculate percentage changes through tools, use vectorized Pandas, avoid unnecessary repeated calls, and only mention charts that were actually generated.

The model is explicitly prevented from treating example values as evidence. The current dataset and tool results are the source of truth.

## 7. Model Selection

The current implementation uses `Qwen/Qwen3-32B` through Hugging Face Inference Providers. The model is configured with `HF_TOKEN`, `HF_MODEL`, and `HF_PROVIDER`.

It was selected because it can follow structured instructions, produce tool calls, reason over tool outputs, maintain multi-step context, and decide whether more analysis is needed.

The original task recommended a local Ollama SLM. The project currently uses Qwen through Hugging Face Inference Providers as the approved implementation approach.

## 8. Data-Analysis Strategy

The analysis strategy depends on the question. A simple question such as “Which category generated the most revenue?” may only require grouped aggregation.

A deeper question such as “Why did sales decrease last month?” can require:

1. determining the latest and previous periods;
2. calculating month-over-month change;
3. comparing categories;
4. comparing regions;
5. investigating products;
6. generating a chart when useful;
7. producing a grounded final report.

## 9. Evidence Validation

The project validates unsupported percentage claims and fake chart references. If Qwen reports a percentage that is not present in analytical tool evidence, the answer can be rejected. If it mentions a chart that was not generated by `create_chart()`, the answer can also be rejected.

These mechanisms reduce hallucinated analytical claims.

## 10. Problems Encountered

During development, several issues were encountered:

- an earlier Qwen model was not supported by the configured inference provider;
- Qwen sometimes returned empty visible responses;
- generated Python sometimes attempted imports or restricted features;
- chart calls used invalid columns such as `value`;
- chart filtering was incorrectly attempted using `period="month"`;
- Hugging Face inference was occasionally blocked by service errors or exhausted credits.

The implementation was improved through model changes, prompt updates, restricted-executor guidance, chart-tool corrections, local tests, and a local required-query evaluation framework.

## 11. Testing and Evaluation

The project currently has:

```text
76 passing automated tests
5/5 required-query evaluation checks passing
```

Tests cover dataset loading, statistics, aggregation, filtering, restricted Python, charts, report generation, agent loops, tool-result feedback, evidence validation, tool-call budgets, required queries, and evaluation reporting.

Run:

```bash
python -m pytest -q
python -m evaluation.required_queries
```

## 12. Required Questions

```text
1. Why did sales decrease last month?
2. Which category generated the most revenue?
3. Which region has the highest profit margin?
4. Are there any unusual sales patterns?
5. Give me a management summary of the dataset.
```

## 13. Reporting

`tools/reports.py` generates Markdown analysis reports under `output/reports/`. Reports can include the question, final answer, dataset information, tools used, and tool status.

## 14. User Interfaces

CLI:

```bash
python main.py
```

Streamlit:

```bash
python -m streamlit run app.py
```

Live deployment: https://rexyai-data-analyst.streamlit.app/

Both interfaces use the same underlying agent and analytical tools.

## 15. Limitations

- External business causes cannot be proven unless relevant variables exist in the dataset.
- The restricted Python executor intentionally limits general Python features and system access.
- Live model responses depend on Hugging Face provider availability and quota.
- The project focuses on the provided fictional sales dataset.
- The executor is a controlled project sandbox, not a full production security boundary.

## 16. Future Improvements

Possible improvements include arbitrary CSV uploads, automatic profiling, persistent conversation memory, stronger anomaly detection, richer chart selection, PDF reports, provider fallback, local Ollama support, human approval for costly actions, multi-agent design, and persistent analysis history.

## 17. Conclusion

The project demonstrates an agentic analytical architecture in which Qwen acts as the analyst and orchestrator, while Python and Pandas perform numerical computation.

The system dynamically selects tools, observes their results, decides whether more evidence is needed, validates important claims, and produces grounded reports.

This demonstrates the key assignment concepts of language-model integration, tool calling, agent loops, data analysis, visualization, reasoning, and evidence grounding.
