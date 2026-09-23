AI Data Analyst Agent — 5–7 Minute Demo Script
Demo Goal
Show that the project is an agentic AI Data Analyst, not a fixed dashboard.
The required flow is:
```text
CSV loaded
→ User asks a question
→ Qwen selects tools
→ Tools execute
→ Qwen examines results
→ Additional tools are selected when needed
→ Chart is generated
→ Final evidence-based report is produced
```
Before the Demo
```bash
source .venv/Scripts/activate
python -m pytest -q
python -m evaluation.required_queries
```
Expected results:
```text
76 passed
Overall: 5/5 PASS
```
Live deployment: https://rexyai-data-analyst.streamlit.app/
Part 1 — Introduction (30–40 seconds)
Explain that the project is built using Qwen, Python, Pandas, and Streamlit. Emphasize that the model dynamically decides which analytical tools to use and that Python/Pandas perform the actual calculations.
Part 2 — Dataset (20–30 seconds)
Show `data/sales.csv`. Explain that it contains 1,337 fictional sales records and 13 columns, including order date, product, category, region, quantity, revenue, cost, profit, and salesperson.
Part 3 — Architecture (40–50 seconds)
Open `agent/agent.py` and explain:
```text
User
  ↓
Agent
  ↓
Qwen
  ↓
Tool Selection
  ↓
Python / Pandas Tool
  ↓
Tool Result
  ↓
Qwen
  ↓
More Analysis or Final Answer
```
Point out the tool registry, tool schemas, and `ask()` method. Explain that the workflow is dynamic rather than hard-coded.
Part 4 — Available Tools (35–45 seconds)
Show:
```text
inspect_dataset()
calculate_statistics()
group_and_aggregate()
filter_dataset()
run_python()
create_chart()
```
Mention that `run_python()` uses restricted Pandas execution and does not provide unrestricted OS or filesystem access.
Part 5 — Simple Analytical Question (40–50 seconds)
Run:
```bash
python main.py
```
Ask:
```text
Which category generated the most revenue?
```
Expected result:
```text
Electronics
$796,047.76
```
Explain that a simple question should require only a small number of useful tool calls.
Part 6 — Complex Investigation (1–2 minutes)
Ask:
```text
Why did sales decrease last month?
```
Explain that a deeper question may require:
```text
monthly change
→ category comparison
→ region comparison
→ product investigation
→ visualization
→ grounded report
```
Verified evidence includes:
```text
August 2026 revenue: $157,493.06
September 2026 revenue: $133,662.60
Month-over-month change: -15.13%
```
Explain that the decline is concentrated strongly in Electronics, but the dataset cannot prove external causes such as competition or seasonality.
Part 7 — Visualization (30–40 seconds)
Show a generated chart from `output/charts/` or from the Streamlit interface. Explain that line, bar, pie, and scatter charts are supported.
Part 8 — Reliability (30–40 seconds)
Explain that unsupported percentage claims can be rejected, fake chart references can be rejected, the agent has a tool-call budget, and failed tool calls can be corrected and retried.
Part 9 — Required Query Evaluation (30 seconds)
Run:
```bash
python -m evaluation.required_queries
```
Show the five required questions passing and the overall `5/5 PASS` result.
Part 10 — Automated Tests (20 seconds)
Run:
```bash
python -m pytest -q
```
Show:
```text
76 passed
```
Part 11 — Streamlit Deployment (30 seconds)
Open the live app and show the dataset status, question input, Analyze button, final response, tool history, and chart area.
Part 12 — Reports (20 seconds)
Show `output/reports/` and explain that successful analyses can be stored as Markdown reports containing the question, answer, dataset, and tool history.
Part 13 — Conclusion (25–30 seconds)
Explain that Qwen acts as the analyst and orchestrator, while Python and Pandas act as the calculator. The agent dynamically selects tools, evaluates their results, decides whether more evidence is needed, and produces a grounded analytical report.
Backup Demo Plan
If Hugging Face is unavailable because of provider downtime or quota limits, demonstrate:
```bash
python -m pytest -q
python -m evaluation.required_queries
```
Also show generated charts and reports. Explain clearly that a provider outage affects live inference, while the local analytical tools and test suite remain functional.