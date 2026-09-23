AI Data Analyst Agent
An agentic AI data-analysis system that analyzes a CSV dataset using Qwen, Python, Pandas, and dynamic tool calling.
The system allows a user to ask analytical business questions in natural language. Instead of following a fixed workflow, the AI model decides which tools are required, executes them, inspects their results, performs additional analysis when necessary, and produces an evidence-based report.
Live Demo
Open AI Data Analyst Agent
GitHub Repository
AI-Data-Analyst
---
Project Objective
The objective of this project is to build an AI Data Analyst Agent capable of analyzing a realistic sales CSV dataset.
The main concepts demonstrated are:
Small Language Model / open-model integration
Tool calling
Dynamic agent loop
Data analysis
Statistical computation
Restricted Python execution
Data visualization
Grounded reasoning
Evidence validation
Automatic report generation
The main architectural principle is:
```text
Qwen = Analyst / Orchestrator
Python + Pandas = Calculator
Tools = Controlled analytical capabilities
Agent Loop = Decision-making process
```
The model decides what analysis should be performed, while Python and Pandas calculate the actual numerical results.
---
What Is an SLM?
An SLM, or Small Language Model, is a language model designed to perform tasks such as instruction following, reasoning, structured output, and tool selection while generally requiring fewer computational resources than very large language models.
In this project, the language model is used as the decision-making component of the AI agent. It does not directly perform large numerical calculations. Instead, it decides which analytical tools should be called and interprets the results returned by Python and Pandas.
---
Model Selection
The project currently uses `Qwen/Qwen3-32B` through Hugging Face Inference Providers.
The selected model is configured through environment variables:
```env
HF_TOKEN=your_hugging_face_token
HF_MODEL=Qwen/Qwen3-32B
HF_PROVIDER=auto
```
The model was selected because it supports instruction following, structured tool calling, reasoning over tool results, multi-step conversations, and decision-making inside an agent loop.
The model integration is isolated inside `agent/llm.py`, which allows the provider to be changed later without rewriting the analytical tool layer.
> Note: the original assignment suggested a local Ollama-based SLM. The current implementation uses Qwen through Hugging Face Inference Providers as the approved project implementation approach.
---
What Makes This Application Agentic?
This project is not a fixed dashboard and not a normal chatbot.
A normal chatbot typically works like this:
```text
Question
   ↓
LLM
   ↓
Answer
```
A traditional data application usually follows a predefined pipeline:
```text
Input
   ↓
Fixed Python Functions
   ↓
Charts / Dashboard
```
This project instead follows an agentic loop:
```text
User Question
      ↓
Qwen
      ↓
"What analysis do I need?"
      ↓
Select Tool
      ↓
Execute Tool
      ↓
Observe Result
      ↓
Qwen
      ↓
"Do I need more evidence?"
   ↙             ↘
 Yes              No
  ↓                ↓
More Tools     Final Report
```
Different questions can therefore result in different tool sequences. This dynamic tool selection is the central feature of the project.
---
Dataset
The project uses a realistic fictional sales dataset at `data/sales.csv` containing 1,337 rows and 13 columns:
```text
order_id
order_date
customer_id
product
category
region
quantity
unit_price
discount
revenue
cost
profit
salesperson
```
The dataset was intentionally designed with meaningful analytical patterns including month-over-month revenue changes, category-level differences, regional differences, product-level differences, unusually large quantity orders, and missing discount values.
The dataset generator is available at `data/generate_sales.py`.
---
Important Discovered Pattern
Verified analytical results include:
```text
August 2026 Revenue: $157,493.06
September 2026 Revenue: $133,662.60
Month-over-Month Change: -15.13%
```
The category with the strongest decline is Electronics. These findings are calculated from the dataset using analytical tools rather than being hard-coded into the agent's final answer.
---
Project Architecture
```text
User
  ↓
CLI / Streamlit UI
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
---
Available Analytical Tools
`inspect_dataset()`
Returns dataset structure, rows, columns, data types, missing values, duplicates, date ranges, numeric summaries, and sample rows. Implemented in `tools/dataset.py`.
`calculate_statistics()`
Supports mean, median, sum, min, max, standard deviation, count, and percentage change with month/year/quarter grouping. Implemented in `tools/statistics.py`.
`group_and_aggregate()`
Groups by dimensions such as category, region, product, or salesperson and calculates aggregations such as sum, mean, median, min, max, and count. Implemented in `tools/aggregation.py`.
`filter_dataset()`
Supports focused filtering using operators such as equals, not_equals, greater_than, greater_than_or_equal, less_than, less_than_or_equal, and contains. Implemented in `tools/filters.py`.
`run_python()`
Provides restricted custom Pandas analysis with `df` and `pd` available. Imports, unrestricted filesystem access, OS access, file writing, loops, and dangerous operations are blocked. Implemented in `tools/python_executor.py`.
`create_chart()`
Generates line, bar, pie, and scatter charts using actual dataset columns. Generated charts are saved in `output/charts/`. Implemented in `tools/charts.py`.
---
Agent Loop
The core agent logic is implemented in `agent/agent.py`.
Conceptually:
```python
while not finished:
    response = ask_model(
        user_question,
        conversation_context,
        available_tools,
        previous_results,
    )

    if response requests a tool:
        result = execute_tool(response.tool, response.arguments)
        add_tool_result_to_context(result)
    else:
        return response.final_answer
```
The model chooses the next action dynamically. There is no single fixed sequence that runs for every question.
---
Tool-Calling Format
A model-selected tool request can conceptually look like:
```json
{
  "name": "group_and_aggregate",
  "arguments": {
    "group_by": "category",
    "metric": "revenue",
    "aggregation": "sum"
  }
}
```
The agent validates the tool call, executes the corresponding Python function, stores the result, sends the result back to Qwen, and lets Qwen decide the next action.
---
Grounded Reasoning
The project distinguishes between observed facts and unsupported explanations. For example, the dataset may show that Electronics revenue declined, but the model should not claim that competition, customer preferences, seasonality, or marketing caused the decline unless the dataset directly supports those claims.
The agent identifies where a change occurred and clearly states when the dataset cannot establish the external cause.
---
Evidence Validation
The agent includes reliability checks for unsupported percentages and fake chart references. If a percentage is not present in analytical tool evidence, or if a referenced chart was never generated, the proposed answer can be rejected and corrected.
---
Required User Questions
```text
1. Why did sales decrease last month?
2. Which category generated the most revenue?
3. Which region has the highest profit margin?
4. Are there any unusual sales patterns?
5. Give me a management summary of the dataset.
```
---
Required Query Evaluation
Run:
```bash
python -m evaluation.required_queries
```
Current verified result:
```text
1. Why did sales decrease last month? PASS
2. Which category generated the most revenue? PASS
3. Which region has the highest profit margin? PASS
4. Are there any unusual sales patterns? PASS
5. Give me a management summary of the dataset. PASS

Overall: 5/5 PASS
```
---
Automated Testing
Run:
```bash
python -m pytest -q
```
Current verified result:
```text
76 passed
```
The tests cover dataset handling, statistics, aggregation, filtering, restricted Python execution, charts, agent-loop behavior, evidence validation, report generation, required questions, and the evaluation framework.
---
Automatic Report Generation
Successful analyses can be saved as Markdown reports under `output/reports/`. Reports can contain the user question, final answer, dataset information, tools used, and tool execution status. Implemented in `tools/reports.py`.
---
CLI Application
Run:
```bash
python main.py
```
---
Streamlit Application
Run locally:
```bash
python -m streamlit run app.py
```
Live deployment:
https://rexyai-data-analyst.streamlit.app/
---
Installation
```bash
git clone https://github.com/Moiz-Rajput78/AI-Data-Analyst.git
cd AI-Data-Analyst
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
```
---
Environment Configuration
Create `.env` using:
```env
HF_TOKEN=your_hugging_face_token_here
HF_MODEL=Qwen/Qwen3-32B
HF_PROVIDER=auto
```
Do not commit the real token.
---
Usage
```bash
python main.py
python -m streamlit run app.py
python -m pytest -q
python -m evaluation.required_queries
python data/generate_sales.py
```
---
Example Queries
```text
Why did sales decrease last month?
Which category generated the most revenue?
Which region has the highest profit margin?
Are there any unusual sales patterns?
Give me a management summary of the dataset.
Which salesperson generated the most revenue?
Compare revenue by region.
What is the monthly revenue trend?
Which products have unusually high order quantities?
```
---
Example Findings
```text
Top Revenue Category: Electronics
Electronics Revenue: $796,047.76
August 2026 Revenue: $157,493.06
September 2026 Revenue: $133,662.60
Month-over-Month Change: -15.13%
Missing discount values: 12
Unusually large quantity orders: 4
```
---
Deployment
The project is deployed using Streamlit Community Cloud. Deployment secrets are configured through Streamlit's secret settings rather than being committed to GitHub.
---
Limitations
The dataset can identify where changes occurred but cannot always establish external business causes.
The restricted Python environment is intentionally not a general-purpose Python runtime.
Live AI responses depend on Hugging Face Inference Provider availability and quota.
The current implementation focuses on one fictional sales dataset.
---
Future Improvements
arbitrary CSV uploads
stronger anomaly detection
automatic dataset profiling
persistent conversation memory
richer chart selection
PDF report export
provider fallback
local Ollama support
human approval before expensive operations
multi-agent architecture
persistent analysis history
---
Key Learning Outcome
```text
Qwen = Analyst / Orchestrator
Python + Pandas = Numerical Computation
Tools = Controlled Actions
Agent Loop = Dynamic Decision-Making
Evidence Validation = Reliability Layer
```
The agent decides what analysis should happen. Python and Pandas calculate the evidence. Qwen examines the evidence and decides whether more analysis is required. Only after sufficient evidence exists does the agent produce the final report.