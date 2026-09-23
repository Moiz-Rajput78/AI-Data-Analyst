# AI Data Analyst Agent — Technical Report

## 1. Introduction

The AI Data Analyst Agent is an agentic data-analysis system developed to demonstrate how a Small Language Model can work together with structured analytical tools to answer business questions from a CSV dataset.

The project focuses on a fictional sales dataset containing order, product, category, region, pricing, revenue, cost, profit, and salesperson information.

The primary goal was not to build a fixed dashboard or a predefined analytics pipeline. Instead, the system was designed so that the AI model decides which analytical operation should be performed according to the user's question.

The project combines:

- Qwen Small Language Model integration
- Hugging Face Inference Providers
- dynamic tool calling
- Pandas-based analysis
- statistical functions
- restricted Python execution
- chart generation
- evidence validation
- automated testing
- report generation

The central architectural principle is that the language model acts as the analyst and orchestrator, while Python and Pandas perform numerical calculations.

---

## 2. Dataset

A realistic fictional sales dataset was created for this project.

The dataset contains 1,337 records and 13 columns.

The fields are:

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