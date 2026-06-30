# AI-Powered Product Analytics Pipeline

A machine learning pipeline that analyzes e-commerce product listing data using multiple modeling approaches, then uses the OpenAI API to automatically translate technical model outputs into plain-language business insights.

## Overview

This project analyzes a dataset of 1,000+ Amazon India product listings to uncover patterns in pricing, category performance, and product characteristics. Rather than relying on a single model, the pipeline runs three complementary analyses and synthesizes the findings — then uses GPT to generate readable summaries of what the numbers actually mean for a business audience.

## What It Does

- **Random Forest Classification** — predicts product category/performance tier based on listing features, with feature importance analysis to identify the strongest signals
- **Single-Feature Model Comparison** — benchmarks individual features against each other to evaluate predictive power in isolation, helping identify which variables matter most
- **K-Means Clustering** — segments products into natural groupings based on shared characteristics, surfacing patterns not visible from category labels alone
- **GPT-Generated Summaries** — feeds model outputs into the OpenAI API to auto-generate business-readable explanations of each analysis, then synthesizes all three into a single executive-level conclusion

## Tech Stack

- **Python** — pandas, NumPy, scikit-learn (Random Forest, K-Means)
- **OpenAI API** — GPT-based natural language summarization
- **Matplotlib** — visualization of clusters and feature importance

## Key Learnings

- Designed a multi-model pipeline rather than relying on one technique, comparing classification, isolated feature analysis, and unsupervised clustering to triangulate findings from different angles
- Integrated a live LLM API into an analytical workflow, including handling real-world constraints like API rate limits and quota management
- Practiced sequencing a notebook so outputs from earlier cells (model results) correctly feed into later cells (GPT summarization), debugging execution-order issues along the way

## How to Run

1. Clone the repo and install dependencies: `pip install pandas numpy scikit-learn matplotlib openai`
2. Add your OpenAI API key as an environment variable: `export OPENAI_API_KEY=your_key_here`
3. Run the notebook/script in order — data loading → modeling → GPT summarization

## Project Context

Built as a final project for a deep learning / business analytics course (BAIS 409), University of Mississippi.
