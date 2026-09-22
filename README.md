# Zepto Data & AI Platform Capstone

An integrated end-to-end AI/ML platform combining **catalog data engineering, predictive analytics, and a grounded GenAI support assistant** for Zepto's analytics guild.

---

## Project Architecture

This repository contains three core modules:

### 1. Data Pipeline

**Directory:** `/data_pipeline`

Scrapes catalog pricing and availability, cleans and normalizes the data into SQLite, and verifies SQL query results against Pandas.

### 2. Analytics Pipeline

**Directory:** `/analytics`

Profiles the Titanic dataset, performs EDA, trains predictive classification and fare regression models, and exports the complete preprocessing and modeling pipeline.

### 3. Support Assistant

**Directory:** `/support_assistant`

A RAG-based support assistant using FastAPI, ChromaDB, Sentence Transformers, and LangGraph intent routing, grounded in Zepto policy documents.

---

# 1. Data Pipeline

## Overview

The data pipeline extracts book catalog information from:

```text
books.toscrape.com
```

The scraper collects catalog listings across multiple categories and produces more than 60 product rows.

## How to Run

```bash
cd data_pipeline
pip install -r requirements.txt
python scraper.py
```

## Data Cleaning

The following transformations are applied:

* **Price:** Currency symbols such as `£` are removed and the value is converted to `float`.
* **Rating:** Text-based ratings such as `"Three"` are converted into integer values from `1` to `5`.
* **Stock:** Availability text is converted into Boolean values.
* **Price Conversion:** GBP prices are converted to INR using the fixed exchange rate:

```text
1 GBP = 105.50 INR
```

## Database Schema

The pipeline uses a normalized two-table SQLite database:

```text
categories
    |
    | category_id
    ↓
books
```

The `books` table contains `category_id` as a foreign key referencing the `categories` table.

## Verification

The SQL `JOIN` query output is verified against an equivalent Pandas `merge()` operation to ensure that both approaches produce equivalent results.

---

# 2. Analytics Pipeline

## Overview

The analytics module profiles the Titanic dataset, performs exploratory data analysis, handles missing values, trains predictive models, and exports the complete machine learning pipeline.

## How to Run

```bash
cd analytics
pip install -r requirements.txt
python titanic_ml.py
```

## Offline Dataset

The Titanic dataset is initially loaded using:

```python
sns.load_dataset("titanic")
```

The dataset is then saved locally as:

```text
analytics/titanic.csv
```

This allows the dataset to be loaded offline using:

```python
pd.read_csv("titanic.csv")
```

## Missing Value Strategy

Missing values are handled according to the percentage of missing data.

| Missing Percentage | Handling                             |
| ------------------ | ------------------------------------ |
| `< 5%`             | Drop affected rows                   |
| `5% – 30%`         | Impute using median/mode             |
| `> 30%`            | Encode missing values as `"Missing"` |

For example, the `deck` column is handled using a `"Missing"` category.

## Data Leakage Prevention

All preprocessing steps are fitted **only on the training data**.

The project uses:

* `SimpleImputer`
* `OneHotEncoder`
* `StandardScaler`
* `ColumnTransformer`
* Scikit-learn `Pipeline`

This prevents information from the test dataset from leaking into the training process.

## Imbalance Handling

The project evaluates:

1. Baseline models
2. Models using:

```python
class_weight="balanced"
```

3. Models using **SMOTE oversampling**

SMOTE is applied only to the training folds.

## Model Pipeline Export

The complete preprocessing and modeling pipeline is saved as:

```text
titanic_survival_pipeline.joblib
```

The exported pipeline is re-tested using raw, unprocessed input data to verify that the saved artifact works correctly.

---

# 3. Support Assistant

## Overview

The support assistant is an end-to-end **Retrieval-Augmented Generation (RAG)** system with a deterministic fallback.

By default:

```text
MOCK_LLM=True
```

## How to Run Locally

```bash
cd support_assistant
pip install -r requirements.txt
python main.py
```

Swagger API documentation is available at:

```text
http://localhost:7860/docs
```

## Running with Docker

Build the Docker image:

```bash
docker build -t zepto-support-assistant .
```

Run the container:

```bash
docker run -p 7860:7860 zepto-support-assistant
```

---

# RAG Architecture

## 1. Ingestion & Embedding

**File:** `ingestion.py`

The policy documents are:

```text
doc_01.txt
doc_02.txt
doc_03.txt
doc_04.txt
doc_05.txt
doc_06.txt
doc_07.txt
doc_08.txt
```

The documents are:

1. Loaded
2. Chunked
3. Converted into embeddings
4. Stored in a local ChromaDB collection

The embedding model used is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The local vector database is stored in:

```text
chroma_db/
```

---

## 2. Intent Classification

**Function:** `classify_intent`
**File:** `graph.py`

The assistant determines whether a query is related to a Zepto policy.

Policy-related keywords include:

```text
delivery
refund
return
cancel
tracking
```

Policy-related queries are routed to:

```text
policy_question
```

Other queries are routed to:

```text
general_question
```

---

## 3. Retrieval & Answer

**Function:** `retrieve_and_answer`
**File:** `graph.py`

For policy-related questions:

1. The query is converted into an embedding.
2. ChromaDB searches for relevant document chunks.
3. The most relevant chunks are retrieved.
4. The retrieved context is used to generate the answer.

---

## 4. Direct Answer

**Function:** `direct_answer`
**File:** `graph.py`

General questions do not require vector search.

Instead, the assistant returns a predefined response.

---

# LLM Modes

The project supports two LLM modes.

## MOCK_LLM Mode

```text
MOCK_LLM=True
```

In this mode, the application:

* Does not require a network connection.
* Returns deterministic responses.
* Includes retrieved context.
* Includes document source IDs.

This mode makes the application easier to test and demonstrate.

## Real LLM Mode

```text
MOCK_LLM=False
```

In this mode:

1. Retrieved context is sent to an LLM provider.
2. The response is validated using a structured Pydantic schema.
3. Formatting failures can trigger corrective retries.
4. Up to two corrective retries are supported.

---

# Example API Execution

## Policy Retrieval Question

### Endpoint

```text
POST /ask
```

### Request

```json
{
  "query": "What is the delivery policy for orders below INR 149?"
}
```

### Response

```json
{
  "answer": "Based on the retrieved context: Standard delivery is free on orders above INR 149. Orders below this threshold incur a flat INR 25 delivery fee.",
  "sources": ["doc_01"],
  "confidence": 1.0
}
```

---

## General Question

### Endpoint

```text
POST /ask
```

### Request

```json
{
  "query": "Hello, how are you today?"
}
```

### Response

```json
{
  "answer": "I am Zepto's GenAI Support Assistant. How can I assist you with our policies today?",
  "sources": [],
  "confidence": 1.0
}
```

---

# Project Structure

```text
zepto-data-ai-platform/
│
├── README.md
│
├── data_pipeline/
│   ├── scraper.py
│   ├── requirements.txt
│   └── ...
│
├── analytics/
│   ├── titanic_ml.py
│   ├── titanic.csv
│   ├── requirements.txt
│   └── ...
│
└── support_assistant/
    ├── main.py
    ├── graph.py
    ├── ingestion.py
    ├── mock_llm.py
    ├── real_llm.py
    ├── requirements.txt
    ├── Dockerfile
    ├── chroma_db/
    └── ...
```

---

# Git Workflow & Integrity

The project was developed using a feature-branch workflow:

```text
feature/setup
      ↓
     main
```

The repository contains multiple commits showing incremental development and progress.

All custom code, data analysis, database schema designs, and written interpretations were developed specifically for this capstone project.
