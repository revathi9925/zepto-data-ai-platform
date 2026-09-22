# Module 3 — Support Assistant (`/support_assistant`)

## RAG Pipeline Architecture Description

1. **Ingestion & Embedding Stage**:
   - **File / Function**: `ingestion.py` :: `init_vector_store()`
   - **Details**: Reads all 8 corpus text files (`docs/doc_01.txt` ... `docs/doc_08.txt`), treating each file as a single chunk. Embeds each document using the `all-MiniLM-L6-v2` model from `sentence-transformers` and indexes the vectors in a persistent ChromaDB collection (`zepto_policies`).

2. **Intent Classification Stage**:
   - **File / Function**: `graph.py` :: `classify_intent_node()`
   - **Details**: Receives input query and determines whether retrieval is needed (`policy_question` vs `general_question`).
   - **`MOCK_LLM` Branch Point**:
     - *Mock (`MOCK_LLM` unset/1)*: Calls `mock_llm.py` :: `mock_classify_intent()`, which uses a keyword heuristic (`delivery`, `return`, `refund`, etc.) with zero LLM/network calls.
     - *Real (`MOCK_LLM=0)*: Calls `real_llm.py` :: `call_real_llm()` to classify using Groq.

3. **Retrieval Stage**:
   - **File / Function**: `ingestion.py` :: `query_vector_store()`, invoked by `graph.py` :: `retrieve_and_answer_node()`.
   - **Details**: Embeds the query and performs cosine similarity search in ChromaDB to fetch the top-3 matching document chunks. **This stage executes for real in both Mock and Real LLM modes.**

4. **Generation Stage**:
   - **File / Function**: `graph.py` :: `retrieve_and_answer_node()` / `direct_answer_node()`
   - **`MOCK_LLM` Branch Point**:
     - *Mock (`MOCK_LLM` unset/1)*: Calls `mock_llm.py` :: `mock_retrieve_and_answer()`, returning a deterministic canned string built from the top chunk snippet alongside all 3 retrieved source IDs (`sources = ["doc_01.txt", ...]`). Non-policy queries trigger `mock_llm.py` :: `mock_direct_answer()`.
     - *Real (`MOCK_LLM=0)*: Calls `real_llm.py` :: `real_llm_with_retry()` with the structured prompt template from `prompt_templates.py`. Validates against `QueryResponse` Pydantic schema in `schemas.py` and retries up to 2 times on validation failure.

---

## Example Test Call Outputs (`MOCK_LLM=1` Baseline)

### 1. Policy Query (Triggering Retrieval)
**Request:** `POST http://localhost:7860/ask`
```json
{
  "query": "What is Zepto's delivery policy?"
}