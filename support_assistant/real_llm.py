import os
import json
from schemas import QueryResponse

def call_real_llm(prompt: str) -> dict:
    from groq import Groq
    api_key = os.environ.get("GROQ_API_KEY", "")
    client = Groq(api_key=api_key)
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def real_llm_with_retry(prompt: str, max_retries: int = 2) -> dict:
    current_prompt = prompt
    for attempt in range(max_retries + 1):
        try:
            raw_response = call_real_llm(current_prompt)
            validated = QueryResponse(**raw_response)
            return validated.model_dump()
        except Exception as e:
            if attempt == max_retries:
                return {
                    "answer": "Failed to format response after retries.",
                    "sources": [],
                    "confidence": 0.0
                }
            current_prompt += f"\n\n[CORRECTION NOTICE]: Previous response failed validation with error: {str(e)}. Correct your format to match JSON output strictly."