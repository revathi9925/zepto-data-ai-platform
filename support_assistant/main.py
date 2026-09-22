from fastapi import FastAPI
from schemas import QueryRequest, QueryResponse
from graph import app_graph

app = FastAPI(title="Zepto Support Assistant")

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    initial_state = {
        "query": request.query,
        "intent": "",
        "context": [],
        "final_response": {}
    }
    result = app_graph.invoke(initial_state)
    return result["final_response"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)