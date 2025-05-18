import os
from typing import Dict, List, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader

# Load env
from dotenv import load_dotenv
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="Open Source LLM Retrieval App")

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Pinecone client
pine_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pine_key)
index_name = "llm-retrieval"

# Ensure index exists
if index_name not in pc.list_indexes().names():
    pc.create_index(name=index_name, dimension=384, metric="cosine",
                     spec=ServerlessSpec(cloud="aws", region="us-east-1"))
index = pc.Index(index_name)

# In-memory store
txts = {}

# Load and index PDF on startup
loader = PyPDFLoader("documents/The_IT_Job_market_situation_in_2025_report.pdf")
docs = loader.load_and_split()
vectors = []
for i, doc in enumerate(docs):
    text = doc.page_content.strip()
    if not text:
        continue
    emb = model.encode(text).tolist()
    vid = f"report_page_{i}"
    vectors.append((vid, emb))
    txts[vid] = text
# Bulk upsert
index.upsert(vectors=vectors)

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


# Custom validation error handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={
            "error": "Invalid request payload",
            "details": exc.errors(),
        }
    )


# Root endpoint
@app.get("/", include_in_schema=False)
def root():
    return {"message": "Welcome to the LLM Retrieval API. POST queries to /query."}


# Data schema
class Query(BaseModel):
    query: str
    top_k: int = 5

@app.post("/query", response_model=Dict[str, List[Dict[str, Any]]])
def run_query(q: Query):
    emb = model.encode(q.query).tolist()
    res = index.query(vector=emb, top_k=q.top_k)
    if not res.matches:
        raise HTTPException(status_code=404, detail="No matches found")
    docs = []
    for m in res.matches:
        docs.append({"id": m.id, "score": m.score, "text": txts.get(m.id, "")})
    return {"results": docs}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)