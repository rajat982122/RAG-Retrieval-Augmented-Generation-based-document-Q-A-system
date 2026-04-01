import os
import shutil
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app import config
from app.db import init_db, log_query
from app.rag_pipeline import RAGPipeline


class QueryRequest(BaseModel):
    question: str


# Start important app pieces once when the API boots.
init_db()
rag_pipeline = RAGPipeline()

app = FastAPI(title="RAG Document Q&A System")


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Please choose a PDF file.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    os.makedirs(config.UPLOAD_DIR, exist_ok=True)

    # Replace spaces so file paths stay cleaner inside Docker and local runs.
    clean_file_name = file.filename.replace(" ", "_")
    file_path = os.path.join(config.UPLOAD_DIR, clean_file_name)

    with open(file_path, "wb") as output_file:
        shutil.copyfileobj(file.file, output_file)

    try:
        result = rag_pipeline.ingest_pdf(file_path)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    return {
        "message": "PDF uploaded and indexed successfully.",
        "file_name": result["file_name"],
        "pages_read": result["pages_read"],
        "chunks_added": result["chunks_added"],
        "total_chunks": result["total_chunks"],
    }


@app.post("/query")
def query_pdf(request: QueryRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    start_time = time.time()

    try:
        result = rag_pipeline.answer_question(question)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    response_time_ms = int((time.time() - start_time) * 1000)

    log_query(
        question=question,
        answer=result["answer"],
        response_time_ms=response_time_ms,
    )

    return {
        "question": question,
        "answer": result["answer"],
        "sources": result["sources"],
        "response_time_ms": response_time_ms,
    }
