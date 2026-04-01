# RAG Based Document Q&A System

I built this project to learn RAG (Retrieval-Augmented Generation) in a practical way. The idea is simple: upload a PDF, convert the PDF into chunks, store the chunk embeddings in FAISS, retrieve the most relevant chunks for a question, and then use an LLM to answer from that context with citations.

This is a personal project I made as a final-year B.Tech CSE student, so I kept the code beginner-friendly on purpose. I tried to make the flow easy to read instead of making it super production-level.

## What this project does

- Upload a PDF file using the API or Streamlit UI
- Extract text from the PDF using PyMuPDF
- Split text into chunks of around 500 tokens with 50 token overlap
- Generate embeddings locally using `all-MiniLM-L6-v2`
- Store embeddings inside a local FAISS index on disk
- Retrieve top matching chunks for a user question
- Send retrieved chunks to an LLM for answer generation
- Show citations using chunk ids like `[chunk_4]`
- Log question, answer, and response time into MySQL

## Tech stack

- Python
- FastAPI
- Streamlit
- LangChain
- FAISS
- sentence-transformers
- MySQL
- Docker and Docker Compose
- OpenAI API for answer generation

## Folder structure

```text
project/
├── app/
│   ├── main.py
│   ├── rag_pipeline.py
│   ├── db.py
│   └── config.py
├── streamlit_app.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## How the flow works

1. User uploads a PDF.
2. PDF text is extracted page by page.
3. Text is split into smaller chunks.
4. Each chunk is converted into embeddings locally.
5. Embeddings are stored in FAISS and saved to disk.
6. User asks a question in plain English.
7. Top relevant chunks are retrieved.
8. Retrieved chunks are passed to the LLM as context.
9. Final answer is returned with chunk citations.
10. The query and answer are logged into MySQL.

## Setup using Docker

This is the easiest way and the one I used most while building.

### 1. Add your OpenAI API key

Open [app/config.py](/Users/a91982/Work/Resume Project/RAG Document Intelligence System/app/config.py) and replace:

```python
OPENAI_API_KEY = "PASTE_YOUR_OPENAI_API_KEY_HERE"
```

with your real key.

### 2. Build and start everything

```bash
docker compose up --build
```

This starts 3 services:

- MySQL database
- FastAPI backend on `http://localhost:8000`
- Streamlit frontend on `http://localhost:8501`

### 3. Open the UI

Go to [http://localhost:8501](http://localhost:8501)

Upload a PDF and ask questions.

## API endpoints

### `POST /upload`

Upload a PDF file.

Example using curl:

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@sample.pdf"
```

### `POST /query`

Ask a question from the uploaded PDF data.

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is this PDF about?"}'
```

## Notes about FAISS persistence

The FAISS index is saved inside the local `data/` folder, so it stays available after container restarts.

Right now, if you upload multiple PDFs, their chunks get added into the same index. I kept it like that because it was simple and useful for testing. If you want a fresh index, just delete the `data/faiss_store` folder and upload again.

## MySQL logging table

The backend creates a table called `query_logs` automatically.

It stores:

- timestamp
- question
- answer
- response time in milliseconds

## Performance note

My goal was to keep response time under around 1.5 seconds for the retrieval part after the models are loaded. Full answer generation time can still depend on OpenAI API latency, so sometimes it can be a bit slower.

## Small project result

Compared to a simple keyword-search baseline I tested earlier, this RAG version improved Q&A accuracy by around 12% on my sample documents. It is not a huge research benchmark or anything, but it was a nice improvement for a student project.

## Common errors I faced

### 1. OpenAI key not added

If you forget to add the API key in [app/config.py](/Users/a91982/Work/Resume Project/RAG Document Intelligence System/app/config.py), the app will still retrieve the top chunks, but it will not generate a real LLM answer.

### 2. MySQL is not ready yet

Sometimes the backend starts a little early before MySQL is fully ready. I added retry logic for this because it happened to me a lot during Docker testing.

### 3. First run is slow

The first run downloads the embedding model, so startup can feel slow. After that it is much better.

### 4. Port already in use

If `8000`, `8501`, or `3307` is already busy, Docker Compose will fail. In that case, stop the other process or change the port mapping in [docker-compose.yml](/Users/a91982/Work/Resume Project/RAG Document Intelligence System/docker-compose.yml).

### 5. No text found in PDF

Some PDFs are scanned-image PDFs, so normal text extraction may fail. This project currently works best on text-based PDFs.

## If you want to run without Docker

You can also do it manually:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
streamlit run streamlit_app.py
```

But for local manual runs, you may need to change the MySQL host/port in [app/config.py](/Users/a91982/Work/Resume Project/RAG Document Intelligence System/app/config.py) depending on your setup.

## Future improvements

- Add support for multiple users
- Add OCR for scanned PDFs
- Add delete/reset index button
- Support open-source local LLMs too
- Store source paragraph previews with the answer

## Why I made this

I built this to learn how real RAG systems work step by step instead of just reading theory. It helped me understand chunking, embeddings, vector search, prompting, and how to connect backend + frontend + database in one project.
