import json
import os

import faiss
import fitz
import numpy as np
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from app import config


class RAGPipeline:
    def __init__(self):
        # Create folders at startup so file saving does not fail later.
        os.makedirs(config.UPLOAD_DIR, exist_ok=True)
        os.makedirs(config.FAISS_DIR, exist_ok=True)

        # Load the local embedding model once and reuse it for all uploads and queries.
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        self.embedding_size = self.embedding_model.get_sentence_embedding_dimension()

        # Use a tokenizer-aware splitter so the chunk size is closer to token count.
        tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL_NAME)
        self.splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=tokenizer,
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )

        # Keep chunk text + metadata in a simple JSON file next to the FAISS index.
        self.index = faiss.IndexFlatIP(self.embedding_size)
        self.chunk_records = []

        # Load old index from disk if it already exists.
        self.load_saved_index()

        # Create the LLM client once. If the key is missing, query fallback still works.
        self.llm = self.build_llm()

    def build_llm(self):
        if config.LLM_PROVIDER != "openai":
            return None

        if not config.OPENAI_API_KEY or "PASTE_YOUR" in config.OPENAI_API_KEY:
            return None

        llm_settings = {
            "model": config.OPENAI_MODEL,
            "temperature": config.LLM_TEMPERATURE,
            "api_key": config.OPENAI_API_KEY,
        }

        if config.OPENAI_BASE_URL:
            llm_settings["base_url"] = config.OPENAI_BASE_URL

        return ChatOpenAI(**llm_settings)

    def load_saved_index(self):
        if os.path.exists(config.FAISS_INDEX_FILE) and os.path.exists(
            config.FAISS_METADATA_FILE
        ):
            self.index = faiss.read_index(config.FAISS_INDEX_FILE)

            with open(config.FAISS_METADATA_FILE, "r", encoding="utf-8") as file:
                self.chunk_records = json.load(file)

    def save_index(self):
        faiss.write_index(self.index, config.FAISS_INDEX_FILE)

        with open(config.FAISS_METADATA_FILE, "w", encoding="utf-8") as file:
            json.dump(self.chunk_records, file, ensure_ascii=False, indent=2)

    def ingest_pdf(self, file_path):
        file_name = os.path.basename(file_path)

        pdf_document = fitz.open(file_path)
        documents = []

        # Read text page by page so we can keep page number metadata for citations.
        for page_number, page in enumerate(pdf_document, start=1):
            page_text = page.get_text("text")

            if page_text.strip():
                documents.append(
                    Document(
                        page_content=page_text,
                        metadata={
                            "source": file_name,
                            "page": page_number,
                        },
                    )
                )

        pdf_document.close()

        if not documents:
            raise ValueError("No readable text was found inside this PDF.")

        # Split long page text into smaller chunks for better retrieval.
        chunks = self.splitter.split_documents(documents)

        if not chunks:
            raise ValueError("Chunking failed, so no text chunks were created.")

        texts = []
        new_chunk_records = []
        current_count = len(self.chunk_records)

        for index, chunk in enumerate(chunks, start=1):
            chunk_id = f"chunk_{current_count + index}"
            chunk_text = chunk.page_content.strip()

            texts.append(chunk_text)
            new_chunk_records.append(
                {
                    "chunk_id": chunk_id,
                    "source": chunk.metadata.get("source", file_name),
                    "page": chunk.metadata.get("page", "unknown"),
                    "text": chunk_text,
                }
            )

        # Create embeddings locally. These are saved into FAISS after this.
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        embeddings = np.array(embeddings).astype("float32")
        self.index.add(embeddings)
        self.chunk_records.extend(new_chunk_records)
        self.save_index()

        return {
            "file_name": file_name,
            "pages_read": len(documents),
            "chunks_added": len(new_chunk_records),
            "total_chunks": len(self.chunk_records),
        }

    def retrieve_chunks(self, question, top_k=None):
        if self.index.ntotal == 0 or not self.chunk_records:
            raise ValueError("Please upload a PDF before asking a question.")

        if top_k is None:
            top_k = config.TOP_K

        question_embedding = self.embedding_model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        question_embedding = np.array(question_embedding).astype("float32")

        # Search the top matching chunk vectors from FAISS.
        scores, indices = self.index.search(
            question_embedding,
            min(top_k, len(self.chunk_records)),
        )

        results = []

        for score, chunk_index in zip(scores[0], indices[0]):
            if chunk_index == -1:
                continue

            chunk_data = self.chunk_records[chunk_index].copy()
            chunk_data["score"] = float(score)
            results.append(chunk_data)

        return results

    def answer_question(self, question):
        retrieved_chunks = self.retrieve_chunks(question)
        context_parts = []

        for chunk in retrieved_chunks:
            context_parts.append(
                f"[{chunk['chunk_id']} | source={chunk['source']} | page={chunk['page']}]\n"
                f"{chunk['text']}"
            )

        context_text = "\n\n".join(context_parts)

        prompt = PromptTemplate.from_template(
            """
            You are a helpful document question answering assistant.
            Answer only from the context below.
            If the answer is not present in the context, say "I could not find the answer in the uploaded PDF."
            Keep the answer clear and beginner-friendly.
            Add citations in square brackets using the chunk ids, for example [chunk_2].
            Do not use outside knowledge.

            Question:
            {question}

            Context:
            {context}
            """
        )

        final_prompt = prompt.format(question=question, context=context_text)

        if self.llm is not None:
            try:
                response = self.llm.invoke(final_prompt)
                answer = response.content
            except Exception as error:
                answer = self.build_fallback_answer(
                    retrieved_chunks,
                    f"LLM call failed ({error}), so I am showing the most relevant retrieved chunks instead:\n\n",
                )
        else:
            answer = self.build_fallback_answer(
                retrieved_chunks,
                "OpenAI key is not configured in app/config.py, so I am showing the most relevant retrieved chunks instead:\n\n",
            )

        sources = []

        for chunk in retrieved_chunks:
            sources.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "source": chunk["source"],
                    "page": chunk["page"],
                    "score": round(chunk["score"], 4),
                }
            )

        return {
            "answer": answer,
            "sources": sources,
        }

    def build_fallback_answer(self, retrieved_chunks, intro_text):
        # This fallback is not a real generative answer, but it keeps the app useful.
        fallback_lines = []

        for chunk in retrieved_chunks:
            short_text = chunk["text"].replace("\n", " ").strip()
            fallback_lines.append(
                f"{short_text[:220]}... [{chunk['chunk_id']}]"
            )

        return intro_text + "\n".join(fallback_lines)
