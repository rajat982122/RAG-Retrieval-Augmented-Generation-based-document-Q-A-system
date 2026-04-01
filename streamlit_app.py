import requests
import streamlit as st

from app import config


def find_backend_url():
    # Try local URL first for manual runs, then Docker URL.
    possible_urls = [config.LOCAL_BACKEND_URL, config.BACKEND_URL]

    for url in possible_urls:
        try:
            response = requests.get(f"{url}/docs", timeout=2)
            if response.status_code == 200:
                return url
        except requests.RequestException:
            continue

    # If nothing responds yet, localhost is the safer default for non-Docker runs.
    return config.LOCAL_BACKEND_URL


st.set_page_config(page_title="RAG PDF Q&A", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "upload_status" not in st.session_state:
    st.session_state.upload_status = ""

backend_url = find_backend_url()

st.title("RAG Based Document Q&A System")
st.write("Upload a PDF and ask questions in plain English.")

with st.sidebar:
    st.header("PDF Upload")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if st.button("Upload and Index", use_container_width=True):
        if uploaded_file is None:
            st.warning("Please choose a PDF file first.")
        else:
            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "application/pdf",
                    )
                }

                response = requests.post(
                    f"{backend_url}/upload",
                    files=files,
                    timeout=config.REQUEST_TIMEOUT,
                )

                if response.status_code == 200:
                    data = response.json()
                    st.session_state.upload_status = (
                        f"Indexed {data['file_name']} with {data['chunks_added']} chunks."
                    )
                    st.success(st.session_state.upload_status)
                else:
                    st.error(response.json().get("detail", "Upload failed."))
            except requests.RequestException as error:
                st.error(f"Could not connect to backend: {error}")

    if st.session_state.upload_status:
        st.info(st.session_state.upload_status)

st.subheader("Ask Questions")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            st.write("Citations:")
            for source in message["sources"]:
                st.write(
                    f"- {source['chunk_id']} | {source['source']} | page {source['page']} | score {source['score']}"
                )

question = st.chat_input("Ask something from your PDF")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching relevant chunks and generating answer..."):
            try:
                response = requests.post(
                    f"{backend_url}/query",
                    json={"question": question},
                    timeout=config.REQUEST_TIMEOUT,
                )

                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])
                    response_time_ms = data.get("response_time_ms", 0)

                    st.markdown(answer)
                    st.caption(f"Response time: {response_time_ms} ms")

                    if sources:
                        st.write("Citations:")
                        for source in sources:
                            st.write(
                                f"- {source['chunk_id']} | {source['source']} | page {source['page']} | score {source['score']}"
                            )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )
                else:
                    error_text = response.json().get("detail", "Query failed.")
                    st.error(error_text)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_text,
                        }
                    )
            except requests.RequestException as error:
                error_text = f"Could not connect to backend: {error}"
                st.error(error_text)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_text,
                    }
                )
