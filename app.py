import hashlib
import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

st.set_page_config(page_title="AI Knowledge Base", page_icon="📚", layout="wide")
st.title("📚 AI-Powered Knowledge Base")
st.write("Upload one or more PDFs and ask grounded questions with source citations.")


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def build_vector_store(corpus_hash: str, chunks_data: list[dict]):
    embeddings = get_embeddings()
    documents = [
        Document(
            page_content=chunk["text"],
            metadata={
                "source": chunk["source"],
                "page": chunk["page"],
            },
        )
        for chunk in chunks_data
    ]
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=f"kb_{corpus_hash[:16]}",
    )


def extract_from_pdfs(files) -> tuple[list[dict], list[str], str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = []
    skipped_files = []
    hasher = hashlib.sha256()

    for file in sorted(files, key=lambda f: f.name):
        file_bytes = file.getvalue()
        hasher.update(file.name.encode("utf-8"))
        hasher.update(file_bytes)

        reader = PdfReader(file)
        file_has_text = False

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if page_text:
                file_has_text = True
                for chunk in text_splitter.split_text(page_text):
                    chunks.append({
                        "text": chunk,
                        "source": file.name,
                        "page": page_number,
                    })

        if not file_has_text:
            skipped_files.append(file.name)

    return chunks, skipped_files, hasher.hexdigest()


with st.sidebar:
    st.header("Documents & Chat")
    uploaded_files = st.file_uploader(
        "Choose PDF file(s)",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded_files:
    chunks, skipped_files, corpus_hash = extract_from_pdfs(uploaded_files)

    for fname in skipped_files:
        st.warning(f"No readable text found in `{fname}`.")

    if chunks:
        st.success(f"Indexed {len(uploaded_files) - len(skipped_files)} PDF(s) into ChromaDB!")

        vector_store = build_vector_store(corpus_hash, chunks)

        with st.expander("Debug Info"):
            st.subheader("First chunk")
            st.write(chunks[0])

            st.subheader("First three metadata entries")
            st.write([
                {"source": c["source"], "page": c["page"]}
                for c in chunks[:3]
            ])

            st.subheader("Stored ChromaDB metadata")
            stored_data = vector_store.get(include=["metadatas"])
            st.write(stored_data["metadatas"][:3])

        llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            google_api_key=st.secrets["GEMINI_API_KEY"],
            temperature=0,
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})

        # Render previous chat messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    st.caption("Sources: " + ", ".join(msg["sources"]))

        # Chat input box
        question = st.chat_input("Ask a question about your uploaded PDFs...")

        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            relevant_docs = retriever.invoke(question)

            sources = []
            context_blocks = []
            for doc in relevant_docs:
                src = doc.metadata.get("source", "Uploaded PDF")
                pg = doc.metadata.get("page", "?")
                label = f"{src} (Page {pg})"
                if label not in sources:
                    sources.append(label)
                context_blocks.append(f"[{label}]\n{doc.page_content}")

            context = "\n\n---\n\n".join(context_blocks)

            prompt = f"""Answer the question using only the context below.
If the answer is not in the context, say you could not find the answer in the uploaded PDFs.

Context:
{context}

Question:
{question}
"""

            with st.chat_message("assistant"):
                try:
                    response = llm.invoke(prompt)
                    answer = response.content
                    if not isinstance(answer, str):
                        answer = "".join(
                            item["text"]
                            for item in answer
                            if isinstance(item, dict) and item.get("type") == "text"
                        )

                    st.markdown(answer)
                    if sources:
                        st.caption("Sources: " + ", ".join(sources))

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })
                except Exception as e:
                    st.error(f"Error generating answer: {e}")

        st.divider()
        st.write(f"Total text chunks: {len(chunks)}")
        st.text_area(
            "First text chunk",
            chunks[0]["text"],
            height=180,
        )
    else:
        st.warning("No readable text found in any of the uploaded PDFs.")
else:
    st.info("Upload one or more PDF files to get started.")