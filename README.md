
# AI-Powered Knowledge Base

## 🚀 Live Demo

Try the application here: [AI-Powered Knowledge Base](https://ai-knowledge-base-9jiwwnpo2kryv4urwtwarg.streamlit.app/)

## Project Overview

AI-Powered Knowledge Base is a document question-answering application built using Retrieval-Augmented Generation (RAG).

It allows users to upload one or more PDF documents and ask questions about their content. The application retrieves relevant information from the uploaded documents and uses Google's Gemini model to generate context-based answers with source references.

## Features

- Upload multiple PDF documents
- Extract text from PDF files
- Split extracted text into smaller chunks
- Generate text embeddings using Hugging Face
- Store and retrieve vectors using ChromaDB
- Perform semantic search using LangChain
- Generate answers using Google Gemini
- Display source PDF names and page numbers
- Interactive chat interface
- Maintain conversation history during the session
- Clear chat functionality
- Cache embeddings and vector stores for improved performance
- Identify uploaded document collections using SHA-256 hashing

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Streamlit | Web application interface |
| LangChain | RAG pipeline and document retrieval |
| PyPDF | PDF text extraction |
| Hugging Face | Text embeddings |
| ChromaDB | Vector database |
| Google Gemini | AI-powered answer generation |
| Sentence Transformers | Embedding model |

## How It Works

1. Users upload one or more PDF documents.
2. PyPDF extracts text from the documents.
3. RecursiveCharacterTextSplitter divides the text into smaller chunks.
4. Hugging Face generates embeddings for the chunks.
5. ChromaDB stores the document vectors and metadata.
6. Users enter questions through the chat interface.
7. LangChain retrieves the most relevant document chunks.
8. The retrieved information is passed to Google Gemini.
9. Gemini generates an answer using the retrieved context.
10. The application displays the answer along with source PDF names and page numbers.

## RAG Architecture

PDF Upload → Text Extraction → Text Chunking → Embeddings → ChromaDB → Relevant Document Retrieval → Gemini → Answer with Sources

## Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the Project Folder

```bash
cd ai-knowledge-base
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

Windows Command Prompt:

```bash
.venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure the Gemini API Key

Create a folder named `.streamlit` in the project directory.

Inside it, create `secrets.toml` and add:

```toml
GEMINI_API_KEY = "your_gemini_api_key"
```

Keep your API key private. Never upload `secrets.toml` to GitHub.

### 7. Run the Application

```bash
streamlit run app.py
```

## Embedding Model

The project uses:

`sentence-transformers/all-MiniLM-L6-v2`

This model converts text chunks into numerical vector representations for semantic similarity search.

## Project Structure

```text
ai-knowledge-base/
│
├── .streamlit/
│   └── secrets.toml
│
├── .venv/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

Note: `.streamlit/secrets.toml` and `.venv/` should not be uploaded to GitHub.

## Future Improvements

- Persistent chat history
- Improved document management
- Downloadable answers and summaries
- More advanced retrieval strategies
- Support for additional document formats

## Project Status

Completed – Mini Project

## Author

Kavya Mani
