# RAG Book Assistant

A Streamlit application that lets users upload a PDF book and ask questions grounded in its content. It uses LangChain, ChromaDB, Hugging Face sentence-transformer embeddings, and a Groq-hosted chat model.

## Features

- PDF upload with file type and size validation (25 MB maximum).
- PDF text extraction with page metadata.
- Text chunking with overlap.
- Local embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
- Persistent, separate Chroma collection for each PDF's content hash.
- MMR retrieval.
- Context-grounded answers with an explicit fallback when the answer is not in the document.
- Retrieved page numbers and passages shown with answers.
- Automated tests with pytest.
- GitHub Actions workflow to run tests on pushes and pull requests.

## Project structure

```text
rag-book-assistant/
├── app.py
├── rag_utils.py
├── requirements.txt
├── .env.example
├── .gitignore
├── pytest.ini
├── tests/
│   └── test_rag_utils.py
└── .github/
    └── workflows/
        └── tests.yml
```

## Setup

Use Python 3.11 where possible.

```bash
git clone https://github.com/priyanshu847700/RAG_BOOK_ASSISTANCE.git
cd RAG_BOOK_ASSISTANCE

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create a `.env` file from `.env.example` and add your own Groq API key:

```env
GROQ_API_KEY=your_actual_key
GROQ_MODEL=openai/gpt-oss-120b
```

Never commit `.env` or publish API keys.

## Run

```bash
streamlit run app.py
```

## Run tests

```bash
pytest -q
```

The test suite checks upload validation, book identifiers, question validation, chunking, page metadata, and source formatting.

## How it works

1. The app validates the uploaded PDF.
2. The PDF is read and page numbers are preserved.
3. Text is split into overlapping chunks.
4. Hugging Face creates local embeddings.
5. ChromaDB stores chunks in a collection associated with the PDF's content hash.
6. MMR retrieves relevant chunks for a question.
7. Groq generates an answer using the retrieved context.
8. The app displays the answer and source page numbers.

## Limitations

- Scanned/image-only PDFs need OCR; this version reports when no extractable text is found.
- The first embedding-model download requires internet access.
- Retrieved text is sent to the configured Groq API for answer generation. Do not upload confidential documents unless you are permitted to share their content with that service.
- RAG answers can be incomplete or incorrect. Check important answers against the source book.
- ChromaDB stores data locally in `chroma_db/`. Back up or delete this folder as appropriate.

## Security notes

- Keep API keys in `.env`, never in source code.
- The app checks the file header and upload size before processing.
- Uploaded PDFs are parsed in memory rather than written to a temporary file.
- The prompt treats document content as untrusted context and instructs the model not to follow embedded instructions.
