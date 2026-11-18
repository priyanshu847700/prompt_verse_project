"""Core functions for the RAG Book Assistant."""

from __future__ import annotations

import hashlib
import re
from io import BytesIO
from typing import Iterable

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


MAX_PDF_SIZE_MB = 25
MAX_PDF_SIZE_BYTES = MAX_PDF_SIZE_MB * 1024 * 1024


def validate_pdf_bytes(pdf_bytes: bytes, max_size_bytes: int = MAX_PDF_SIZE_BYTES) -> None:
    """Validate basic PDF upload properties. Raises ValueError for invalid files."""
    if not pdf_bytes:
        raise ValueError("The uploaded file is empty.")

    if len(pdf_bytes) > max_size_bytes:
        raise ValueError(f"PDF is too large. Maximum allowed size is {max_size_bytes // (1024 * 1024)} MB.")

    if not pdf_bytes.lstrip().startswith(b"%PDF-"):
        raise ValueError("Invalid PDF file. Please upload a valid PDF document.")


def get_book_id(pdf_bytes: bytes) -> str:
    """Return a stable, safe identifier for a PDF's contents."""
    return hashlib.sha256(pdf_bytes).hexdigest()[:20]


def get_collection_name(book_id: str) -> str:
    """Create a Chroma-compatible collection name."""
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "", book_id)
    return f"book_{safe_id}"


def load_pdf_documents(pdf_bytes: bytes) -> list[Document]:
    """Read PDF pages and preserve page metadata."""
    validate_pdf_bytes(pdf_bytes)

    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:
                raise ValueError("This PDF is password-protected.") from exc

        documents = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"page": page_number},
                    )
                )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Could not read this PDF. It may be damaged or unsupported.") from exc

    if not documents:
        raise ValueError("No readable text was found. Scanned PDFs may require OCR.")

    return documents


def split_documents(
    documents: Iterable[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split pages into overlapping chunks while preserving metadata."""
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be non-negative and smaller than chunk size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(list(documents))

    if not chunks:
        raise ValueError("No text chunks were created from this PDF.")

    return chunks


def build_context(documents: Iterable[Document]) -> str:
    """Build context with page labels for the answer prompt."""
    sections = []
    for doc in documents:
        page = doc.metadata.get("page", "Unknown")
        sections.append(f"[Page {page}]\n{doc.page_content}")
    return "\n\n".join(sections)


def validate_question(question: str) -> str:
    """Normalize and validate a user question."""
    cleaned = (question or "").strip()
    if not cleaned:
        raise ValueError("Please enter a question.")
    if len(cleaned) > 1000:
        raise ValueError("Question is too long. Please keep it under 1000 characters.")
    return cleaned


def format_sources(documents: Iterable[Document]) -> list[int]:
    """Return unique page numbers in the order they were retrieved."""
    pages = []
    seen = set()
    for doc in documents:
        page = doc.metadata.get("page")
        if page is not None and page not in seen:
            seen.add(page)
            pages.append(int(page))
    return pages
