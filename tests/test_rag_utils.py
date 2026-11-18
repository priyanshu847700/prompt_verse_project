import pytest
from langchain_core.documents import Document

from rag_utils import (
    build_context,
    format_sources,
    get_book_id,
    get_collection_name,
    split_documents,
    validate_pdf_bytes,
    validate_question,
)


def test_rejects_empty_pdf():
    with pytest.raises(ValueError, match="empty"):
        validate_pdf_bytes(b"")


def test_rejects_non_pdf_file():
    with pytest.raises(ValueError, match="Invalid PDF"):
        validate_pdf_bytes(b"this is not a pdf")


def test_rejects_oversized_pdf():
    with pytest.raises(ValueError, match="too large"):
        validate_pdf_bytes(b"%PDF-" + b"x" * 20, max_size_bytes=10)


def test_accepts_pdf_header():
    validate_pdf_bytes(b"%PDF-1.7\ncontent", max_size_bytes=100)


def test_book_id_is_stable():
    assert get_book_id(b"%PDF-test") == get_book_id(b"%PDF-test")


def test_different_books_have_different_ids():
    assert get_book_id(b"%PDF-one") != get_book_id(b"%PDF-two")


def test_collection_name_is_safe():
    name = get_collection_name("abc-123")
    assert name == "book_abc-123"
    assert name.replace("_", "").replace("-", "").isalnum()


def test_question_is_trimmed():
    assert validate_question("  What is RAG?  ") == "What is RAG?"


def test_empty_question_is_rejected():
    with pytest.raises(ValueError, match="enter a question"):
        validate_question("   ")


def test_long_question_is_rejected():
    with pytest.raises(ValueError, match="too long"):
        validate_question("a" * 1001)


def test_split_documents_preserves_page_metadata():
    docs = [Document(page_content="word " * 500, metadata={"page": 3})]
    chunks = split_documents(docs, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(chunk.metadata["page"] == 3 for chunk in chunks)


def test_build_context_includes_page_number():
    docs = [Document(page_content="RAG retrieves context.", metadata={"page": 4})]
    assert "[Page 4]" in build_context(docs)


def test_format_sources_returns_unique_pages():
    docs = [
        Document(page_content="A", metadata={"page": 2}),
        Document(page_content="B", metadata={"page": 2}),
        Document(page_content="C", metadata={"page": 5}),
    ]
    assert format_sources(docs) == [2, 5]
