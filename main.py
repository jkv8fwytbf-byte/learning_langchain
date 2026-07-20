"""Document loaders: source in → Document(s) out."""

import os

# WebBaseLoader reads this at import time.
os.environ.setdefault("USER_AGENT", "learning-doc-loaders/1.0")

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    WebBaseLoader,
    DirectoryLoader,
    PyPDFLoader,
)

DOCS_DIR = "./docs"


def show(docs: list[Document], preview: int = 120) -> None:
    print(f"  loaded {len(docs)} document(s)")
    for i, doc in enumerate(docs):
        text = doc.page_content[:preview].replace("\n", " ")
        print(f"  [{i}] {text!r}...")
        print(f"      metadata: {doc.metadata}")


def demo_document() -> None:
    # Manual Document — what every loader returns.
    doc = Document(
        page_content="This is a sample document.",
        metadata={"source": "manual", "author": "learner"},
    )
    print("Document:")
    print(f"  page_content: {doc.page_content!r}")
    print(f"  metadata: {doc.metadata}")


def demo_text_loader() -> None:
    # path in → list[Document] out
    docs = TextLoader(f"{DOCS_DIR}/intro.txt").load()
    show(docs)


def demo_directory_loader() -> None:
    # directory + glob in → Documents one-by-one via lazy_load()
    loader = DirectoryLoader(DOCS_DIR, glob="*.txt", loader_cls=TextLoader)
    print("  streaming with lazy_load():")
    for doc in loader.lazy_load():
        print(f"  - {doc.metadata['source']}: {doc.page_content[:40]!r}...")


def demo_web_loader() -> None:
    # URL in → list[Document] out
    docs = WebBaseLoader("https://en.wikipedia.org/wiki/Web_scraping").load()
    show(docs, preview=200)


def demo_pdf_loader() -> None:
    # PDF URL/path in → one Document per page
    url = "https://arxiv.org/pdf/1706.03762.pdf"  # Attention Is All You Need
    docs = PyPDFLoader(url).load()
    show(docs[:2], preview=100)  # first 2 pages is enough for a demo
    print(f"  (total pages: {len(docs)})")


if __name__ == "__main__":
    demos = [
        ("1. Document (the unit)", demo_document),
        ("2. TextLoader", demo_text_loader),
        ("3. DirectoryLoader + lazy_load", demo_directory_loader),
        ("4. WebBaseLoader", demo_web_loader),
        ("5. PyPDFLoader", demo_pdf_loader),
    ]
    for title, fn in demos:
        print(f"\n=== {title} ===")
        fn()
