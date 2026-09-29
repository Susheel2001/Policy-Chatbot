from __future__ import annotations

import html
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

from .pdf_text import extract_pdf_blocks


SUPPORTED_DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".markdown",
    ".html",
    ".htm",
    ".rtf",
    ".csv",
    ".json",
}


def extract_document_blocks(document_path: str | Path) -> list[dict[str, str]]:
    """Return readable blocks from a commonly used document format."""
    path = Path(document_path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_blocks(path)
    if suffix == ".docx":
        text = _extract_docx_text(path)
    elif suffix in {".html", ".htm"}:
        text = _extract_html_text(path)
    elif suffix == ".rtf":
        text = _extract_rtf_text(path)
    elif suffix in SUPPORTED_DOCUMENT_EXTENSIONS:
        text = _read_plain_text(path)
    else:
        raise ValueError(f"Unsupported document type: {path.suffix or 'no extension'}")

    blocks = [block.strip() for block in re.split(r"\n\s*\n+", text) if block.strip()]
    if not blocks:
        raise ValueError(f"Could not extract any usable text from {path}")
    return [{"block_id": f"block-{index}", "text": block} for index, block in enumerate(blocks, start=1)]


def _read_plain_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return ""


def _extract_docx_text(path: Path) -> str:
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))

    paragraphs: list[str] = []
    for paragraph in root.iter(f"{namespace}p"):
        text = "".join(node.text or "" for node in paragraph.iter(f"{namespace}t")).strip()
        if text:
            paragraphs.append(text)
    return "\n\n".join(paragraphs)


class _HTMLTextExtractor(HTMLParser):
    BLOCK_TAGS = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _extract_html_text(path: Path) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(_read_plain_text(path))
    return html.unescape("".join(parser.parts))


def _extract_rtf_text(path: Path) -> str:
    text = _read_plain_text(path)
    text = re.sub(r"\\par[d]?", "\n\n", text)
    text = re.sub(r"\\'[0-9a-fA-F]{2}", "", text)
    text = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", text)
    return text.replace("{", "").replace("}", "")
