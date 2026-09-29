from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


DROP_LINE_HINTS = (
    "icici lombard general insurance company limited",
    "irda reg.",
    "mailing address:",
    "registered office address:",
    "toll free no",
    "alternate no",
    "website :",
    "website:",
    "e-mail :",
    "e-mail:",
    "customer support",
    "customersupport@icicilombard.com",
    "cin:",
    "uin:",
    "for buy/ renew/ service/ claim related queries",
    "log on to www.icicilombard.com",
)


def extract_pdf_blocks(pdf_path: str | Path) -> list[dict[str, str]]:
    reader = PdfReader(str(pdf_path))
    blocks: list[dict[str, str]] = []

    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned = _clean_page_text(raw_text)
        if cleaned:
            blocks.append(
                {
                    "block_id": f"page-{page_number}",
                    "text": cleaned,
                }
            )

    if not blocks:
        raise ValueError(f"Could not extract any usable text from {pdf_path}")

    return blocks


def extract_pdf_text(pdf_path: str | Path) -> str:
    return "\n\n".join(block["text"] for block in extract_pdf_blocks(pdf_path))


def _clean_page_text(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        cleaned = " ".join(line.split()).strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if any(hint in lowered for hint in DROP_LINE_HINTS):
            continue
        if cleaned.isdigit():
            continue
        if len(cleaned) < 3:
            continue
        lines.append(cleaned)

    if not lines:
        return ""

    collapsed = "\n".join(_dedupe_consecutive(lines))
    return collapsed.strip()


def _dedupe_consecutive(lines: list[str]) -> list[str]:
    deduped: list[str] = []
    previous = None
    for line in lines:
        if line != previous:
            deduped.append(line)
        previous = line
    return deduped