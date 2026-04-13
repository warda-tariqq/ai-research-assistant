from pathlib import Path
import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path: str):
    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_file)
    pages = []

    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({
            "page_number": i + 1,
            "text": text.strip()
        })

    doc.close()
    return pages