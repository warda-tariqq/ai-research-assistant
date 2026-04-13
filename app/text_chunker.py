from typing import List, Dict


def chunk_text(
    pages: List[Dict],
    source_file: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> List[Dict]:
    """
    Split extracted PDF pages into overlapping text chunks.
    """
    chunks = []
    chunk_id = 0

    for page in pages:
        page_number = page["page_number"]
        text = page["text"].strip()

        if not text:
            continue

        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text_value = text[start:end].strip()

            if chunk_text_value:
                chunks.append({
                    "chunk_id": f"{source_file}_p{page_number}_c{chunk_id}",
                    "source_file": source_file,
                    "page_number": page_number,
                    "text": chunk_text_value
                })
                chunk_id += 1

            start += chunk_size - overlap

    return chunks