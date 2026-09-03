"""
This script reads ONE policy document and splits it into clause-level
chunks, based on numbered sections like 2.1, 2.2, etc.

We're testing this on a single file first before running it on all 34.
"""

import re
from pathlib import Path

def parse_document(filepath: str) -> list[dict]:
    """
    Reads a markdown policy document and splits it into clause-level chunks.
    Returns a list of dicts, each with the clause text and its metadata.
    """
    text = Path(filepath).read_text(encoding="utf-8-sig")

    # Extract the Document ID from the header (e.g. "Document ID: COC-04")
    doc_id_match = re.search(r"Document ID:\s*(\S+)", text)
    doc_id = doc_id_match.group(1) if doc_id_match else "UNKNOWN"

    # Extract the document title (the first line, which starts with "# ")
    title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Untitled"

    # Find all numbered clauses like "2.1 Some text..." up until the next
    # clause number or end of file
    clause_pattern = re.compile(
        r"(\d+\.\d+)\s+(.+?)(?=\n\d+\.\d+\s|\Z)", re.DOTALL
    )

    chunks = []
    for match in clause_pattern.finditer(text):
        clause_id = match.group(1)
        clause_text = match.group(2).strip().replace("\n", " ")

        if len(clause_text) < 10:  # skip near-empty matches
            continue

        chunks.append({
            "doc_id": doc_id,
            "title": title,
            "clause_id": clause_id,
            "text": clause_text,
            "source_file": filepath,
        })

    return chunks


if __name__ == "__main__":
    # Test on just one document first
    test_file = "corpus/code-of-conduct/COC-04-gifts-entertainment.md"
    chunks = parse_document(test_file)

    print(f"Found {len(chunks)} clauses in {test_file}\n")
    for chunk in chunks:
        print(f"[{chunk['doc_id']} - {chunk['clause_id']}] {chunk['text'][:80]}...")