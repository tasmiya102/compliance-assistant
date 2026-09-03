"""
Runs our chunking logic across every document in the corpus, and prints
the LAST clause found in each file. This helps us spot any document that
was accidentally cut short (truncated) during creation, since a truncated
file will show its last clause ending mid-sentence or at an unexpectedly
early clause number.
"""

from pathlib import Path
from chunk_document import parse_document

corpus_dir = Path("corpus")
md_files = sorted(corpus_dir.rglob("*.md"))

print(f"Checking {len(md_files)} files...\n")

for filepath in md_files:
    chunks = parse_document(str(filepath))
    if not chunks:
        print(f"⚠️  {filepath.name}: NO CLAUSES FOUND — check this file!")
        continue

    last_chunk = chunks[-1]
    last_text = last_chunk["text"]
    ends_cleanly = last_text.rstrip().endswith(('.', '"'))

    flag = "" if ends_cleanly else "  ⚠️  POSSIBLE TRUNCATION"
    print(f"{filepath.name}: {len(chunks)} clauses, last = "
          f"[{last_chunk['clause_id']}]{flag}")
    if not ends_cleanly:
        print(f"    ...ends with: \"...{last_text[-60:]}\"")