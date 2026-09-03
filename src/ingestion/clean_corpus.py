"""
One-time cleanup script: removes stray backslash-escaping (e.g. "\#", "\&")
that got introduced into the corpus files, likely from copy-pasting markdown
text. Run once, then verify with verify_corpus.py or by rebuilding the index.
"""

from pathlib import Path

corpus_dir = Path("corpus")
md_files = sorted(corpus_dir.rglob("*.md"))

fixed_count = 0
for filepath in md_files:
    original = filepath.read_text(encoding="utf-8-sig")
    cleaned = original.replace("\\#", "#").replace("\\&", "&")

    if cleaned != original:
        filepath.write_text(cleaned, encoding="utf-8")
        fixed_count += 1
        print(f"Fixed: {filepath.name}")

print(f"\nDone. Cleaned {fixed_count} of {len(md_files)} files.")