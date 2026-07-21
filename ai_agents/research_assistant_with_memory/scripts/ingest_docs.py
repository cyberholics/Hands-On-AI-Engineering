"""
Bulk ingestion from the command line.

    python scripts/ingest_docs.py data/
    python scripts/ingest_docs.py data/paper.pdf data/notes.md

Chunks are embedded with BGE and written into the shared Actian collection.
Re-running on the same file appends a fresh copy, so clear the collection
first if you are re-indexing rather than adding.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_assistant.embedder import Embedder  # noqa: E402
from research_assistant.ingest import (  # noqa: E402
    SUPPORTED_SUFFIXES,
    ingest_directory,
    ingest_path,
)
from research_assistant.vectorstore import (  # noqa: E402
    get_client,
    get_collection_counts,
    setup_collections,
)


def main(argv: list[str]) -> int:
    targets = [Path(a) for a in argv] or [Path("data")]

    missing = [str(p) for p in targets if not p.exists()]
    if missing:
        print(f"Not found: {', '.join(missing)}")
        return 1

    embedder = Embedder()
    client = get_client()
    setup_collections(client, dim=embedder.dimension)

    summaries: list[dict] = []
    for target in targets:
        if target.is_dir():
            found = ingest_directory(client, embedder, target)
            if not found:
                print(f"No supported files in {target} "
                      f"({', '.join(sorted(SUPPORTED_SUFFIXES))})")
            summaries.extend(found)
        else:
            if target.suffix.lower() not in SUPPORTED_SUFFIXES:
                print(f"Skipping unsupported file: {target.name}")
                continue
            summaries.append(ingest_path(client, embedder, target))

    if not summaries:
        print("Nothing ingested.")
        return 1

    print()
    for summary in summaries:
        print(f"  {summary['doc_title']:<45} {summary['chunks']:>5} chunks")

    total = sum(s["chunks"] for s in summaries)
    print(f"\nIngested {len(summaries)} document(s), {total} chunks.")

    counts = get_collection_counts(client)
    for name, count in counts.items():
        print(f"  {name}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
