"""
Central configuration. All tuneable constants live here.
Values are overridden by environment variables or a .env file.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Actian VectorAI DB
# ---------------------------------------------------------------------------

VECTORAI_URL = os.getenv("ACTIAN_VECTORAI_URL", "localhost:6574")
VECTORAI_ACCESS_TOKEN = os.getenv("ACTIAN_VECTORAI_ACCESS_TOKEN")

# The shared context layer. Every agent reads through these two collections.
#   research_documents - chunked source material (PDFs, papers, transcripts, notes)
#   research_memory    - persistent long-term memory that survives across sessions
DOCUMENTS_COLLECTION = "research_documents"
MEMORY_COLLECTION = "research_memory"

COLLECTIONS = [DOCUMENTS_COLLECTION, MEMORY_COLLECTION]

# ---------------------------------------------------------------------------
# Embeddings (BGE via sentence-transformers)
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

# BGE models are trained with an instruction prefix on the *query* side only.
# Passages are embedded raw. Skipping this costs a few points of recall.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ---------------------------------------------------------------------------
# Local inference (Ollama)
# ---------------------------------------------------------------------------

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e2b")

# Lower temperature for structured/plan/critique calls, higher for prose.
TEMPERATURE_STRUCTURED = 0.1
TEMPERATURE_PROSE = 0.4

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

CHUNK_SIZE = 900        # characters per chunk
CHUNK_OVERLAP = 150     # character overlap between consecutive chunks

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

TOP_K = 6               # chunks pulled per research query
MEMORY_TOP_K = 3        # long-term memory entries pulled per query
MIN_SCORE = 0.30        # below this a hit is treated as noise and dropped

# ---------------------------------------------------------------------------
# Critic / revision loop
# ---------------------------------------------------------------------------

# The Critic scores correctness, completeness and clarity from 1-5.
# If the average falls below this, the graph loops back for another pass.
CRITIC_PASS_THRESHOLD = float(os.getenv("CRITIC_PASS_THRESHOLD", "3.5"))

# Hard cap on revision loops so a stubborn Critic cannot spin forever.
MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", "2"))
