"""
Centralized configuration, loaded from environment variables (see .env.example
at the repo root). Every value has a sane default so the app also runs
without an .env file present.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Ollama ---
    ollama_base_url: str = "http://ollama:11434"
    llm_model: str = "mistral"
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = 768  # nomic-embed-text output dimension

    # --- Qdrant ---
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "graphrag_chunks"

    # --- Neo4j ---
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "graphrag_password"

    # --- Ingestion ---
    chunking_strategy: str = "recursive"  # "recursive" | "sentence" | "semantic"
    chunk_size: int = 1000          # characters per chunk (recursive strategy)
    chunk_overlap: int = 150        # characters of overlap between chunks (recursive)
    chunk_sentences_per_chunk: int = 8   # sentences per chunk (sentence strategy)
    chunk_sentence_overlap: int = 1      # sentence-level overlap (sentence strategy)
    chunk_semantic_max_sentences: int = 20   # max sentences per chunk (semantic)
    chunk_semantic_threshold: int = 80       # percentile for breakpoint (semantic): 80 = split on top 20% biggest jumps
    upload_dir: str = "/app/data/uploads"
    max_upload_mb: int = 50
    ingestion_concurrency: int = 2  # parallel chunks during embedding phase (keep low on CPU-only Ollama; 4 caused 500 overload)
    extraction_concurrency: int = 1 # parallel chunks during LLM extraction (CPU-bound, keep low)

    # --- Retrieval ---
    vector_top_k: int = 13
    graph_expand_hops: int = 1      # how many hops to expand from seed entities
    graph_max_nodes: int = 40       # cap returned subgraph size for the viz panel

    # --- CORS ---
    cors_allow_origins: str = "*"


settings = Settings()
