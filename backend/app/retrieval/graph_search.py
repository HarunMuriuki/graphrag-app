"""
Graph retrieval step: given the chunks that matched the vector search, find
the entities mentioned in them, then expand outward through the knowledge
graph to pull in related facts that may not appear verbatim in those chunks
but are relevant via the relationships extracted at ingestion time.
"""
from app.core.config import settings
from app.db import neo4j_client


def graph_context_for_chunks(chunk_ids: list[str]) -> dict:
    seed_entities = neo4j_client.get_entities_for_chunks(chunk_ids)
    seed_names = [e["name"] for e in seed_entities]
    subgraph = neo4j_client.expand_subgraph(
        seed_names,
        hops=settings.graph_expand_hops,
        max_nodes=settings.graph_max_nodes,
    )
    return {
        "seed_entities": seed_entities,
        "nodes": subgraph["nodes"],
        "edges": subgraph["edges"],
    }


def facts_as_text(edges: list[dict]) -> list[str]:
    """Render graph edges as short natural-language facts to feed the LLM,
    e.g. 'Acme Corp --[acquired]--> Widget Inc (in 2019 for $40M)'."""
    facts = []
    for e in edges:
        line = f"{e['source']} --[{e['label']}]--> {e['target']}"
        if e.get("description"):
            line += f" ({e['description']})"
        facts.append(line)
    return facts
