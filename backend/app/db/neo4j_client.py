"""
Wrapper around the Neo4j driver.

Graph schema used by this app:

    (:Chunk {id, doc_id, source, chunk_index, text})
    (:Entity {name, type})

    (:Chunk)-[:MENTIONS]->(:Entity)
    (:Entity)-[:RELATED {type, description}]->(:Entity)

`Entity.name` is treated as the natural key (case-sensitive, trimmed at
extraction time). `RELATED.type` holds the human-readable relationship label
extracted by the LLM (e.g. "works_at", "located_in") -- the Neo4j relationship
*label* itself is always the generic `RELATED`, which keeps the Cypher simple
and lets us store an arbitrary, LLM-generated vocabulary of relation types as
data rather than as schema.
"""
from typing import Any

from neo4j import GraphDatabase

from app.core.config import settings

_driver = None


def get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
        )
    return _driver


def close_driver() -> None:
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


def init_constraints() -> None:
    with get_driver().session() as session:
        session.run(
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS "
            "FOR (c:Chunk) REQUIRE c.id IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
        )


def write_chunk_graph(
    doc_id: str,
    chunk_id: str,
    chunk_index: int,
    source: str,
    text: str,
    entities: list[dict[str, str]],
    relations: list[dict[str, str]],
) -> None:
    """Persist one chunk's extracted entities & relations. Called once per
    chunk during ingestion. Idempotent: re-ingesting the same chunk_id
    overwrites the Chunk node and MERGEs (not duplicates) entities/edges."""
    with get_driver().session() as session:
        session.run(
            """
            MERGE (c:Chunk {id: $chunk_id})
            SET c.doc_id = $doc_id,
                c.source = $source,
                c.chunk_index = $chunk_index,
                c.text = $text
            """,
            chunk_id=chunk_id,
            doc_id=doc_id,
            source=source,
            chunk_index=chunk_index,
            text=text,
        )

        for entity in entities:
            name = (entity.get("name") or "").strip()
            if not name:
                continue
            etype = (entity.get("type") or "Unknown").strip()
            session.run(
                """
                MERGE (e:Entity {name: $name})
                ON CREATE SET e.type = $type
                ON MATCH SET e.type = coalesce(e.type, $type)
                WITH e
                MATCH (c:Chunk {id: $chunk_id})
                MERGE (c)-[:MENTIONS]->(e)
                """,
                name=name,
                type=etype,
                chunk_id=chunk_id,
            )

        for rel in relations:
            src = (rel.get("source") or "").strip()
            tgt = (rel.get("target") or "").strip()
            if not src or not tgt or src == tgt:
                continue
            rel_type = (rel.get("type") or "related_to").strip()
            description = (rel.get("description") or "").strip()
            session.run(
                """
                MERGE (a:Entity {name: $src})
                MERGE (b:Entity {name: $tgt})
                MERGE (a)-[r:RELATED {type: $rel_type}]->(b)
                SET r.description = $description
                """,
                src=src,
                tgt=tgt,
                rel_type=rel_type,
                description=description,
            )


def get_entities_for_chunks(chunk_ids: list[str]) -> list[dict[str, str]]:
    if not chunk_ids:
        return []
    with get_driver().session() as session:
        result = session.run(
            """
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
            WHERE c.id IN $chunk_ids
            RETURN DISTINCT e.name AS name, e.type AS type
            """,
            chunk_ids=chunk_ids,
        )
        return [{"name": r["name"], "type": r["type"]} for r in result]


def expand_subgraph(
    seed_names: list[str], hops: int = 1, max_nodes: int = 40
) -> dict[str, list[dict[str, Any]]]:
    """Breadth-first expansion from a set of seed entity names, one hop at a
    time (each hop is a single, cheap Cypher query). Capped at max_nodes so
    the graph panel stays readable."""
    if not seed_names:
        return {"nodes": [], "edges": []}

    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[tuple, dict[str, Any]] = {}
    frontier = set(seed_names)
    seen = set(seed_names)

    with get_driver().session() as session:
        # seed node metadata
        result = session.run(
            "MATCH (e:Entity) WHERE e.name IN $names RETURN e.name AS name, e.type AS type",
            names=list(seed_names),
        )
        for r in result:
            nodes[r["name"]] = {"id": r["name"], "label": r["name"], "type": r["type"]}

        for _ in range(max(hops, 0)):
            if not frontier or len(nodes) >= max_nodes:
                break
            result = session.run(
                """
                MATCH (e:Entity)-[r:RELATED]-(n:Entity)
                WHERE e.name IN $names
                RETURN e.name AS a, e.type AS a_type, n.name AS b, n.type AS b_type,
                       r.type AS rel_type, r.description AS description
                """,
                names=list(frontier),
            )
            next_frontier = set()
            for row in result:
                a, b = row["a"], row["b"]
                nodes.setdefault(a, {"id": a, "label": a, "type": row["a_type"]})
                nodes.setdefault(b, {"id": b, "label": b, "type": row["b_type"]})
                key = tuple(sorted((a, b))) + (row["rel_type"],)
                edges[key] = {
                    "source": a,
                    "target": b,
                    "label": row["rel_type"] or "related_to",
                    "description": row["description"] or "",
                }
                if b not in seen:
                    next_frontier.add(b)
                if a not in seen:
                    next_frontier.add(a)
            seen |= next_frontier
            frontier = next_frontier

    node_list = list(nodes.values())[:max_nodes]
    kept_ids = {n["id"] for n in node_list}
    edge_list = [
        e for e in edges.values() if e["source"] in kept_ids and e["target"] in kept_ids
    ]
    return {"nodes": node_list, "edges": edge_list}


def get_graph_sample(limit: int = 150) -> dict[str, list[dict[str, Any]]]:
    """Returns an arbitrary sample of the graph for the 'overview' view shown
    before the user has asked any question yet."""
    with get_driver().session() as session:
        result = session.run(
            """
            MATCH (a:Entity)-[r:RELATED]->(b:Entity)
            RETURN a.name AS a, a.type AS a_type, b.name AS b, b.type AS b_type,
                   r.type AS rel_type, r.description AS description
            LIMIT $limit
            """,
            limit=limit,
        )
        nodes: dict[str, dict[str, Any]] = {}
        edges = []
        for row in result:
            nodes.setdefault(row["a"], {"id": row["a"], "label": row["a"], "type": row["a_type"]})
            nodes.setdefault(row["b"], {"id": row["b"], "label": row["b"], "type": row["b_type"]})
            edges.append(
                {
                    "source": row["a"],
                    "target": row["b"],
                    "label": row["rel_type"] or "related_to",
                    "description": row["description"] or "",
                }
            )
        return {"nodes": list(nodes.values()), "edges": edges}


def document_stats() -> list[dict[str, Any]]:
    with get_driver().session() as session:
        result = session.run(
            """
            MATCH (c:Chunk)
            RETURN c.doc_id AS doc_id, c.source AS source, count(*) AS chunks
            """
        )
        return [dict(r) for r in result]


def ping() -> bool:
    try:
        with get_driver().session() as session:
            session.run("RETURN 1")
        return True
    except Exception:
        return False
