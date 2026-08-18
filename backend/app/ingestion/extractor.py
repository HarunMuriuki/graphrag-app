"""
LLM-based entity & relationship extraction (the "graph-building" step of the
ingestion pipeline). For each chunk, we ask Mistral (via Ollama) to return a
JSON object listing entities and relationships mentioned in that chunk.

Small local models are not perfectly reliable JSON generators, so this module
is defensive: it asks for Ollama's `format: json` constrained decoding, and
additionally falls back to regex-extracting the first {...} block if parsing
still fails, and simply returns an empty graph for that chunk (rather than
crashing the whole ingestion job) if extraction fails entirely.
"""
import json
import re

from app.db.ollama_client import ollama_client

SYSTEM_PROMPT = (
    "You are an information-extraction engine. You read a short passage of "
    "text and extract a knowledge graph from it: the named entities present "
    "(people, organizations, places, products, concepts, dates, etc.) and the "
    "relationships between them that are explicitly stated or strongly "
    "implied. Be conservative -- only extract what the text actually "
    "supports. Always respond with a single JSON object and nothing else."
)

PROMPT_TEMPLATE = """Extract entities and relationships from the passage below.

Respond with ONLY a JSON object in exactly this shape:
{{
  "entities": [{{"name": "...", "type": "..."}}],
  "relations": [{{"source": "...", "target": "...", "type": "...", "description": "..."}}]
}}

Rules:
- "type" for entities should be a short category like Person, Organization, Location, Product, Event, Concept, Date.
- "type" for relations should be a short snake_case label like "works_at", "located_in", "part_of", "caused_by".
- "source" and "target" in relations must exactly match a "name" from the entities list.
- If no entities or relations are found, return {{"entities": [], "relations": []}}.
- Do not include any text outside the JSON object.

Passage:
\"\"\"
{chunk}
\"\"\"
"""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


async def extract_entities_relations(chunk: str) -> dict[str, list[dict]]:
    prompt = PROMPT_TEMPLATE.format(chunk=chunk[:4000])  # guard against huge chunks
    try:
        raw = await ollama_client.generate(
            prompt=prompt, system=SYSTEM_PROMPT, temperature=0.0, json_mode=True
        )
        return _parse(raw)
    except Exception:
        return {"entities": [], "relations": []}


def _parse(raw: str) -> dict[str, list[dict]]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = _JSON_BLOCK_RE.search(raw)
        if not match:
            return {"entities": [], "relations": []}
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"entities": [], "relations": []}

    entities = data.get("entities", [])
    relations = data.get("relations", [])
    if not isinstance(entities, list):
        entities = []
    if not isinstance(relations, list):
        relations = []

    clean_entities = [
        {"name": str(e.get("name", "")).strip(), "type": str(e.get("type", "Unknown")).strip()}
        for e in entities
        if isinstance(e, dict) and e.get("name")
    ]
    clean_relations = [
        {
            "source": str(r.get("source", "")).strip(),
            "target": str(r.get("target", "")).strip(),
            "type": str(r.get("type", "related_to")).strip(),
            "description": str(r.get("description", "")).strip(),
        }
        for r in relations
        if isinstance(r, dict) and r.get("source") and r.get("target")
    ]
    return {"entities": clean_entities, "relations": clean_relations}
