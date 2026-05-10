import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

COLLECTION  = "image_search"
PRIMARY_DIM = 1024
SIGLIP_DIM  = 1152
PRIMARY_W   = 0.55
SIGLIP_W    = 0.45

_client = QdrantClient(host="localhost", port=6333)


def _point_id(product_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, product_id))


def health_check() -> dict:
    info = _client.get_collection(COLLECTION)
    return {"vectors": info.points_count}


def ensure_collection() -> None:
    existing = {c.name for c in _client.get_collections().collections}
    if COLLECTION in existing:
        try:
            cfg = _client.get_collection(COLLECTION).config.params.vectors
            if isinstance(cfg, dict) and "primary" in cfg and "siglip" in cfg:
                return
        except Exception:
            pass
        # Old single-vector schema → recreate (re-upload required)
        _client.delete_collection(COLLECTION)

    _client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "primary": VectorParams(size=PRIMARY_DIM, distance=Distance.COSINE),
            "siglip":  VectorParams(size=SIGLIP_DIM,  distance=Distance.COSINE),
        },
    )


def reset_collection() -> int:
    count = _client.get_collection(COLLECTION).points_count or 0
    _client.delete_collection(COLLECTION)
    _client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "primary": VectorParams(size=PRIMARY_DIM, distance=Distance.COSINE),
            "siglip":  VectorParams(size=SIGLIP_DIM,  distance=Distance.COSINE),
        },
    )
    return count


def upsert_image(
    product_id: str,
    primary_vec: list[float],
    siglip_vec: list[float],
    status: int,
    name: str = "",
    category: str = "",
    tags: list = None,
    description: str = "",
) -> None:
    _client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=_point_id(product_id),
                vector={"primary": primary_vec, "siglip": siglip_vec},
                payload={
                    "product_id":  product_id,
                    "status":      status,
                    "name":        name,
                    "category":    category,
                    "tags":        tags or [],
                    "description": description,
                },
            )
        ],
    )


def update_image(
    product_id: str,
    primary_vec: list[float],
    siglip_vec: list[float],
    default_status: int = 1,
    name: str = "",
    category: str = "",
    tags: list = None,
    description: str = "",
) -> None:
    point_id = _point_id(product_id)
    existing = _client.retrieve(collection_name=COLLECTION, ids=[point_id], with_payload=True)
    if existing:
        old = existing[0].payload
        status      = old.get("status",      default_status)
        name        = name        or old.get("name",        "")
        category    = category    or old.get("category",    "")
        tags        = tags        or old.get("tags",        [])
        description = description or old.get("description", "")
    else:
        status = default_status
    _client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=point_id,
                vector={"primary": primary_vec, "siglip": siglip_vec},
                payload={
                    "product_id":  product_id,
                    "status":      status,
                    "name":        name,
                    "category":    category,
                    "tags":        tags or [],
                    "description": description,
                },
            )
        ],
    )


def update_status(product_id: str, status: int) -> bool:
    point_id = _point_id(product_id)
    if not _client.retrieve(collection_name=COLLECTION, ids=[point_id]):
        return False
    _client.set_payload(
        collection_name=COLLECTION,
        payload={"status": status},
        points=[point_id],
    )
    return True


def search_similar(
    primary_vec: list[float],
    siglip_vec: list[float],
    top_k: int = 20,
    score_threshold: float = 0.20,
    candidate_k: int = 100,
) -> list[dict]:
    """Two-stage search: ViT-H-14-378 HNSW retrieval → SigLIP re-rank.

    Stage 1: fast COSINE search with primary vector (top candidate_k results).
    Stage 2: recompute score as 0.55 * primary + 0.45 * siglip for each candidate.
    Returns top_k results above score_threshold sorted by final score.
    """
    response = _client.query_points(
        collection_name=COLLECTION,
        using="primary",
        query=primary_vec,
        query_filter=Filter(
            must=[FieldCondition(key="status", match=MatchValue(value=1))]
        ),
        limit=max(candidate_k, top_k),
        with_payload=True,
        with_vectors=["siglip"],
    )

    q_sig = np.array(siglip_vec, dtype="float32")
    q_sig /= np.linalg.norm(q_sig) + 1e-8

    seen, candidates = set(), []
    for h in response.points:
        pid = h.payload["product_id"]
        if pid in seen:
            continue
        seen.add(pid)

        primary_score = float(h.score)
        stored_sig    = (h.vector or {}).get("siglip")
        if stored_sig:
            sv = np.array(stored_sig, dtype="float32")
            sv /= np.linalg.norm(sv) + 1e-8
            siglip_score = float(np.dot(q_sig, sv))
        else:
            siglip_score = primary_score

        final = PRIMARY_W * primary_score + SIGLIP_W * siglip_score
        candidates.append({
            "product_id":    pid,
            "score":         round(final, 6),
            "primary_score": round(primary_score, 6),
            "siglip_score":  round(siglip_score, 6),
            "name":          h.payload.get("name", ""),
            "category":      h.payload.get("category", ""),
            "tags":          h.payload.get("tags", []),
            "description":   h.payload.get("description", ""),
        })

    return [
        r for r in sorted(candidates, key=lambda x: x["score"], reverse=True)
        if r["score"] >= score_threshold
    ][:top_k]
