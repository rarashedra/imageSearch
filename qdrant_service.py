import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

COLLECTION = "image_search"
VECTOR_SIZE = 512

_client = QdrantClient(host="localhost", port=6333)


def _point_id(product_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, product_id))


def health_check() -> dict:
    info = _client.get_collection(COLLECTION)
    return {"vectors": info.points_count}


def ensure_collection() -> None:
    existing = {c.name for c in _client.get_collections().collections}
    if COLLECTION not in existing:
        _client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def reset_collection() -> int:
    """Drop and recreate the collection. Returns count of deleted vectors."""
    count = _client.get_collection(COLLECTION).points_count or 0
    _client.delete_collection(COLLECTION)
    _client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    return count


def upsert_image(product_id: str, vector: list[float], status: int, label: str = "") -> None:
    _client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=_point_id(product_id),
                vector=vector,
                payload={"product_id": product_id, "status": status, "label": label},
            )
        ],
    )


def update_image(product_id: str, vector: list[float], default_status: int = 1, label: str = "") -> bool:
    point_id = _point_id(product_id)
    found = _client.retrieve(collection_name=COLLECTION, ids=[point_id], with_payload=True)
    if not found:
        # Product not in Qdrant — insert fresh
        current_status = default_status
        current_label = label
    else:
        # Preserve existing status and label unless new values provided
        current_status = found[0].payload.get("status", default_status)
        current_label = label or found[0].payload.get("label", "")
    _client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=point_id,
                vector=vector,
                payload={"product_id": product_id, "status": current_status, "label": current_label},
            )
        ],
    )
    return True


def get_labels() -> list[str]:
    """Return all unique labels currently stored in the collection."""
    labels = set()
    offset = None
    while True:
        points, offset = _client.scroll(
            collection_name=COLLECTION,
            with_payload=["label"],
            limit=100,
            offset=offset,
        )
        for p in points:
            lbl = p.payload.get("label", "")
            if lbl:
                labels.add(lbl)
        if offset is None:
            break
    return sorted(labels)


def update_status(product_id: str, status: int) -> bool:
    point_id = _point_id(product_id)
    found = _client.retrieve(collection_name=COLLECTION, ids=[point_id])
    if not found:
        return False
    _client.set_payload(
        collection_name=COLLECTION,
        payload={"status": status},
        points=[point_id],
    )
    return True


def search_similar(
    vector: list[float],
    label: str = "",
    top_k: int = 20,
    score_threshold: float = 0.75,
    with_vectors: bool = False,
) -> list[dict]:
    must = [FieldCondition(key="status", match=MatchValue(value=1))]
    if label:
        must.append(FieldCondition(key="label", match=MatchValue(value=label)))
    response = _client.query_points(
        collection_name=COLLECTION,
        query=vector,
        query_filter=Filter(must=must),
        limit=top_k,
        score_threshold=score_threshold,
        with_payload=True,
        with_vectors=with_vectors,
    )
    results = []
    for h in response.points:
        item = {"product_id": h.payload["product_id"], "score": round(h.score, 6)}
        if with_vectors and h.vector is not None:
            item["vector"] = h.vector
        results.append(item)
    return results
