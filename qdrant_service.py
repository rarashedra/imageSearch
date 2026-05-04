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


def upsert_image(product_id: str, vector: list[float], status: int) -> None:
    _client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=_point_id(product_id),
                vector=vector,
                payload={"product_id": product_id, "status": status},
            )
        ],
    )


def update_image(product_id: str, vector: list[float], default_status: int = 1) -> None:
    point_id = _point_id(product_id)
    found = _client.retrieve(collection_name=COLLECTION, ids=[point_id], with_payload=True)
    current_status = found[0].payload.get("status", default_status) if found else default_status
    _client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=point_id,
                vector=vector,
                payload={"product_id": product_id, "status": current_status},
            )
        ],
    )


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


def search_similar(vector: list[float], top_k: int = 5, score_threshold: float = 0.7) -> list[dict]:
    response = _client.query_points(
        collection_name=COLLECTION,
        query=vector,
        query_filter=Filter(
            must=[FieldCondition(key="status", match=MatchValue(value=1))]
        ),
        limit=top_k,
        score_threshold=score_threshold,
        with_payload=True,
    )
    return [{"product_id": h.payload["product_id"], "score": round(h.score, 6)} for h in response.points]
