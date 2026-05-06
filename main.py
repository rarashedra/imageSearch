from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, field_validator

import clip_encoder
import qdrant_service

templates = Jinja2Templates(directory="templates")

# Broad bootstrap labels used only when Qdrant has no labels yet.
# Once products are uploaded and labels are stored, Qdrant's own labels
# take over and this list is no longer used.
_BOOTSTRAP_LABELS = [
    "shirt", "t-shirt", "dress", "pants", "jeans", "jacket", "coat",
    "shoes", "sneakers", "boots", "sandals", "bag", "handbag", "backpack",
    "watch", "sunglasses", "headphones", "phone", "laptop", "tablet",
    "guitar", "camera", "book", "chair", "sofa", "table",
    "apple", "mango", "banana", "orange", "fruit", "vegetable",
]


def _get_labels() -> list[str]:
    """Return stored Qdrant labels, falling back to bootstrap when DB is empty."""
    stored = qdrant_service.get_labels()
    return stored if stored else _BOOTSTRAP_LABELS


@asynccontextmanager
async def lifespan(app: FastAPI):
    qdrant_service.ensure_collection()
    yield


app = FastAPI(title="Image Search Service", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root(request: Request):
    issues = []

    try:
        info = qdrant_service.health_check()
        qdrant_status = "ok"
        qdrant_detail = (
            f"connected &nbsp;·&nbsp; collection <b>{qdrant_service.COLLECTION}</b>"
            f" &nbsp;·&nbsp; <b>{info['vectors']}</b> vectors indexed"
        )
    except Exception as e:
        qdrant_status = "error"
        qdrant_detail = str(e)
        issues.append(f"Qdrant: {e}")

    clip_status = "ok" if clip_encoder._model is not None else "idle"
    clip_detail = (
        "model loaded in memory"
        if clip_encoder._model is not None
        else "model not yet loaded (loads on first request)"
    )

    return templates.TemplateResponse(request, "index.html", {
        "status_color": "#22c55e" if not issues else "#ef4444",
        "overall": "All systems operational" if not issues else f"{len(issues)} issue(s) detected",
        "issues": issues,
        "qdrant_status": qdrant_status,
        "qdrant_detail": qdrant_detail,
        "clip_status": clip_status,
        "clip_detail": clip_detail,
    })


@app.get("/guide", response_class=HTMLResponse, include_in_schema=False)
async def guide(request: Request):
    return templates.TemplateResponse(request, "guide.html")


class StatusBody(BaseModel):
    status: int

    @field_validator("status")
    @classmethod
    def must_be_binary(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError("status must be 0 or 1")
        return v


@app.post("/upload", summary="Upload a product image")
async def upload(
    product_id: str = Form(...),
    status: int = Form(...),
    image: UploadFile = File(...),
    label: str = Form(None),
):
    if status not in (0, 1):
        raise HTTPException(status_code=422, detail="status must be 0 or 1")
    image_bytes = await image.read()
    if label:
        vector = clip_encoder.encode_image(image_bytes)
        detected = label.strip().lower()
        confidence = 1.0
    else:
        vector, detected, confidence = clip_encoder.encode_and_detect(image_bytes, _get_labels())
    qdrant_service.upsert_image(product_id, vector, status, detected)
    return {
        "message": "Image uploaded successfully",
        "product_id": product_id,
        "detected_label": detected or "unknown",
        "confidence": confidence,
    }


@app.patch("/image/{product_id}", summary="Update product image")
async def update_image(
    product_id: str,
    image: UploadFile = File(...),
    status: int = Form(None),
    label: str = Form(None),
):
    if status is not None and status not in (0, 1):
        raise HTTPException(status_code=422, detail="status must be 0 or 1")
    image_bytes = await image.read()
    if label:
        vector = clip_encoder.encode_image(image_bytes)
        detected = label.strip().lower()
        confidence = 1.0
    else:
        vector, detected, confidence = clip_encoder.encode_and_detect(image_bytes, _get_labels())
    qdrant_service.update_image(
        product_id,
        vector,
        default_status=status if status is not None else 1,
        label=detected,
    )
    return {
        "message": "Image updated successfully",
        "product_id": product_id,
        "detected_label": detected or "unknown",
        "confidence": confidence,
    }


@app.delete("/collection/reset", summary="Clear all vectors and reset collection")
async def reset_collection():
    deleted = qdrant_service.reset_collection()
    return {"message": "Collection reset successfully", "deleted_vectors": deleted}


@app.patch("/status/{product_id}", summary="Update product status")
async def update_status(product_id: str, body: StatusBody):
    found = qdrant_service.update_status(product_id, body.status)
    if not found:
        raise HTTPException(status_code=404, detail=f"Product '{product_id}' not found")
    return {"message": "Status updated successfully", "product_id": product_id, "status": body.status}


@app.post("/search", summary="Search for similar active products")
async def search(
    image: UploadFile = File(...),
    threshold: float = 0.75,
):
    image_bytes = await image.read()
    candidates = _get_labels()
    vector, predicted_label, accuracy = clip_encoder.encode_and_detect(image_bytes, candidates)
    stored_labels = qdrant_service.get_labels()
    if stored_labels:
        raw = qdrant_service.search_similar(
            vector, label=predicted_label, top_k=20, score_threshold=threshold, with_vectors=True
        )
    else:
        raw = qdrant_service.search_similar(
            vector, top_k=20, score_threshold=threshold, with_vectors=True
        )
    results = clip_encoder.verify_results(raw, query_vector=vector, match_threshold=threshold)
    return {
        "predicted_label": predicted_label,
        "accuracy": accuracy,
        "results": results,
    }
