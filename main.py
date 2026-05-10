from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, field_validator

import clip_encoder
import qdrant_service

templates = Jinja2Templates(directory="templates")


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
):
    if status not in (0, 1):
        raise HTTPException(status_code=422, detail="status must be 0 or 1")
    image_bytes = await image.read()
    vector = clip_encoder.encode_image(image_bytes)
    del image_bytes
    qdrant_service.upsert_image(product_id, vector, status)
    return {
        "message": "Image uploaded successfully",
        "product_id": product_id,
        "status": status,
    }


@app.patch("/image/{product_id}", summary="Update product image")
async def update_image(
    product_id: str,
    image: UploadFile = File(...),
    status: int = Form(None),
):
    if status is not None and status not in (0, 1):
        raise HTTPException(status_code=422, detail="status must be 0 or 1")
    image_bytes = await image.read()
    vector = clip_encoder.encode_image(image_bytes)
    del image_bytes
    qdrant_service.update_image(
        product_id,
        vector,
        default_status=status if status is not None else 1,
    )
    return {
        "success": True,
        "product_id": product_id,
        "message": "Image updated successfully",
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
    top_k: int = 20,
):
    image_bytes = await image.read()
    vector = clip_encoder.encode_image(image_bytes)
    del image_bytes

    raw = qdrant_service.search_similar(vector, top_k=top_k, score_threshold=threshold)
    results = [{"product_id": r["product_id"], "score": r["score"]} for r in raw]

    return {
        "threshold": threshold,
        "count": len(results),
        "results": results,
    }
