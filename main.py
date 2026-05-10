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
    clip_encoder._load_primary()
    clip_encoder._load_siglip()
    clip_encoder._load_florence()
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

    primary_loaded = clip_encoder._primary_model is not None
    siglip_loaded  = clip_encoder._siglip_model is not None
    clip_status    = "ok" if primary_loaded else "idle"
    clip_detail    = (
        f"primary={'loaded' if primary_loaded else 'idle'}  "
        f"siglip={'loaded' if siglip_loaded else 'idle'}"
    )

    return templates.TemplateResponse(request, "index.html", {
        "status_color": "#22c55e" if not issues else "#ef4444",
        "overall":      "All systems operational" if not issues else f"{len(issues)} issue(s) detected",
        "issues":       issues,
        "qdrant_status": qdrant_status,
        "qdrant_detail": qdrant_detail,
        "clip_status":   clip_status,
        "clip_detail":   clip_detail,
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
    product_id:  str  = Form(...),
    status:      int  = Form(...),
    image:       UploadFile = File(...),
    name:        str  = Form(None),
    description: str  = Form(None),
    category:    str  = Form(None),
    tags:        str  = Form(None),
):
    if status not in (0, 1):
        raise HTTPException(status_code=422, detail="status must be 0 or 1")
    image_bytes = await image.read()

    # Only call moondream auto-analysis when the caller hasn't provided metadata
    has_metadata = any([name, description, category, tags])
    if not has_metadata:
        analyzed        = clip_encoder.analyze_image(image_bytes)
        final_category  = analyzed["category"]
        final_tags      = analyzed["tags"]
        final_desc      = analyzed["description"]
    else:
        final_category  = category    or ""
        final_tags      = [t.strip() for t in tags.split(",")] if tags else []
        final_desc      = description or ""

    text = " ".join(filter(None, [name, final_desc, final_category, " ".join(final_tags)])).strip()
    primary_vec, siglip_vec = clip_encoder.encode_image_both(image_bytes, text=text, tta=False)
    del image_bytes

    qdrant_service.upsert_image(
        product_id, primary_vec, siglip_vec, status,
        name=name or "",
        category=final_category,
        tags=final_tags,
        description=final_desc,
    )
    return {
        "message":    "Image uploaded successfully",
        "product_id": product_id,
        "status":     status,
        "name":       name or "",
        "category":   final_category,
        "tags":       final_tags,
        "description": final_desc,
    }


@app.patch("/image/{product_id}", summary="Update product image")
async def update_image(
    product_id:  str  = ...,
    image:       UploadFile = File(...),
    status:      int  = Form(None),
    name:        str  = Form(None),
    description: str  = Form(None),
    category:    str  = Form(None),
    tags:        str  = Form(None),
):
    if status is not None and status not in (0, 1):
        raise HTTPException(status_code=422, detail="status must be 0 or 1")
    image_bytes = await image.read()

    # Only call moondream auto-analysis when the caller hasn't provided metadata
    has_metadata = any([name, description, category, tags])
    if not has_metadata:
        analyzed        = clip_encoder.analyze_image(image_bytes)
        final_category  = analyzed["category"]
        final_tags      = analyzed["tags"]
        final_desc      = analyzed["description"]
    else:
        final_category  = category    or ""
        final_tags      = [t.strip() for t in tags.split(",")] if tags else []
        final_desc      = description or ""

    text = " ".join(filter(None, [name, final_desc, final_category, " ".join(final_tags)])).strip()
    primary_vec, siglip_vec = clip_encoder.encode_image_both(image_bytes, text=text, tta=False)
    del image_bytes

    qdrant_service.update_image(
        product_id, primary_vec, siglip_vec,
        default_status=status if status is not None else 1,
        name=name or "",
        category=final_category,
        tags=final_tags,
        description=final_desc,
    )
    return {
        "success":     True,
        "product_id":  product_id,
        "message":     "Image updated successfully",
        "name":        name or "",
        "category":    final_category,
        "tags":        final_tags,
        "description": final_desc,
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
    image:     UploadFile = File(...),
    threshold: float = 0.20,
):
    image_bytes = await image.read()
    primary_vec, siglip_vec = clip_encoder.encode_image_both(image_bytes, tta=False)
    del image_bytes

    results = qdrant_service.search_similar(
        primary_vec, siglip_vec, top_k=20, score_threshold=threshold
    )
    return {
        "threshold": threshold,
        "count":     len(results),
        "results":   results,
    }
