import io

import numpy as np
import open_clip
import torch
from PIL import Image

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

PRIMARY_DIM = 1024
SIGLIP_DIM  = 1152
IMG_WEIGHT  = 0.70
TEXT_WEIGHT = 0.30
PRIMARY_W   = 0.55
SIGLIP_W    = 0.45

PROMPT_TEMPLATES = [
    "a product photo of {}",
    "a photo of a {}",
    "a high quality e-commerce image of {}",
    "an online store product photo of {}",
    "{}",
]

_primary_model      = None
_primary_preprocess = None
_primary_tokenizer  = None
_siglip_model       = None
_siglip_preprocess  = None
_siglip_tokenizer   = None
_florence_model     = None
_florence_processor = None



def _load_florence():
    global _florence_model, _florence_processor
    if _florence_model is None:
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM as _AMCL
            _florence_processor = AutoProcessor.from_pretrained(
                "microsoft/Florence-2-base", trust_remote_code=True
            )
            _florence_model = _AMCL.from_pretrained(
                "microsoft/Florence-2-base",
                trust_remote_code=True,
                torch_dtype=torch.float16 if DEVICE != "cpu" else torch.float32,
            ).to(DEVICE).eval()
        except Exception as e:
            print(f"[florence] Failed to load: {e}")
    return _florence_model, _florence_processor


def _florence_run(task: str, img: Image.Image):
    dtype = torch.float16 if DEVICE != "cpu" else torch.float32
    inputs = _florence_processor(text=task, images=img, return_tensors="pt")
    inputs = {
        "input_ids":    inputs["input_ids"].to(DEVICE),
        "pixel_values": inputs["pixel_values"].to(DEVICE, dtype),
    }
    with torch.no_grad():
        ids = _florence_model.generate(
            **inputs, max_new_tokens=256, do_sample=False, num_beams=1
        )
    raw = _florence_processor.batch_decode(ids, skip_special_tokens=False)[0]
    return _florence_processor.post_process_generation(
        raw, task=task, image_size=(img.width, img.height)
    ).get(task, "")


def _load_primary():
    global _primary_model, _primary_preprocess, _primary_tokenizer
    if _primary_model is None:
        _primary_model, _, _primary_preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14-378-quickgelu", pretrained="dfn5b", device=DEVICE
        )
        _primary_model.eval()
        _primary_tokenizer = open_clip.get_tokenizer("ViT-H-14-378-quickgelu")


def _load_siglip():
    global _siglip_model, _siglip_preprocess, _siglip_tokenizer
    if _siglip_model is None:
        _siglip_model, _, _siglip_preprocess = open_clip.create_model_and_transforms(
            "ViT-SO400M-14-SigLIP-384", pretrained="webli", device=DEVICE
        )
        _siglip_model.eval()
        _siglip_tokenizer = open_clip.get_tokenizer("ViT-SO400M-14-SigLIP-384")





def _crop_tensors(img: Image.Image, preprocess) -> torch.Tensor:
    """5-crop TTA: center + 4 corners."""
    w, h  = img.size
    side  = min(w, h)
    lft   = (w - side) // 2
    top   = (h - side) // 2
    crops = [img.crop((lft, top, lft + side, top + side))]
    for cx, cy in [(0, 0), (w - side, 0), (0, h - side), (w - side, h - side)]:
        crops.append(img.crop((cx, cy, cx + side, cy + side)))
    return torch.stack([preprocess(c) for c in crops]).to(DEVICE)


def _norm(vec: np.ndarray) -> np.ndarray:
    return vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-8)


def _primary_img_emb(img: Image.Image, tta: bool = True) -> np.ndarray:
    if tta:
        batch = _crop_tensors(img, _primary_preprocess)
        with torch.no_grad():
            embs = _primary_model.encode_image(batch)
        return embs.mean(dim=0, keepdim=True).cpu().numpy().astype("float32")
    else:
        tensor = _primary_preprocess(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = _primary_model.encode_image(tensor)
        return emb.cpu().numpy().astype("float32")


def _primary_txt_emb(text: str) -> np.ndarray:
    tokens = _primary_tokenizer([t.format(text) for t in PROMPT_TEMPLATES]).to(DEVICE)
    with torch.no_grad():
        embs = _primary_model.encode_text(tokens)
    return embs.mean(dim=0, keepdim=True).cpu().numpy().astype("float32")


def _siglip_img_emb(img: Image.Image, tta: bool = True) -> np.ndarray:
    if tta:
        batch = _crop_tensors(img, _siglip_preprocess)
        with torch.no_grad():
            embs = _siglip_model.encode_image(batch)
        return embs.mean(dim=0, keepdim=True).cpu().numpy().astype("float32")
    else:
        tensor = _siglip_preprocess(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = _siglip_model.encode_image(tensor)
        return emb.cpu().numpy().astype("float32")


def _siglip_txt_emb(text: str) -> np.ndarray:
    tokens = _siglip_tokenizer([t.format(text) for t in PROMPT_TEMPLATES]).to(DEVICE)
    with torch.no_grad():
        embs = _siglip_model.encode_text(tokens)
    return embs.mean(dim=0, keepdim=True).cpu().numpy().astype("float32")


def encode_image_both(
    image_bytes: bytes, text: str = "", tta: bool = True
) -> tuple[list[float], list[float]]:
    """Encode with both models. Returns (primary 1024-dim, siglip 1152-dim).

    tta=True  for upload/indexing  — 5-crop average, better quality.
    tta=False for search queries   — single crop, ~5x faster.
    """
    _load_primary()
    _load_siglip()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    p_img = _norm(_primary_img_emb(img, tta=tta))
    s_img = _norm(_siglip_img_emb(img, tta=tta))

    if text:
        p_fused = _norm(IMG_WEIGHT * p_img + TEXT_WEIGHT * _norm(_primary_txt_emb(text)))
        s_fused = _norm(IMG_WEIGHT * s_img + TEXT_WEIGHT * _norm(_siglip_txt_emb(text)))
    else:
        p_fused, s_fused = p_img, s_img

    return p_fused[0].tolist(), s_fused[0].tolist()


def analyze_image(image_bytes: bytes) -> dict:
    """Auto-detect category, tags, and description using Florence-2.

    Runs real image understanding — no predefined lists, fully open-ended.
    Returns {"category": str, "tags": list[str], "description": str}.
    Downloads microsoft/Florence-2-base (~900 MB) on first call.
    """
    model, processor = _load_florence()
    if model is None:
        return {"category": "", "tags": [], "description": ""}

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Short caption → use as category
    category = _florence_run("<CAPTION>", img).strip()

    # Detailed caption → description
    description = _florence_run("<DETAILED_CAPTION>", img).strip()

    # Object detection → extract detected object labels as tags
    od = _florence_run("<OD>", img)
    if isinstance(od, dict):
        raw_labels = od.get("labels", [])
    else:
        raw_labels = []

    # Deduplicate and clean labels
    seen, tags = set(), []
    for label in raw_labels:
        label = label.lower().strip()
        if label and label not in seen:
            seen.add(label)
            tags.append(label)
        if len(tags) == 7:
            break

    return {"category": category, "tags": tags, "description": description}


def encode_images_mean(
    images_bytes: list[bytes],
) -> tuple[list[float], list[float]]:
    """Mean embedding across multiple images using both models."""
    _load_primary()
    _load_siglip()
    p_vecs, s_vecs = [], []
    for b in images_bytes:
        img = Image.open(io.BytesIO(b)).convert("RGB")
        p_vecs.append(_norm(_primary_img_emb(img)))
        s_vecs.append(_norm(_siglip_img_emb(img)))
    p = _norm(np.mean(np.vstack(p_vecs), axis=0, keepdims=True))
    s = _norm(np.mean(np.vstack(s_vecs), axis=0, keepdims=True))
    return p[0].tolist(), s[0].tolist()
