import clip
import io

import numpy as np
import torch
from PIL import Image
from rembg import remove as remove_bg

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

_model = None
_preprocess = None


def _load():
    global _model, _preprocess
    if _model is None:
        _model, _preprocess = clip.load("ViT-B/16", device=DEVICE)
        _model.eval()


def _strip_background(image_bytes: bytes) -> Image.Image:
    """Remove background and paste object on neutral gray so CLIP focuses on the object."""
    clean = remove_bg(image_bytes)
    rgba = Image.open(io.BytesIO(clean)).convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (128, 128, 128, 255))
    bg.paste(rgba, mask=rgba.split()[3])
    return bg.convert("RGB")


def _to_image(image_bytes: bytes) -> Image.Image:
    """Decode bytes, strip background, resize to 224×224 for stable CLIP input."""
    image = _strip_background(image_bytes)
    return image.resize((224, 224), Image.LANCZOS)


def _image_features(tensor) -> np.ndarray:
    """Encode image tensor → normalized float32 numpy vector."""
    with torch.no_grad():
        features = _model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().astype("float32")


def encode_image(image_bytes: bytes) -> list[float]:
    _load()
    tensor = _preprocess(_to_image(image_bytes)).unsqueeze(0).to(DEVICE)
    return _image_features(tensor)[0].tolist()


def encode_and_detect(
    image_bytes: bytes, labels: list[str]
) -> tuple[list[float], str, float]:
    """Encode image and detect label using CLIP's learned temperature (logit_scale).

    logit_scale is CLIP's trained temperature parameter — using it makes
    prediction scores sharper and more calibrated than raw dot products.
    Returns (vector, predicted_label, accuracy_percent).
    """
    _load()
    tensor = _preprocess(_to_image(image_bytes)).unsqueeze(0).to(DEVICE)
    prompts = clip.tokenize([f"a photo of a {label}" for label in labels]).to(DEVICE)

    with torch.no_grad():
        img_feat = _model.encode_image(tensor)
        txt_feat = _model.encode_text(prompts)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        # Use CLIP's learned temperature — sharpens prediction confidence
        logit_scale = _model.logit_scale.exp()
        scores = (logit_scale * img_feat @ txt_feat.T).squeeze(0).cpu().numpy().astype("float32")

    # Softmax → probability distribution over labels
    exp_scores = np.exp(scores - np.max(scores))       # subtract max for numerical stability
    probabilities = exp_scores / exp_scores.sum()

    best_idx = int(np.argmax(probabilities))
    accuracy = round(float(probabilities[best_idx]) * 100, 2)
    return img_feat.cpu().numpy().astype("float32")[0].tolist(), labels[best_idx], accuracy


def verify_results(
    results: list[dict],
    query_vector: list[float],
    match_threshold: float = 0.70,
) -> list[dict]:
    """Image-to-image similarity judge — CLIP equivalent of an LLM similarity prompt.

    Compares the query image vector against each candidate's stored vector.
    Focuses on: same product category, same shape/structure/design.
    Ignores: background, lighting, camera angle (CLIP handles this naturally).

    Returns only MATCH results sorted by confidence descending.
    """
    if not results:
        return results

    q_vec = np.array(query_vector, dtype="float32")
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-8)

    verified = []
    for item in results:
        stored_vec = item.get("vector")
        if stored_vec is not None:
            c_vec = np.array(stored_vec, dtype="float32")
            c_vec = c_vec / (np.linalg.norm(c_vec) + 1e-8)
            # Cosine similarity: query image vs candidate image
            similarity = float(np.dot(q_vec, c_vec))
        else:
            # Fall back to the Qdrant score when vector not available
            similarity = item["score"]

        # Map cosine similarity [0, 1] → confidence [0, 100]
        confidence = round(similarity * 100, 1)
        is_match = similarity >= match_threshold

        if is_match:
            verified.append({
                "product_id": item["product_id"],
                "score":      round(similarity, 6),
                "match":      "MATCH",
                "confidence": confidence,
            })

    # Sort highest confidence first
    verified.sort(key=lambda x: x["confidence"], reverse=True)
    return verified


def encode_text(text: str) -> list[float]:
    _load()
    tokens = clip.tokenize([text]).to(DEVICE)
    with torch.no_grad():
        features = _model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().astype("float32")[0].tolist()


def encode_images_mean(images_bytes: list[bytes]) -> list[float]:
    _load()
    tensors = [
        _preprocess(_to_image(b))
        for b in images_bytes
    ]
    batch = torch.stack(tensors).to(DEVICE)
    with torch.no_grad():
        features = _model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)
    vectors = features.cpu().numpy().astype("float32")
    mean_vec = np.mean(vectors, axis=0)
    mean_vec = mean_vec / np.linalg.norm(mean_vec)
    return mean_vec.tolist()
