import clip
import io

import numpy as np
import torch
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None
_preprocess = None


def _load():
    global _model, _preprocess
    if _model is None:
        _model, _preprocess = clip.load("ViT-B/32", device=DEVICE)
        _model.eval()


def encode_image(image_bytes: bytes) -> list[float]:
    _load()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = _model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
    return features[0].tolist()


def encode_text(text: str) -> list[float]:
    """Encode a text query into the same 512D space as images."""
    _load()
    tokens = clip.tokenize([text]).to(DEVICE)
    with torch.no_grad():
        features = _model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    return features[0].tolist()


def encode_images_mean(images_bytes: list[bytes]) -> list[float]:
    """Encode multiple images and return their mean vector (re-normalized).

    Averaging vectors from different angles/shots of the same product
    produces a more robust representation than any single image alone.
    """
    _load()
    tensors = [
        _preprocess(Image.open(io.BytesIO(b)).convert("RGB"))
        for b in images_bytes
    ]
    batch = torch.stack(tensors).to(DEVICE)
    with torch.no_grad():
        features = _model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)
    vectors = features.cpu().numpy()          # (N, 512)
    mean_vec = np.mean(vectors, axis=0)       # (512,)
    mean_vec = mean_vec / np.linalg.norm(mean_vec)  # re-normalize after averaging
    return mean_vec.tolist()
