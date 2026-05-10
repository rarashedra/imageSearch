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
    clean = remove_bg(image_bytes)
    rgba = Image.open(io.BytesIO(clean)).convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (128, 128, 128, 255))
    bg.paste(rgba, mask=rgba.split()[3])
    return bg.convert("RGB")


def _to_image(image_bytes: bytes) -> Image.Image:
    image = _strip_background(image_bytes)
    return image.resize((224, 224), Image.LANCZOS)


def _image_features(tensor) -> np.ndarray:
    with torch.no_grad():
        features = _model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().astype("float32")


def encode_image(image_bytes: bytes) -> list[float]:
    _load()
    tensor = _preprocess(_to_image(image_bytes)).unsqueeze(0).to(DEVICE)
    return _image_features(tensor)[0].tolist()


def encode_text(text: str) -> list[float]:
    _load()
    tokens = clip.tokenize([text]).to(DEVICE)
    with torch.no_grad():
        features = _model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().astype("float32")[0].tolist()


def encode_images_mean(images_bytes: list[bytes]) -> list[float]:
    _load()
    tensors = [_preprocess(_to_image(b)) for b in images_bytes]
    batch = torch.stack(tensors).to(DEVICE)
    with torch.no_grad():
        features = _model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)
    vectors = features.cpu().numpy().astype("float32")
    mean_vec = np.mean(vectors, axis=0)
    mean_vec = mean_vec / np.linalg.norm(mean_vec)
    return mean_vec.tolist()
