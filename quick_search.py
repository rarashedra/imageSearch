"""
quick_search.py — Fast real-time search using ViT-L/14 + local FAISS index.
Loads in ~3 seconds. Run sync_images.py first to build the index.

Usage:
    python quick_search.py                   # searches all images in query/
    python quick_search.py path/to/img.jpg   # search a specific image
    python quick_search.py --top-k 10 --threshold 0.10 path/to/img.jpg
"""

import io
import os
import sys
import json
import argparse
import logging
import warnings

import numpy as np
import faiss
import torch
import clip
from PIL import Image
from rembg import remove, new_session

warnings.filterwarnings("ignore")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

INDEX_DIR        = "index"
QUERY_DIR        = "query"
QUICK_INDEX_FILE = os.path.join(INDEX_DIR, "quick.faiss")
QUICK_META_FILE  = os.path.join(INDEX_DIR, "quick_meta.json")
BG_CACHE_DIR     = os.path.join(QUERY_DIR, "bg_removed")

TOP_K            = 5
SIM_THRESHOLD    = 0.15
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")

# ── Verify index exists ──────────────────────────────────────────────────────
for _f in [QUICK_INDEX_FILE, QUICK_META_FILE]:
    if not os.path.exists(_f):
        print(f"Quick index not found: {_f}")
        print("Run sync_images.py first.")
        sys.exit(1)

# ── Load model ───────────────────────────────────────────────────────────────
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Loading ViT-L/14 ({device}) ...", end=" ", flush=True)
model, preprocess = clip.load("ViT-L/14", device=device)
model.eval()
print("ready.")

print("Loading rembg ...", end=" ", flush=True)
bg_session = new_session("isnet-general-use")
os.makedirs(BG_CACHE_DIR, exist_ok=True)
print("ready.")

# ── Load index ───────────────────────────────────────────────────────────────
index = faiss.read_index(QUICK_INDEX_FILE)
with open(QUICK_META_FILE) as f:
    quick_data = json.load(f)

valid_files = quick_data["files"]
meta_map    = quick_data["meta"]
print(f"Index loaded — {len(valid_files)} images.\n")


# ── Helpers ──────────────────────────────────────────────────────────────────

def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    return vec / (norm + 1e-8)


def get_bg_removed(image_path: str) -> Image.Image:
    filename    = os.path.basename(image_path)
    cached_path = os.path.join(BG_CACHE_DIR, os.path.splitext(filename)[0] + ".png")
    if os.path.exists(cached_path):
        img = Image.open(cached_path).convert("RGBA")
    else:
        with open(image_path, "rb") as f:
            raw = f.read()
        result = remove(raw, session=bg_session)
        img    = Image.open(io.BytesIO(result)).convert("RGBA")
        img.save(cached_path)
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    return bg


def get_query_embedding(image_path: str) -> np.ndarray:
    img    = get_bg_removed(image_path)
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(tensor)
    return normalize(emb.cpu().numpy().astype("float32"))


# ── Search ───────────────────────────────────────────────────────────────────

def search(query_path: str, top_k: int = TOP_K, threshold: float = SIM_THRESHOLD) -> None:
    query_file = os.path.basename(query_path)
    query_emb  = get_query_embedding(query_path)
    D, I       = index.search(query_emb, min(top_k * 3, len(valid_files)))

    results = []
    for rank, idx in enumerate(I[0]):
        score = float(D[0][rank])
        if score < threshold:
            continue
        filename = valid_files[idx]
        meta     = meta_map.get(filename, {})
        results.append({
            "file":       filename,
            "name":       meta.get("name", filename),
            "description": meta.get("description", ""),
            "category":   meta.get("category", ""),
            "tags":       meta.get("tags", []),
            "similarity": round(score * 100, 1),
        })

    results = results[:top_k]

    print(f"Query : {query_file}")
    print("─" * 60)
    if not results:
        print("  No results above threshold.")
    else:
        for rank, r in enumerate(results):
            bar = "█" * int(r["similarity"] / 4)
            print(f"  {rank+1}. {r['file']:<28}  {r['name']}")
            print(f"     [{r['category']:<22}]  similarity={r['similarity']}%  {bar}")
    print()


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fast image search using ViT-L/14 + FAISS")
    parser.add_argument("image",       nargs="?",            help="Path to query image")
    parser.add_argument("--top-k",     type=int,   default=TOP_K,         help="Max results")
    parser.add_argument("--threshold", type=float, default=SIM_THRESHOLD, help="Min similarity (0–1 scale internally)")
    args = parser.parse_args()

    threshold = args.threshold / 100 if args.threshold > 1 else args.threshold

    if args.image:
        if not os.path.exists(args.image):
            print(f"File not found: {args.image}")
            sys.exit(1)
        search(args.image, top_k=args.top_k, threshold=threshold)
    else:
        if not os.path.isdir(QUERY_DIR):
            print(f"Query dir '{QUERY_DIR}/' not found. Pass an image path as argument.")
            sys.exit(1)
        query_files = sorted([
            f for f in os.listdir(QUERY_DIR)
            if os.path.isfile(os.path.join(QUERY_DIR, f))
            and f.lower().endswith(VALID_EXTENSIONS)
        ])
        if not query_files:
            print(f"No query images in '{QUERY_DIR}/'")
            sys.exit(1)
        for qf in query_files:
            search(os.path.join(QUERY_DIR, qf), top_k=args.top_k, threshold=threshold)


if __name__ == "__main__":
    main()
