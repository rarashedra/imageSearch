"""
search_images.py — Offline dual-model search against the pre-built local FAISS index.
No Qdrant required. Run sync_images.py first to build the index.

Pipeline:
  1. Load primary.faiss (ViT-H-14-378, dim=1024) + siglip.npy + files.json + files_meta.json
  2. Encode query with both models (bg-removal + 5-crop TTA + prompt ensemble)
  3. Stage 1 — HNSW retrieval with primary model (top CANDIDATE_K)
  4. Stage 2 — re-rank: 0.55 * primary + 0.45 * siglip
  5. Filter < threshold, return top TOP_K

Usage:
    python search_images.py path/to/query.jpg
    python search_images.py                       # searches all images in query/
    python search_images.py --top-k 10 --threshold 0.15 path/to/query.jpg
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
from PIL import Image
from rembg import remove, new_session

warnings.filterwarnings("ignore", message=".*unauthenticated.*")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clip_encoder

INDEX_DIR          = "index"
QUERY_DIR          = "query"
PRIMARY_INDEX_FILE = os.path.join(INDEX_DIR, "primary.faiss")
SIGLIP_MATRIX_FILE = os.path.join(INDEX_DIR, "siglip.npy")
FILES_LIST_FILE    = os.path.join(INDEX_DIR, "files.json")
FILES_META_FILE    = os.path.join(INDEX_DIR, "files_meta.json")
BG_CACHE_DIR       = os.path.join(QUERY_DIR, "bg_removed")

TOP_K         = 5
CANDIDATE_K   = 50
SIM_THRESHOLD = 0.20
PRIMARY_W     = 0.55
SIGLIP_W      = 0.45

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")


# ── Verify index files exist ──────────────────────────────────────────────────
for _f in [PRIMARY_INDEX_FILE, SIGLIP_MATRIX_FILE, FILES_LIST_FILE, FILES_META_FILE]:
    if not os.path.exists(_f):
        print(f"Index file not found: {_f}")
        print("Run sync_images.py first.")
        sys.exit(1)

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading models ...")
clip_encoder._load_primary()
clip_encoder._load_siglip()

print("Loading rembg ...", end=" ", flush=True)
bg_session = new_session("isnet-general-use")
os.makedirs(BG_CACHE_DIR, exist_ok=True)
print("ready.")

# ── Load index ────────────────────────────────────────────────────────────────
print("Loading FAISS index ...", end=" ", flush=True)
primary_index = faiss.read_index(PRIMARY_INDEX_FILE)
primary_index.hnsw.efSearch = 128
siglip_matrix = np.load(SIGLIP_MATRIX_FILE)
with open(FILES_LIST_FILE) as f:
    valid_files = json.load(f)
with open(FILES_META_FILE) as f:
    files_meta = json.load(f)
print(f"ready — {len(valid_files)} images indexed.\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def get_query_embeddings(image_path: str, meta: dict) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    try:
        img  = get_bg_removed(image_path)
        tags = " ".join(meta.get("tags", []))
        text = " ".join(filter(None, [
            meta.get("name", ""),
            meta.get("description", ""),
            meta.get("category", ""),
            tags,
        ])).strip() or os.path.basename(image_path)

        p_img = normalize(clip_encoder._primary_img_emb(img))
        p_txt = normalize(clip_encoder._primary_txt_emb(text))
        p_fused = normalize(clip_encoder.IMG_WEIGHT * p_img + clip_encoder.TEXT_WEIGHT * p_txt)

        s_img = normalize(clip_encoder._siglip_img_emb(img))
        s_txt = normalize(clip_encoder._siglip_txt_emb(text))
        s_fused = normalize(clip_encoder.IMG_WEIGHT * s_img + clip_encoder.TEXT_WEIGHT * s_txt)

        return p_fused, s_fused
    except Exception as e:
        print(f"  [encode error] {image_path}: {e}")
        return None, None


# ── Search ────────────────────────────────────────────────────────────────────

def search(query_path: str, top_k: int = TOP_K, threshold: float = SIM_THRESHOLD) -> None:
    query_file = os.path.basename(query_path)
    meta       = files_meta.get(query_file, {})

    print(f"Query : {query_file}")
    print("─" * 65)

    q_primary, q_siglip = get_query_embeddings(query_path, meta)
    if q_primary is None:
        print("  Failed to encode query image.")
        print()
        return

    candidate_k = min(CANDIDATE_K, len(valid_files))
    D, I        = primary_index.search(q_primary, candidate_k)

    q_sig = q_siglip[0]

    candidates = []
    for rank, idx in enumerate(I[0]):
        primary_score = float(D[0][rank])
        sv            = normalize(siglip_matrix[idx:idx+1])[0]
        siglip_score  = float(np.dot(q_sig, sv))
        final_score   = PRIMARY_W * primary_score + SIGLIP_W * siglip_score
        if final_score < threshold:
            continue
        file_meta = files_meta.get(valid_files[idx], {})
        candidates.append({
            "file":          valid_files[idx],
            "name":          file_meta.get("name", valid_files[idx]),
            "description":   file_meta.get("description", ""),
            "category":      file_meta.get("category", ""),
            "tags":          file_meta.get("tags", []),
            "primary_score": round(primary_score * 100, 1),
            "siglip_score":  round(siglip_score  * 100, 1),
            "similarity":    round(final_score   * 100, 2),
        })

    results = sorted(candidates, key=lambda x: x["similarity"], reverse=True)[:top_k]

    if not results:
        print("  No results above similarity threshold.")
    else:
        for rank, r in enumerate(results):
            bar = "█" * int(r["similarity"] / 4)
            print(f"  {rank+1}. {r['file']:<28}  {r['name']}")
            print(f"     [{r['category']:<22}]  "
                  f"primary={r['primary_score']}%  "
                  f"siglip={r['siglip_score']}%  "
                  f"final={r['similarity']}%")
            print(f"     {bar}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Offline dual-model search using local FAISS index")
    parser.add_argument("image",       nargs="?",            help="Path to query image (omit to search query/ folder)")
    parser.add_argument("--top-k",     type=int,   default=TOP_K,         help="Max results to return")
    parser.add_argument("--threshold", type=float, default=SIM_THRESHOLD, help="Min similarity score (0–1)")
    args = parser.parse_args()

    if args.image:
        if not os.path.exists(args.image):
            print(f"File not found: {args.image}")
            sys.exit(1)
        search(args.image, top_k=args.top_k, threshold=args.threshold)
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
            search(os.path.join(QUERY_DIR, qf), top_k=args.top_k, threshold=args.threshold)


if __name__ == "__main__":
    main()
