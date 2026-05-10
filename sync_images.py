"""
sync_images.py — Index all images from images/ into Qdrant AND build local FAISS indexes.
Reads metadata.json for name, description, category, tags — stores all keys in Qdrant payload.

Saves to index/:
  primary.faiss    — HNSW index (ViT-H-14-378, dim=1024)  → used by search_images.py
  siglip.npy       — SigLIP embedding matrix (dim=1152)    → used by search_images.py for re-rank
  files.json       — ordered filename list                 → used by search_images.py
  files_meta.json  — filename → {name,description,category,tags} → used by search_images.py
  quick.faiss      — flat IP index (ViT-L/14, dim=768)     → used by quick_search.py
  quick_meta.json  — {files:[...], meta:{...}}             → used by quick_search.py

Usage:
    python sync_images.py
    python sync_images.py --images-dir /path/to/images --metadata metadata.json
    python sync_images.py --skip-qdrant    # only rebuild FAISS indexes
    python sync_images.py --skip-faiss     # only upload to Qdrant
"""

import io
import os
import sys
import json
import logging
import warnings
import argparse

import numpy as np
import torch
from PIL import Image

warnings.filterwarnings("ignore", message=".*unauthenticated.*")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clip_encoder
import qdrant_service

IMAGES_DIR    = "images"
METADATA_FILE = "metadata.json"
INDEX_DIR     = "index"

HNSW_M               = 64
HNSW_EF_CONSTRUCTION = 400
HNSW_EF_SEARCH       = 128

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")
DEFAULT_STATUS   = 1


def load_metadata(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    return vec / (norm + 1e-8)


def build_hnsw(dim: int) -> faiss.IndexHNSWFlat:
    idx = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    idx.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    idx.hnsw.efSearch       = HNSW_EF_SEARCH
    return idx


def get_bg_removed(image_path: str, bg_session, cache_dir: str) -> Image.Image:
    os.makedirs(cache_dir, exist_ok=True)
    filename    = os.path.basename(image_path)
    cached_path = os.path.join(cache_dir, os.path.splitext(filename)[0] + ".png")
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


def main():
    parser = argparse.ArgumentParser(description="Sync images to Qdrant and build FAISS indexes")
    parser.add_argument("--images-dir",  default=IMAGES_DIR,    help="Folder containing product images")
    parser.add_argument("--metadata",    default=METADATA_FILE, help="Path to metadata.json")
    parser.add_argument("--index-dir",   default=INDEX_DIR,     help="Output folder for FAISS index files")
    parser.add_argument("--status",      type=int, default=DEFAULT_STATUS, help="Qdrant status (0=inactive, 1=active)")
    parser.add_argument("--skip-qdrant", action="store_true", help="Skip Qdrant upload, only build FAISS indexes")
    parser.add_argument("--skip-faiss",  action="store_true", help="Skip FAISS build, only upload to Qdrant")
    args = parser.parse_args()

    primary_index_file = os.path.join(args.index_dir, "primary.faiss")
    siglip_matrix_file = os.path.join(args.index_dir, "siglip.npy")
    files_list_file    = os.path.join(args.index_dir, "files.json")
    files_meta_file    = os.path.join(args.index_dir, "files_meta.json")
    quick_index_file   = os.path.join(args.index_dir, "quick.faiss")
    quick_meta_file    = os.path.join(args.index_dir, "quick_meta.json")
    bg_cache_dir       = os.path.join(args.images_dir, "bg_removed")

    os.makedirs(args.index_dir, exist_ok=True)

    metadata = load_metadata(args.metadata)

    image_files = sorted([
        f for f in os.listdir(args.images_dir)
        if os.path.isfile(os.path.join(args.images_dir, f))
        and f.lower().endswith(VALID_EXTENSIONS)
    ])

    if not image_files:
        print(f"No images found in '{args.images_dir}/'")
        return

    # ── Phase 1: Qdrant upload + collect embeddings for heavy FAISS ────────
    primary_list = []
    siglip_list  = []
    valid_files  = []
    files_meta   = {}

    if not args.skip_qdrant or not args.skip_faiss:
        if not args.skip_qdrant:
            print("Ensuring Qdrant collection ...")
            qdrant_service.ensure_collection()

        print(f"\nProcessing {len(image_files)} images ...\n")
        clip_encoder._load_primary()
        clip_encoder._load_siglip()

        for filename in image_files:
            path = os.path.join(args.images_dir, filename)
            meta = metadata.get(filename, {})
            name        = meta.get("name", filename)
            description = meta.get("description", "")
            category    = meta.get("category", "")
            tags        = meta.get("tags", [])

            text = " ".join(filter(None, [name, description, category, " ".join(tags)])).strip()
            try:
                with open(path, "rb") as f:
                    image_bytes = f.read()

                primary_vec, siglip_vec = clip_encoder.encode_image_both(image_bytes, text=text)

                if not args.skip_qdrant:
                    qdrant_service.upsert_image(
                        product_id=filename,
                        primary_vec=primary_vec,
                        siglip_vec=siglip_vec,
                        status=args.status,
                        name=name,
                        category=category,
                        tags=tags,
                        description=description,
                    )

                if not args.skip_faiss:
                    primary_list.append(np.array(primary_vec, dtype="float32").reshape(1, -1))
                    siglip_list.append(np.array(siglip_vec,  dtype="float32").reshape(1, -1))
                    valid_files.append(filename)
                    files_meta[filename] = {
                        "name":        name,
                        "description": description,
                        "category":    category,
                        "tags":        tags,
                    }

                label = "Qdrant+FAISS" if (not args.skip_qdrant and not args.skip_faiss) \
                        else ("Qdrant" if not args.skip_qdrant else "FAISS")
                print(f"  ✓  [{label}]  {filename:<28}  {name}")
            except Exception as e:
                print(f"  ✗  {filename}: {e}")

        print()

    # ── Phase 2: Save heavy FAISS index (primary + siglip) ────────────────
    if not args.skip_faiss and primary_list:
        import faiss
        primary_matrix = np.vstack(primary_list)
        siglip_matrix  = np.vstack(siglip_list)

        hnsw = build_hnsw(primary_matrix.shape[1])
        hnsw.add(primary_matrix)

        faiss.write_index(hnsw, primary_index_file)
        np.save(siglip_matrix_file, siglip_matrix)
        with open(files_list_file, "w") as f:
            json.dump(valid_files, f, indent=2)
        with open(files_meta_file, "w") as f:
            json.dump(files_meta, f, indent=2, ensure_ascii=False)

        print(f"✓ Heavy FAISS index saved ({len(valid_files)} images):")
        print(f"  {primary_index_file}  (dim={primary_matrix.shape[1]})")
        print(f"  {siglip_matrix_file}  (dim={siglip_matrix.shape[1]})")
        print(f"  {files_list_file}")
        print(f"  {files_meta_file}\n")

    # ── Phase 3: Quick FAISS index (ViT-L/14) ─────────────────────────────
    if not args.skip_faiss:
        import faiss
        import clip
        from rembg import remove, new_session
        print("Building quick index (ViT-L/14) ...")
        device = clip_encoder.DEVICE
        print(f"  Device : {device}")

        print("  Loading rembg ...", end=" ", flush=True)
        bg_session = new_session("isnet-general-use")
        print("ready.")

        print("  Loading ViT-L/14 ...", end=" ", flush=True)
        q_model, q_preprocess = clip.load("ViT-L/14", device=device)
        q_model.eval()
        print("ready.\n")

        quick_list  = []
        quick_files = []
        quick_meta  = {}

        for filename in image_files:
            path = os.path.join(args.images_dir, filename)
            meta = metadata.get(filename, {})
            try:
                img    = get_bg_removed(path, bg_session, bg_cache_dir)
                tensor = q_preprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = q_model.encode_image(tensor)
                emb = normalize(emb.cpu().numpy().astype("float32"))
                quick_list.append(emb)
                quick_files.append(filename)
                quick_meta[filename] = {
                    "name":        meta.get("name", filename),
                    "description": meta.get("description", ""),
                    "category":    meta.get("category", ""),
                    "tags":        meta.get("tags", []),
                }
                print(f"  ✓  {filename:<30}  {meta.get('name', filename)}")
            except Exception as e:
                print(f"  ✗  {filename}: {e}")

        if quick_list:
            quick_matrix = np.vstack(quick_list)
            q_idx = faiss.IndexFlatIP(quick_matrix.shape[1])
            q_idx.add(quick_matrix)
            faiss.write_index(q_idx, quick_index_file)
            with open(quick_meta_file, "w") as f:
                json.dump({"files": quick_files, "meta": quick_meta}, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Quick index saved ({len(quick_files)} images):")
            print(f"  {quick_index_file}  (dim={quick_matrix.shape[1]})")
            print(f"  {quick_meta_file}")
        else:
            print("No images encoded for quick index.")


if __name__ == "__main__":
    main()
