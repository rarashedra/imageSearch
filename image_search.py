"""
image_search.py — Full-accuracy search against Qdrant.
Uses dual-model pipeline (ViT-H-14-378 + SigLIP), same as the API.

Run sync_images.py first to populate Qdrant.

Usage:
    python image_search.py path/to/query.jpg
    python image_search.py                    # searches all images in query/
    python image_search.py --top-k 10 --threshold 0.15 path/to/query.jpg
"""

import os
import sys
import argparse
import logging
import warnings

warnings.filterwarnings("ignore", message=".*unauthenticated.*")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clip_encoder
import qdrant_service

QUERY_DIR        = "query"
TOP_K            = 5
THRESHOLD        = 0.20
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")


def search(query_path: str, top_k: int = TOP_K, threshold: float = THRESHOLD) -> None:
    query_file = os.path.basename(query_path)
    print(f"Query : {query_file}")
    print("─" * 65)

    with open(query_path, "rb") as f:
        image_bytes = f.read()

    primary_vec, siglip_vec = clip_encoder.encode_image_both(image_bytes)

    results = qdrant_service.search_similar(
        primary_vec, siglip_vec, top_k=top_k, score_threshold=threshold
    )

    if not results:
        print("  No results above similarity threshold.")
    else:
        for rank, r in enumerate(results):
            sim  = round(r["score"] * 100, 2)
            bar  = "█" * int(sim / 4)
            name = r.get("name", "") or r["product_id"]
            print(f"  {rank+1}. {r['product_id']:<28}  {name}")
            print(f"     [{r.get('category',''):<22}]  "
                  f"primary={round(r['primary_score']*100, 1)}%  "
                  f"siglip={round(r['siglip_score']*100, 1)}%  "
                  f"final={sim}%")
            print(f"     {bar}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Search Qdrant with a query image (full accuracy)")
    parser.add_argument("image", nargs="?",            help="Path to query image (omit to search all images in query/)")
    parser.add_argument("--top-k",     type=int,   default=TOP_K,     help="Max results to return")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="Minimum similarity score (0–1)")
    args = parser.parse_args()

    print("Loading models ...")
    clip_encoder._load_primary()
    clip_encoder._load_siglip()
    print()

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
