"""Build catalog-aligned CLIP vectors and a same-category visual review report."""

import argparse
import hashlib
import html
import json
import os
from pathlib import Path

import torch
from PIL import Image, ImageOps

from .affinity import load_products_from_excel


def load_image(path):
    # White compositing avoids treating transparent backgrounds as black.
    with Image.open(path) as source:
        rgba = ImageOps.exif_transpose(source).convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        background.alpha_composite(rgba)
        rgb = background.convert("RGB")
    # Keep the entire product visible through CLIP's square center crop.
    side = max(rgb.size)
    square = Image.new("RGB", (side, side), "white")
    square.paste(rgb, ((side - rgb.width) // 2, (side - rgb.height) // 2))
    return square


def nearest_neighbors(features, products, top_k):
    scores = features @ features.T
    results = []
    for index, product in enumerate(products):
        candidates = [
            j for j, other in enumerate(products)
            if j != index and other.category == product.category
        ]
        candidates.sort(key=lambda j: (-float(scores[index, j]), products[j].product_id))
        results.append([
            {"index": j, "product_id": products[j].product_id,
             "product_code": products[j].product_code,
             "cosine_similarity": float(scores[index, j])}
            for j in candidates[:top_k]
        ])
    return results


def write_report(path, products, image_paths, neighbors):
    def card(index, caption):
        product = products[index]
        relative = Path(os.path.relpath(image_paths[index], path.parent)).as_posix()
        return (
            '<figure><img loading="lazy" src="' + html.escape(relative, quote=True)
            + '"><figcaption><strong>' + html.escape(caption)
            + '</strong><br>' + html.escape(product.name)
            + '<br><small>' + html.escape(product.product_code) + '</small></figcaption></figure>'
        )

    rows = []
    for i, product in enumerate(products):
        cards = card(i, "기준 상품") + ''.join(
            card(n["index"], f'유사도 {n["cosine_similarity"]:.3f}') for n in neighbors[i]
        )
        rows.append('<section><h2>' + html.escape(product.category) + ' · '
                    + str(product.product_id) + '</h2><div class="row">' + cards + '</div></section>')
    path.write_text('''<!doctype html><html lang="ko"><meta charset="utf-8">
<title>상품 이미지 유사도 검토</title><style>
body{font-family:system-ui,sans-serif;background:#f5f5f5;margin:24px;color:#222}
.row{display:grid;grid-template-columns:repeat(6,minmax(130px,1fr));gap:12px}
figure{margin:0;padding:12px;background:white;border-radius:8px}figure:first-child{outline:2px solid #3568ac}
img{width:100%;height:200px;object-fit:contain}figcaption{font-size:13px;line-height:1.6}
section{margin:32px 0}h2{font-size:18px}small{overflow-wrap:anywhere}
@media(max-width:850px){.row{grid-template-columns:repeat(3,1fr)}}
</style><h1>상품 이미지 유사도 검토</h1>
<p>같은 카테고리 내 CLIP 코사인 유사도 상위 상품입니다. 유사도는 선호 확률이 아닙니다.
색상뿐 아니라 형태·패턴도 비슷한지 확인하세요.</p>'''
                    + ''.join(rows) + '</html>', encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="MCM_제품리스트_통합_추천모델용.xlsx")
    parser.add_argument("--images", type=Path, default=Path("images"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/image_similarity"))
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.batch_size < 1 or args.top_k < 1:
        parser.error("batch-size and top-k must be positive")
    products = load_products_from_excel(args.catalog)
    if not products or len({p.product_code for p in products}) != len(products):
        raise ValueError("Catalog must contain unique product codes and at least one product")
    paths = [(args.images / (p.product_code + ".png")).resolve() for p in products]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise ValueError(f"Missing images: {missing}")
    # Validate all files before any model download.
    for path in paths:
        with Image.open(path) as source:
            source.verify()
    args.output.mkdir(parents=True, exist_ok=True)
    from transformers import CLIPImageProcessor, CLIPModel

    cache = args.output / "model_cache"
    processor = CLIPImageProcessor.from_pretrained(
        args.model, revision=args.revision, cache_dir=cache
    )
    model = CLIPModel.from_pretrained(
        args.model, revision=args.revision, cache_dir=cache, use_safetensors=True
    ).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    batches = []
    with torch.inference_mode():
        for start in range(0, len(paths), args.batch_size):
            images = [load_image(p) for p in paths[start:start + args.batch_size]]
            inputs = processor(images=images, return_tensors="pt").to(device)
            vectors = model.get_image_features(**inputs)
            batches.append(torch.nn.functional.normalize(vectors, dim=-1).cpu())
            print(f"Encoded {min(start + args.batch_size, len(paths))}/{len(paths)}", flush=True)
    features = torch.cat(batches)
    if not torch.isfinite(features).all() or (features.norm(dim=1) < 0.99).any():
        raise ValueError("Invalid image embeddings")
    metadata = {
        "model": args.model, "revision": getattr(model.config, "_commit_hash", args.revision),
        "preprocessing": "exif transpose; white alpha composite; square white padding; CLIP processor",
        "products": [
            {"product_id": p.product_id, "product_code": p.product_code,
             "category": p.category, "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for p, path in zip(products, paths)
        ],
    }
    torch.save({"features": features, **metadata}, args.output / "embeddings.pt")
    neighbors = nearest_neighbors(features, products, args.top_k)
    (args.output / "neighbors.json").write_text(
        json.dumps({**metadata, "neighbors": neighbors}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(args.output / "report.html", products, paths, neighbors)
    print(f"Saved {tuple(features.shape)} embeddings and report to {args.output}", flush=True)


if __name__ == "__main__":
    main()
