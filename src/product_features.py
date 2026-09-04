import hashlib
import re

import torch


DEFAULT_CONTENT_FEATURE_DIM = 256


def build_product_feature_matrix(
    products,
    product_mapper,
    feature_dim: int = DEFAULT_CONTENT_FEATURE_DIM,
) -> torch.Tensor:
    """Build deterministic fixed-width features from product metadata."""
    features = torch.zeros(
        product_mapper.num_products,
        feature_dim,
        dtype=torch.float32,
    )
    for product in products:
        product_index = product_mapper.encode(product.product_id)
        for token in _metadata_tokens(product):
            digest = hashlib.blake2b(
                token.encode("utf-8"), digest_size=8
            ).digest()
            bucket = int.from_bytes(digest, "little") % feature_dim
            features[product_index, bucket] += 1.0

    norms = features.norm(dim=1, keepdim=True).clamp(min=1.0)
    return features / norms


def _metadata_tokens(product) -> list[str]:
    fields = {
        "name": product.name,
        "gender": product.gender,
        "category": product.category,
        "subcategory": product.sub_category,
        "zone": product.zone,
        "color": product.color,
    }
    tokens = []
    for field, raw_value in fields.items():
        if raw_value is None:
            continue
        value = str(raw_value).strip().casefold()
        if not value:
            continue
        tokens.append(f"{field}={value}")
        tokens.extend(
            f"{field}:{part}"
            for part in re.findall(r"[0-9a-zA-Z가-힣]+", value)
        )
    return tokens
