import math
from collections import Counter


CONTRASTIVE_MIDDLE_START = 0.30
CONTRASTIVE_MIDDLE_END = 0.70
REFRESH_RESULT_LIMIT = 6


def has_new_interactions(current_interactions, previous_interactions):
    current_counts = Counter(
        tuple(sorted(interaction.items()))
        for interaction in current_interactions
    )
    previous_counts = Counter(
        tuple(sorted(interaction.items()))
        for interaction in previous_interactions
    )
    return bool(current_counts - previous_counts)


def select_contrastive_products(
    scored_products,
    previous_product_ids,
    limit=REFRESH_RESULT_LIMIT,
):
    previous_product_ids = set(previous_product_ids)
    unique_products = {}
    for item in scored_products:
        product_id = item["productId"]
        if product_id in previous_product_ids:
            continue
        current = unique_products.get(product_id)
        if current is None or item["score"] > current["score"]:
            unique_products[product_id] = item

    ranked = sorted(
        unique_products.values(),
        key=lambda item: (-item["score"], item["productId"]),
    )
    if not ranked or limit <= 0:
        return []

    middle_start = math.floor(len(ranked) * CONTRASTIVE_MIDDLE_START)
    middle_end = math.ceil(len(ranked) * CONTRASTIVE_MIDDLE_END)
    selected_indexes = list(range(middle_start, middle_end))[:limit]

    # Expand toward adjacent score bands when the middle band is too small.
    upper_index = middle_start - 1
    lower_index = middle_end
    while len(selected_indexes) < min(limit, len(ranked)):
        if upper_index >= 0:
            selected_indexes.append(upper_index)
            upper_index -= 1
            if len(selected_indexes) == min(limit, len(ranked)):
                break
        if lower_index < len(ranked):
            selected_indexes.append(lower_index)
            lower_index += 1

    selected_indexes.sort()
    return [ranked[index] for index in selected_indexes]
