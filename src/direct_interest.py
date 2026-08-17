import math


ONLINE_WISHLIST_SCORE = 2.0
ACTION_SCORES = {
    "PRODUCT_SELECT": 2.0,
    "FITTING": 3.0,
}
SESSION_WISHLIST_ADD_SCORE = 5.0
SESSION_WISHLIST_REMOVE_SCORE = -2.0
REPEAT_DECAY = 0.5
RECENCY_HALF_LIFE_EVENTS = 8.0


def score_direct_interest(
    products,
    online_wishlist_product_ids,
    interactions,
):
    """Return positively scored products ranked by explicit user interest."""
    products_by_id = {
        product.product_id: product
        for product in products
    }
    online_wishlist_ids = {
        int(product_id)
        for product_id in online_wishlist_product_ids
        if int(product_id) in products_by_id
    }
    valid_interactions = []
    for interaction in interactions:
        try:
            product_id = int(interaction["productId"])
        except (KeyError, TypeError, ValueError):
            continue
        interaction_type = interaction.get("interactionType")
        if (
            product_id not in products_by_id
            or interaction_type not in {
                *ACTION_SCORES,
                "WISHLIST_ADD",
                "WISHLIST_REMOVE",
            }
        ):
            continue
        valid_interactions.append((product_id, interaction_type))

    scores = {
        product_id: ONLINE_WISHLIST_SCORE
        for product_id in online_wishlist_ids
    }
    evidence = {
        product_id: ["ONLINE_WISHLIST"]
        for product_id in online_wishlist_ids
    }
    latest_positions = {
        product_id: -1
        for product_id in online_wishlist_ids
    }
    repeat_counts = {}
    latest_wishlist_events = {}
    last_position = len(valid_interactions) - 1

    for position, (product_id, interaction_type) in enumerate(valid_interactions):
        latest_positions[product_id] = position
        evidence.setdefault(product_id, []).append(interaction_type)

        if interaction_type in {"WISHLIST_ADD", "WISHLIST_REMOVE"}:
            latest_wishlist_events[product_id] = (interaction_type, position)
            continue

        repeat_key = (product_id, interaction_type)
        occurrence = repeat_counts.get(repeat_key, 0)
        repeat_counts[repeat_key] = occurrence + 1
        scores[product_id] = scores.get(product_id, 0.0) + (
            ACTION_SCORES[interaction_type]
            * math.pow(REPEAT_DECAY, occurrence)
            * _recency_factor(last_position - position)
        )

    for product_id, (interaction_type, position) in latest_wishlist_events.items():
        # A session wishlist event supersedes the online wishlist snapshot.
        scores[product_id] = scores.get(product_id, 0.0) - (
            ONLINE_WISHLIST_SCORE if product_id in online_wishlist_ids else 0.0
        )
        wishlist_score = (
            SESSION_WISHLIST_ADD_SCORE
            if interaction_type == "WISHLIST_ADD"
            else SESSION_WISHLIST_REMOVE_SCORE
        )
        scores[product_id] = scores.get(product_id, 0.0) + (
            wishlist_score * _recency_factor(last_position - position)
        )

    ranked_products = []
    for product_id, score in scores.items():
        if score <= 0.0:
            continue
        product = products_by_id[product_id]
        ranked_products.append({
            "productId": product_id,
            "name": getattr(product, "name", None),
            "category": product.category,
            "subCategory": getattr(product, "sub_category", None),
            "zone": product.zone,
            "color": getattr(product, "color", None),
            "directInterestScore": float(score),
            "latestInteractionPosition": latest_positions.get(product_id, -1),
            "evidence": evidence.get(product_id, []),
        })

    ranked_products.sort(
        key=lambda item: (
            -item["directInterestScore"],
            -item["latestInteractionPosition"],
            item["productId"],
        )
    )
    return ranked_products


def _recency_factor(events_since):
    return math.pow(0.5, events_since / RECENCY_HALF_LIFE_EVENTS)
