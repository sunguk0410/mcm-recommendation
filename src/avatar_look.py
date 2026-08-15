AVATAR_LOOK_THRESHOLD = 0.5
AVATAR_LOOK_FALLBACK_LIMIT = 3


def normalize_avatar_scores(recommendations):
    if not recommendations:
        return []

    scores = [item["score"] for item in recommendations]
    minimum = min(scores)
    maximum = max(scores)
    score_range = maximum - minimum

    return [
        {
            **item,
            "normalizedScore": (item["score"] - minimum) / score_range,
        }
        for item in recommendations
    ]


def select_avatar_look_products(scored_products):
    if not scored_products:
        return []

    scores = [item["score"] for item in scored_products]
    all_scores_equal = max(scores) == min(scores)

    if not all_scores_equal:
        candidates = [
            item
            for item in normalize_avatar_scores(scored_products)
            if item["normalizedScore"] >= AVATAR_LOOK_THRESHOLD
        ]

        best_by_category = {}
        for candidate in candidates:
            current = best_by_category.get(candidate["category"])
            if (
                current is None
                or candidate["normalizedScore"] > current["normalizedScore"]
            ):
                best_by_category[candidate["category"]] = candidate

        if best_by_category:
            return list(best_by_category.values())

    return select_raw_score_fallback(scored_products)


def select_raw_score_fallback(scored_products):
    selected = []
    used_categories = set()
    for item in sorted(
        scored_products,
        key=lambda product: (-product["score"], product["productId"]),
    ):
        if item["category"] in used_categories:
            continue
        selected.append(item)
        used_categories.add(item["category"])
        if len(selected) == AVATAR_LOOK_FALLBACK_LIMIT:
            break

    return selected
