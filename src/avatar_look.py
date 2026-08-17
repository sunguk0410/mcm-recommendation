import logging


logger = logging.getLogger(__name__)

AVATAR_LOOK_THRESHOLD = 0.75
AVATAR_LOOK_FALLBACK_LIMIT = 3


def get_avatar_look_limit(interaction_count):
    if interaction_count is None:
        return None
    if interaction_count <= 2:
        return 3
    if interaction_count <= 5:
        return 4
    return None


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


def select_avatar_look_products(
    scored_products,
    ar_session_id=None,
    interaction_count=None,
):
    if not scored_products:
        logger.info(
            "Avatar Look selection. arSessionId=%s, scoredProductsCount=0, "
            "thresholdPassCount=0, rawScoreFallbackApplied=false, "
            "finalProductsCount=0, productIds=[]",
            ar_session_id,
        )
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
            selected = sorted(
                best_by_category.values(),
                key=lambda item: (-item["normalizedScore"], item["productId"]),
            )
            result_limit = get_avatar_look_limit(interaction_count)
            if result_limit is not None:
                selected = selected[:result_limit]
            logger.info(
                "Avatar Look selection. arSessionId=%s, scoredProductsCount=%s, "
                "thresholdPassCount=%s, rawScoreFallbackApplied=false, "
                "interactionCount=%s, resultLimit=%s, finalProductsCount=%s, "
                "productIds=%s",
                ar_session_id,
                len(scored_products),
                len(candidates),
                interaction_count,
                result_limit,
                len(selected),
                [item["productId"] for item in selected],
            )
            return selected

    selected = select_raw_score_fallback(scored_products)
    logger.info(
        "Avatar Look selection. arSessionId=%s, scoredProductsCount=%s, "
        "thresholdPassCount=%s, rawScoreFallbackApplied=true, "
        "finalProductsCount=%s, productIds=%s",
        ar_session_id,
        len(scored_products),
        0 if all_scores_equal else len(candidates),
        len(selected),
        [item["productId"] for item in selected],
    )
    return selected


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
