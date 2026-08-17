import logging


logger = logging.getLogger(__name__)

def select_avatar_look_products(
    scored_products,
    ar_session_id=None,
):
    if not scored_products:
        logger.info(
            "Avatar Look selection. arSessionId=%s, scoredProductsCount=0, "
            "finalProductsCount=0, productIds=[]",
            ar_session_id,
        )
        return []

    best_by_category = {}
    for candidate in scored_products:
        current = best_by_category.get(candidate["category"])
        if current is None or _ranking_key(candidate) < _ranking_key(current):
            best_by_category[candidate["category"]] = candidate

    selected = sorted(best_by_category.values(), key=_ranking_key)
    logger.info(
        "Avatar Look selection. arSessionId=%s, scoredProductsCount=%s, "
        "finalProductsCount=%s, productIds=%s",
        ar_session_id,
        len(scored_products),
        len(selected),
        [item["productId"] for item in selected],
    )
    return selected


def _ranking_key(item):
    return (
        -item["directInterestScore"],
        -item.get("latestInteractionPosition", -1),
        item["productId"],
    )
