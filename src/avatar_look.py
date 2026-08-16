ANCHOR_PRIORITY = {
    "PRODUCT_SELECT": 1,
    "FITTING_ADD": 2,
    "WISHLIST_ADD": 3,
}


def calculate_product_active_states(interactions):
    states = {}
    for sequence_index, interaction in enumerate(interactions):
        product_id = int(interaction["productId"])
        interaction_type = interaction["interactionType"]
        state = states.setdefault(
            product_id,
            {
                "productId": product_id,
                "productSelected": False,
                "fittingActive": False,
                "wishlistActive": False,
                "lastProductSelectIndex": -1,
                "lastFittingAddIndex": -1,
                "lastWishlistAddIndex": -1,
                "hasExplicitRemove": False,
            },
        )

        if interaction_type == "PRODUCT_SELECT":
            state["productSelected"] = True
            state["lastProductSelectIndex"] = sequence_index
        elif interaction_type == "FITTING_ADD":
            state["fittingActive"] = True
            state["lastFittingAddIndex"] = sequence_index
        elif interaction_type == "FITTING_REMOVE":
            state["fittingActive"] = False
            state["hasExplicitRemove"] = True
        elif interaction_type == "WISHLIST_ADD":
            state["wishlistActive"] = True
            state["lastWishlistAddIndex"] = sequence_index
        elif interaction_type == "WISHLIST_REMOVE":
            state["wishlistActive"] = False
            state["hasExplicitRemove"] = True

    return states


def get_anchor_priority_and_recency(state):
    if state["wishlistActive"]:
        return ANCHOR_PRIORITY["WISHLIST_ADD"], state["lastWishlistAddIndex"]
    if state["fittingActive"]:
        return ANCHOR_PRIORITY["FITTING_ADD"], state["lastFittingAddIndex"]
    if state["productSelected"]:
        return ANCHOR_PRIORITY["PRODUCT_SELECT"], state["lastProductSelectIndex"]
    return 0, -1


def select_category_anchors(product_states, product_by_id, scores_by_product_id):
    anchors = {}
    anchor_keys = {}
    for product_id, state in product_states.items():
        product = product_by_id.get(product_id)
        if product is None:
            continue

        priority, recency = get_anchor_priority_and_recency(state)
        if priority == 0:
            continue

        score = scores_by_product_id.get(product_id, float("-inf"))
        key = (priority, score, recency, -product_id)
        if key > anchor_keys.get(product.category, (-1, float("-inf"), -1, 0)):
            anchors[product.category] = {
                "productId": product_id,
                "category": product.category,
                "score": score,
                "anchorPriority": priority,
            }
            anchor_keys[product.category] = key

    return anchors


def explicitly_removed_without_positive_state(product_states):
    return {
        product_id
        for product_id, state in product_states.items()
        if state["hasExplicitRemove"]
        and get_anchor_priority_and_recency(state)[0] == 0
    }


def select_category_complement(recommendations, excluded_product_ids):
    return next(
        (
            recommendation
            for recommendation in recommendations
            if recommendation["productId"] not in excluded_product_ids
        ),
        None,
    )
