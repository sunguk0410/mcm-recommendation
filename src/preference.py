ZONE_ALIASES = {"NEW_COLLECTION": "NEW"}


def build_zone_preferences(interactions):
    """Return product-compatible Category x Zone shares of total store dwell."""
    dwell_by_category = {}
    total_dwell = 0.0
    for interaction in interactions:
        category = interaction.category.strip().upper()
        raw_zone = interaction.zone.strip().upper()
        zone = ZONE_ALIASES.get(raw_zone, raw_zone)
        dwell_seconds = max(0.0, interaction.dwellSeconds)
        category_dwell = dwell_by_category.setdefault(category, {})
        category_dwell[zone] = category_dwell.get(zone, 0.0) + dwell_seconds
        total_dwell += dwell_seconds

    if total_dwell <= 0:
        return {
            category: {zone: 0.0 for zone in zone_dwell}
            for category, zone_dwell in dwell_by_category.items()
        }

    return {
        category: {
            zone: dwell_seconds / total_dwell
            for zone, dwell_seconds in zone_dwell.items()
        }
        for category, zone_dwell in dwell_by_category.items()
    }


def build_product_preference_scores(
    products,
    zone_interactions,
    member_scores=None,
):
    zone_preferences = build_zone_preferences(zone_interactions)
    member_scores = member_scores or {}
    has_zone = bool(zone_interactions)
    has_member = bool(member_scores)
    result = {}

    for product in products:
        zone_score = zone_preferences.get(product.category, {}).get(product.zone, 0.0)
        member_score = member_scores.get(product.product_id, 0.0)
        if has_zone and has_member:
            score = 0.7 * zone_score + 0.3 * member_score
        elif has_zone:
            score = zone_score
        else:
            score = member_score
        result[product.product_id] = score

    return result
