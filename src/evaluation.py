import math
import statistics


ZONE_ALIASES = {"NEW_COLLECTION": "NEW"}


def evaluate_personas(personas, recommender):
    results = [evaluate_persona(persona, recommender) for persona in personas]
    return {
        "summary": summarize(results),
        "personas": results,
    }


def evaluate_persona(persona, recommender):
    interactions = [
        item.model_dump(include={"productId", "interactionType"})
        for item in sorted(persona.arInteractions, key=lambda item: item.sequenceNo)
    ]
    if not interactions:
        raise ValueError(f"{persona.personaId}: arInteractions cannot be empty")

    preference_scores = build_preference_scores(persona, recommender)
    recommendations = recommender.recommend(
        interactions=interactions,
        zone_scores=preference_scores,
        category=None,
        top_k=len(recommender.products),
        exclude_seen=False,
    )
    products_by_id = {
        product.product_id: product for product in recommender.products
    }
    scored_products = [
        {
            **item,
            "category": products_by_id[item["productId"]].category,
        }
        for item in recommendations
    ]

    overall = sorted(
        scored_products,
        key=lambda item: (-item["score"], item["productId"]),
    )
    truth = {
        item.productId: item.relevance
        for item in persona.groundTruth.recommendations
    }
    overall_positions = {
        item["productId"]: index + 1
        for index, item in enumerate(overall)
    }
    overall_by_id = {item["productId"]: item for item in overall}
    anchor_id = persona.groundTruth.anchorProductId
    if anchor_id not in overall_by_id:
        raise ValueError(
            f"{persona.personaId}: anchor product {anchor_id} is missing from ranking"
        )
    missing_truth = set(truth) - set(overall_by_id)
    if missing_truth:
        raise ValueError(
            f"{persona.personaId}: Ground Truth products are missing from ranking: "
            f"{sorted(missing_truth)}"
        )
    anchor_item = overall_by_id[anchor_id]
    overall_top5 = overall[:5]

    return {
        "personaId": persona.personaId,
        "personaType": persona.personaType.upper(),
        "anchorEvaluation": {
            "expectedProductId": anchor_id,
            "predictedProductId": overall[0]["productId"],
            "hit": overall[0]["productId"] == anchor_id,
            "expectedAnchorRank": overall_positions[anchor_id],
            "expectedAnchorScore": anchor_item["score"],
            "top5": ranked(overall_top5, truth),
        },
        "rankingEvaluation": {
            "recallAt5": (
                sum(item["productId"] in truth for item in overall_top5)
                / len(truth)
            ),
            "ndcgAt5": ndcg_at5(overall_top5, truth),
            "top5": ranked(overall_top5, truth),
            "groundTruthResults": [
                {
                    "productId": expected.productId,
                    "relevance": expected.relevance,
                    "overallRank": overall_positions[expected.productId],
                    "score": overall_by_id[expected.productId]["score"],
                    "includedInTop5": overall_positions[expected.productId] <= 5,
                }
                for expected in persona.groundTruth.recommendations
            ],
        },
        "confidenceEvaluation": confidence(overall_top5),
    }


def build_preference_scores(persona, recommender):
    dwell_by_category = {}
    for interaction in sorted(
        persona.zoneInteractions,
        key=lambda item: item.sequenceNo,
    ):
        category = interaction.category.strip().upper()
        zone = ZONE_ALIASES.get(
            interaction.zone.strip().upper(),
            interaction.zone.strip().upper(),
        )
        category_dwell = dwell_by_category.setdefault(category, {})
        category_dwell[zone] = (
            category_dwell.get(zone, 0.0) + max(0.0, interaction.dwellSeconds)
        )

    zone_preferences = {}
    for category, dwell_by_zone in dwell_by_category.items():
        total = sum(dwell_by_zone.values())
        zone_preferences[category] = {
            zone: dwell / total if total else 0.0
            for zone, dwell in dwell_by_zone.items()
        }

    member_scores = (
        recommender.build_wishlist_preference_scores(persona.memberWishlists)
        if persona.memberWishlists else {}
    )
    has_zone = bool(persona.zoneInteractions)
    has_member = bool(persona.memberWishlists)
    result = {}
    for product in recommender.products:
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


def ranked(items, truth):
    return [
        {
            "rank": index + 1,
            "productId": item["productId"],
            "category": item["category"],
            "score": item["score"],
            "relevance": truth.get(item["productId"], 0),
        }
        for index, item in enumerate(items)
    ]


def ndcg_at5(top5, truth):
    def dcg(relevances):
        return sum(
            relevance / math.log2(rank + 1)
            for rank, relevance in enumerate(relevances, start=1)
        )

    actual = dcg([truth.get(item["productId"], 0) for item in top5])
    ideal = dcg(sorted(truth.values(), reverse=True)[:5])
    return actual / ideal if ideal else 0.0


def confidence(top5):
    scores = [item["score"] for item in top5]
    gap = scores[0] - scores[1] if len(scores) > 1 else 0.0
    deviation = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    maximum = max(scores)
    exponentials = [math.exp(score - maximum) for score in scores]
    total = sum(exponentials)
    probabilities = [value / total for value in exponentials]
    entropy = -sum(value * math.log(value) for value in probabilities if value)
    normalized = entropy / math.log(len(scores)) if len(scores) > 1 else 0.0
    return {
        "top1Top2Gap": gap,
        "top5StandardDeviation": deviation,
        "normalizedEntropy": normalized,
    }


def summarize(results):
    confident = [item for item in results if item["personaType"] == "CONFIDENT"]
    exploratory = [item for item in results if item["personaType"] == "EXPLORATORY"]

    def average(group, field):
        if not group:
            return None
        return statistics.fmean(
            item["confidenceEvaluation"][field] for item in group
        )

    return {
        "personaCount": len(results),
        "top1AnchorAccuracy": statistics.fmean(
            1.0 if item["anchorEvaluation"]["hit"] else 0.0
            for item in results
        ),
        "meanRecallAt5": statistics.fmean(
            item["rankingEvaluation"]["recallAt5"] for item in results
        ),
        "meanNdcgAt5": statistics.fmean(
            item["rankingEvaluation"]["ndcgAt5"] for item in results
        ),
        "confidentGroupAverageGap": average(confident, "top1Top2Gap"),
        "exploratoryGroupAverageGap": average(exploratory, "top1Top2Gap"),
        "confidentGroupAverageEntropy": average(confident, "normalizedEntropy"),
        "exploratoryGroupAverageEntropy": average(exploratory, "normalizedEntropy"),
    }
