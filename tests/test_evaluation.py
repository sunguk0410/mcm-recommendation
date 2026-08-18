import math
import unittest
from types import SimpleNamespace

from src.evaluation import evaluate_personas


class FixedRecommender:
    def __init__(self):
        self.products = [
            SimpleNamespace(product_id=70, category="BAG", zone="TRAVEL"),
            SimpleNamespace(product_id=50, category="BAG", zone="TRAVEL"),
            SimpleNamespace(product_id=62, category="BAG", zone="NEW"),
            SimpleNamespace(product_id=53, category="BAG", zone="TRAVEL"),
            SimpleNamespace(product_id=91, category="CLOTHING", zone="NEW"),
        ]

    def build_wishlist_preference_scores(self, interactions):
        return {item.productId: 1.0 for item in interactions}

    def recommend(
        self,
        interactions,
        zone_scores,
        category,
        top_k,
        exclude_seen,
        diversify=False,
        preference_product_ids=None,
    ):
        self.assertions = {
            "interactions": interactions,
            "exclude_seen": exclude_seen,
            "category": category,
            "top_k": top_k,
            "diversify": diversify,
            "preference_product_ids": preference_product_ids,
        }
        scores = {
            70: 0.95,
            50: 0.70,
            62: 0.90,
            53: 0.80,
            # A higher score in another category leads the overall ranking.
            91: 1.50,
        }
        seen_product_ids = {
            interaction["productId"] for interaction in interactions
        } if exclude_seen else set()
        products = [
            product for product in self.products
            if product.product_id not in seen_product_ids
        ]
        return [
            {"productId": item.product_id, "score": scores[item.product_id]}
            for item in sorted(products, key=lambda item: -scores[item.product_id])[:top_k]
        ]


class RecommendationEvaluationTest(unittest.TestCase):
    def test_next_item_metrics_use_overall_rank_and_exclude_seen(self):
        recommender = FixedRecommender()
        persona = SimpleNamespace(
            personaId="P1",
            personaType="CONFIDENT",
            zoneInteractions=[
                SimpleNamespace(
                    zone="TRAVEL",
                    category="BAG",
                    dwellSeconds=375,
                    sequenceNo=1,
                )
            ],
            arInteractions=[
                DumpableInteraction(70, "PRODUCT_SELECT", 1),
                DumpableInteraction(70, "FITTING", 2),
            ],
            memberWishlists=[SimpleNamespace(productId=70)],
            groundTruth=SimpleNamespace(
                recommendations=[
                    SimpleNamespace(productId=62, relevance=5),
                    SimpleNamespace(productId=53, relevance=4),
                ],
            ),
        )

        response = evaluate_personas([persona], recommender)
        result = response["personas"][0]

        self.assertTrue(recommender.assertions["exclude_seen"])
        self.assertIsNone(recommender.assertions["category"])
        self.assertFalse(recommender.assertions["diversify"])
        self.assertEqual(len(recommender.products), recommender.assertions["top_k"])
        self.assertNotIn("anchorEvaluation", result)
        self.assertNotIn("top1AnchorAccuracy", response["summary"])
        self.assertEqual("CLOTHING", result["rankingEvaluation"]["top5"][0]["category"])
        self.assertEqual(2, result["rankingEvaluation"]["groundTruthResults"][0]["overallRank"])
        self.assertEqual(1.0, result["rankingEvaluation"]["recallAt5"])
        expected_ndcg = (
            5 / math.log2(3) + 4 / math.log2(4)
        ) / (
            5 / math.log2(2) + 4 / math.log2(3)
        )
        self.assertAlmostEqual(
            expected_ndcg,
            result["rankingEvaluation"]["ndcgAt5"],
        )

    def test_guest_uses_empty_member_wishlist(self):
        recommender = FixedRecommender()
        persona = SimpleNamespace(
            personaId="P7",
            personaType="EXPLORATORY",
            zoneInteractions=[],
            arInteractions=[DumpableInteraction(70, "PRODUCT_SELECT", 1)],
            memberWishlists=[],
            groundTruth=SimpleNamespace(
                recommendations=[SimpleNamespace(productId=62, relevance=5)],
            ),
        )

        response = evaluate_personas([persona], recommender)

        self.assertEqual(1, response["summary"]["personaCount"])
        self.assertTrue(recommender.assertions["diversify"])


class DumpableInteraction:
    def __init__(self, product_id, interaction_type, sequence_no):
        self.productId = product_id
        self.interactionType = interaction_type
        self.sequenceNo = sequence_no

    def model_dump(self, include=None):
        values = {
            "productId": self.productId,
            "interactionType": self.interactionType,
            "sequenceNo": self.sequenceNo,
        }
        return {key: value for key, value in values.items() if include is None or key in include}


if __name__ == "__main__":
    unittest.main()
