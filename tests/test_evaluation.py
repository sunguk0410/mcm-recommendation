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

    def recommend(self, interactions, zone_scores, category, top_k, exclude_seen):
        self.assertions = {
            "interactions": interactions,
            "exclude_seen": exclude_seen,
        }
        scores = {
            70: 0.95,
            50: 0.70,
            62: 0.90,
            53: 0.80,
            # A higher score in another category must not affect BAG metrics.
            91: 1.50,
        }
        products = [item for item in self.products if item.category == category]
        return [
            {"productId": item.product_id, "score": scores[item.product_id]}
            for item in sorted(products, key=lambda item: -scores[item.product_id])[:top_k]
        ]


class RecommendationEvaluationTest(unittest.TestCase):
    def test_anchor_and_metrics_use_category_rank(self):
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
                anchorProductId=70,
                category="BAG",
                recommendations=[
                    SimpleNamespace(productId=62, relevance=5),
                    SimpleNamespace(productId=53, relevance=4),
                ],
            ),
        )

        response = evaluate_personas([persona], recommender)
        result = response["personas"][0]

        self.assertTrue(result["anchorEvaluation"]["hit"])
        self.assertFalse(recommender.assertions["exclude_seen"])
        self.assertEqual(70, result["anchorEvaluation"]["predictedProductId"])
        self.assertTrue(all(
            item["category"] == "BAG"
            for item in result["anchorEvaluation"]["top5"]
        ))
        self.assertEqual(2, result["categoryEvaluation"]["groundTruthResults"][0]["categoryRank"])
        self.assertEqual(1.0, result["categoryEvaluation"]["recallAt5"])
        self.assertGreater(result["categoryEvaluation"]["ndcgAt5"], 0.0)

    def test_guest_uses_empty_member_wishlist(self):
        recommender = FixedRecommender()
        persona = SimpleNamespace(
            personaId="P7",
            personaType="EXPLORATORY",
            zoneInteractions=[],
            arInteractions=[DumpableInteraction(70, "PRODUCT_SELECT", 1)],
            memberWishlists=[],
            groundTruth=SimpleNamespace(
                anchorProductId=70,
                category="BAG",
                recommendations=[SimpleNamespace(productId=62, relevance=5)],
            ),
        )

        response = evaluate_personas([persona], recommender)

        self.assertEqual(1, response["summary"]["personaCount"])


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
