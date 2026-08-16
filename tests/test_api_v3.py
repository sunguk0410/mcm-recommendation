import unittest

from fastapi.testclient import TestClient

from src.inference import CHECKPOINT_PATH
from src.main import (
    AvatarLookRequest,
    CategoryRankingValidationRequest,
    InteractionRequest,
    RecommendationRequest,
    RefreshRecommendationRequest,
    recommender,
    recommend,
    recommend_avatar_look,
    refresh_recommendations,
    store_session_interactions,
    validate_category_rankings,
    app,
)


class RecommendationApiV3Test(unittest.TestCase):
    def test_serving_uses_v3_checkpoint(self):
        self.assertEqual("checkpoints/recrec_v3_best.pt", CHECKPOINT_PATH)

    def test_recommend_accepts_new_fitting_actions_and_excludes_seen_product(self):
        response = recommend(
            RecommendationRequest(
                arSessionId=91001,
                category="BAG",
                topK=6,
                interactions=[
                    InteractionRequest(productId=46, interactionType="PRODUCT_SELECT"),
                    InteractionRequest(productId=46, interactionType="FITTING_ADD"),
                    InteractionRequest(productId=46, interactionType="FITTING_REMOVE"),
                ],
            )
        )
        self.assertNotIn(46, [item["productId"] for item in response["recommendations"]])


class ValidationAndAvatarApiTest(unittest.TestCase):
    interactions = [
        {"productId": 46, "interactionType": "PRODUCT_SELECT"},
        {"productId": 46, "interactionType": "FITTING_ADD"},
        {"productId": 46, "interactionType": "WISHLIST_ADD"},
        {"productId": 53, "interactionType": "PRODUCT_SELECT"},
        {"productId": 53, "interactionType": "FITTING_ADD"},
        {"productId": 76, "interactionType": "PRODUCT_SELECT"},
        {"productId": 76, "interactionType": "FITTING_ADD"},
        {"productId": 94, "interactionType": "PRODUCT_SELECT"},
        {"productId": 94, "interactionType": "FITTING_ADD"},
        {"productId": 81, "interactionType": "PRODUCT_SELECT"},
        {"productId": 81, "interactionType": "FITTING_ADD"},
    ]

    @classmethod
    def setUpClass(cls):
        store_session_interactions(92001, cls.interactions)

    def test_validation_ranks_seen_products_inside_their_categories(self):
        response = validate_category_rankings(
            CategoryRankingValidationRequest(
                arSessionId=92001,
                productIds=[46, 53, 81, 76, 94],
            )
        )
        rankings = {item["productId"]: item for item in response["anchorRankings"]}
        self.assertEqual("BAG", rankings[46]["category"])
        self.assertEqual("BAG", rankings[53]["category"])
        self.assertEqual("TOP", rankings[76]["category"])
        self.assertEqual("TOP", rankings[81]["category"])
        self.assertEqual("BOTTOM", rankings[94]["category"])
        self.assertEqual(51, rankings[46]["categorySize"])
        self.assertEqual(23, rankings[76]["categorySize"])
        self.assertEqual(20, rankings[94]["categorySize"])
        self.assertTrue(all(item["categoryRank"] >= 1 for item in rankings.values()))

        for category in ("BAG", "TOP", "BOTTOM"):
            category_size = sum(p.category == category for p in recommender.products)
            scored = recommender.recommend(
                self.interactions,
                category=category,
                top_k=category_size,
                exclude_seen=False,
            )
            self.assertEqual(
                sorted((item["score"] for item in scored), reverse=True),
                [item["score"] for item in scored],
            )
            positions = {
                item["productId"]: rank
                for rank, item in enumerate(scored, start=1)
            }
            for item in rankings.values():
                if item["category"] == category:
                    self.assertEqual(positions[item["productId"]], item["categoryRank"])

    def test_validation_http_contract_and_unknown_product(self):
        client = TestClient(app)
        response = client.post(
            "/recommendations/validation",
            json={"arSessionId": 92001, "productIds": [46]},
        )
        self.assertEqual(200, response.status_code)
        self.assertEqual(46, response.json()["anchorRankings"][0]["productId"])

        response = client.post(
            "/recommendations/validation",
            json={"arSessionId": 92001, "productIds": [999999]},
        )
        self.assertEqual(404, response.status_code)
        self.assertEqual([999999], response.json()["detail"]["unknownProductIds"])

    def test_avatar_uses_anchors_and_complements(self):
        response = recommend_avatar_look(AvatarLookRequest(arSessionId=92001))
        selected = response["products"]
        selected_ids = [item["productId"] for item in selected]
        selected_by_category = {
            recommender.product_by_id[product_id].category: product_id
            for product_id in selected_ids
        }
        self.assertEqual(46, selected_by_category["BAG"])
        self.assertEqual(94, selected_by_category["BOTTOM"])
        self.assertIn(selected_by_category["TOP"], {76, 81})
        self.assertIn("SHOES", selected_by_category)
        self.assertIn("ACCESSORIES", selected_by_category)
        self.assertEqual(len(selected_ids), len(set(selected_ids)))

    def test_refresh_accepts_new_fitting_action(self):
        response = refresh_recommendations(
            arSessionId=91002,
            categoryCode="BAG",
            request=RefreshRecommendationRequest(
                interactions=[
                    InteractionRequest(productId=46, interactionType="PRODUCT_SELECT"),
                    InteractionRequest(productId=46, interactionType="FITTING_ADD"),
                ]
            ),
        )
        self.assertEqual(6, len(response["recommendations"]))
        self.assertNotIn(46, [item["productId"] for item in response["recommendations"]])


if __name__ == "__main__":
    unittest.main()
