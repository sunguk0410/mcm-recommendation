import unittest
from types import SimpleNamespace

from src.direct_interest import score_direct_interest


class DirectInterestTest(unittest.TestCase):
    def setUp(self):
        self.products = [
            SimpleNamespace(product_id=70, category="BAG", zone="TRAVEL"),
            SimpleNamespace(product_id=50, category="BAG", zone="TRAVEL"),
            SimpleNamespace(product_id=91, category="TOP", zone="NEW"),
            SimpleNamespace(product_id=12, category="SHOES", zone="CLASSIC"),
        ]

    def test_p1_repeated_fitting_makes_70_strongest_interest(self):
        ranked = score_direct_interest(
            products=self.products,
            online_wishlist_product_ids=[70, 50],
            interactions=[
                {"productId": 70, "interactionType": "PRODUCT_SELECT"},
                {"productId": 70, "interactionType": "FITTING"},
                {"productId": 50, "interactionType": "FITTING"},
                {"productId": 91, "interactionType": "FITTING"},
                {"productId": 70, "interactionType": "FITTING"},
            ],
        )

        self.assertEqual(70, ranked[0]["productId"])
        self.assertGreater(
            ranked[0]["directInterestScore"],
            ranked[1]["directInterestScore"],
        )

    def test_repeated_fitting_has_diminishing_returns(self):
        once = score_direct_interest(
            self.products,
            [],
            [{"productId": 70, "interactionType": "FITTING"}],
        )[0]["directInterestScore"]
        twice = score_direct_interest(
            self.products,
            [],
            [
                {"productId": 70, "interactionType": "FITTING"},
                {"productId": 70, "interactionType": "FITTING"},
            ],
        )[0]["directInterestScore"]

        self.assertGreater(twice, once)
        self.assertLess(twice, once * 2)

    def test_wishlist_remove_cancels_online_wishlist(self):
        ranked = score_direct_interest(
            self.products,
            [70],
            [{"productId": 70, "interactionType": "WISHLIST_REMOVE"}],
        )

        self.assertEqual([], ranked)

    def test_latest_wishlist_state_wins(self):
        ranked = score_direct_interest(
            self.products,
            [],
            [
                {"productId": 70, "interactionType": "WISHLIST_ADD"},
                {"productId": 70, "interactionType": "WISHLIST_REMOVE"},
                {"productId": 70, "interactionType": "WISHLIST_ADD"},
            ],
        )

        self.assertEqual(70, ranked[0]["productId"])
        self.assertEqual(5.0, ranked[0]["directInterestScore"])

    def test_ignores_unknown_products_and_actions(self):
        ranked = score_direct_interest(
            self.products,
            [],
            [
                {"productId": 999, "interactionType": "FITTING"},
                {"productId": 70, "interactionType": "UNKNOWN"},
            ],
        )

        self.assertEqual([], ranked)


if __name__ == "__main__":
    unittest.main()
