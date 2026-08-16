import unittest

from src.avatar_look import (
    select_avatar_look_products,
    select_raw_score_fallback,
)


class SelectAvatarLookProductsTest(unittest.TestCase):
    def test_normal_candidates_keep_existing_category_selection(self):
        products = [
            {"productId": 1, "score": 10.0, "category": "BAG"},
            {"productId": 2, "score": 9.0, "category": "BAG"},
            {"productId": 3, "score": 8.0, "category": "TOP"},
            {"productId": 4, "score": 0.0, "category": "SHOES"},
        ]

        selected = select_avatar_look_products(products)

        self.assertEqual([1, 3], [item["productId"] for item in selected])

    def test_interaction_shortage_fallback_uses_raw_top_three_categories(self):
        products = [
            {"productId": 1, "score": 5.31, "category": "BAG"},
            {"productId": 2, "score": 5.02, "category": "TOP"},
            {"productId": 3, "score": 4.87, "category": "SHOES"},
            {"productId": 4, "score": 4.20, "category": "BOTTOM"},
        ]

        selected = select_raw_score_fallback(products)

        self.assertEqual([1, 2, 3], [item["productId"] for item in selected])

    def test_equal_scores_use_fallback_without_duplicate_categories(self):
        products = [
            {"productId": 11, "score": 0.0, "category": "BAG"},
            {"productId": 12, "score": 0.0, "category": "BAG"},
            {"productId": 13, "score": 0.0, "category": "TOP"},
        ]

        selected = select_avatar_look_products(products)

        self.assertEqual([11, 13], [item["productId"] for item in selected])


if __name__ == "__main__":
    unittest.main()
