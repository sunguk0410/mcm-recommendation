import unittest

from src.avatar_look import (
    select_avatar_look_products,
)


class SelectAvatarLookProductsTest(unittest.TestCase):
    def test_selects_highest_direct_interest_in_each_category(self):
        products = [
            {"productId": 1, "directInterestScore": 10.0, "category": "BAG"},
            {"productId": 2, "directInterestScore": 9.0, "category": "BAG"},
            {"productId": 3, "directInterestScore": 8.0, "category": "TOP"},
            {"productId": 4, "directInterestScore": 1.0, "category": "SHOES"},
        ]

        selected = select_avatar_look_products(products)

        self.assertEqual([1, 3, 4], [item["productId"] for item in selected])

    def test_does_not_limit_number_of_categories(self):
        products = [
            {"productId": 1, "directInterestScore": 5.0, "category": "BAG"},
            {"productId": 2, "directInterestScore": 4.0, "category": "TOP"},
            {"productId": 3, "directInterestScore": 3.0, "category": "SHOES"},
            {"productId": 4, "directInterestScore": 2.0, "category": "BOTTOM"},
            {"productId": 5, "directInterestScore": 1.0, "category": "ACCESSORIES"},
        ]

        selected = select_avatar_look_products(products)

        self.assertEqual([1, 2, 3, 4, 5], [item["productId"] for item in selected])

    def test_equal_scores_use_recency_then_product_id(self):
        products = [
            {
                "productId": 11,
                "directInterestScore": 3.0,
                "latestInteractionPosition": 1,
                "category": "BAG",
            },
            {
                "productId": 12,
                "directInterestScore": 3.0,
                "latestInteractionPosition": 2,
                "category": "BAG",
            },
            {
                "productId": 13,
                "directInterestScore": 3.0,
                "latestInteractionPosition": 0,
                "category": "TOP",
            },
        ]

        selected = select_avatar_look_products(products)

        self.assertEqual([12, 13], [item["productId"] for item in selected])


if __name__ == "__main__":
    unittest.main()
