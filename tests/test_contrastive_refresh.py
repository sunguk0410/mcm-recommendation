import unittest

from src.contrastive_refresh import (
    has_new_interactions,
    select_contrastive_products,
)


def products(count):
    return [
        {"productId": index + 1, "score": float(count - index)}
        for index in range(count)
    ]


class ContrastiveRefreshTest(unittest.TestCase):
    def test_new_interaction_keeps_rerank_branch(self):
        previous = [{"productId": 1, "interactionType": "VIEW"}]
        current = previous + [{"productId": 2, "interactionType": "LIKE"}]

        self.assertTrue(has_new_interactions(current, previous))

    def test_reordered_interactions_are_not_treated_as_new(self):
        previous = [
            {"productId": 1, "interactionType": "VIEW"},
            {"productId": 2, "interactionType": "LIKE"},
        ]

        self.assertFalse(has_new_interactions(list(reversed(previous)), previous))

    def test_no_new_interaction_selects_middle_score_band(self):
        scored = products(20)

        selected = select_contrastive_products(scored, set())

        self.assertEqual([7, 8, 9, 10, 11, 12], [p["productId"] for p in selected])

    def test_previous_products_are_never_reused(self):
        previous_ids = {7, 8, 9, 10, 11, 12}

        selected = select_contrastive_products(products(20), previous_ids)

        self.assertTrue(previous_ids.isdisjoint(p["productId"] for p in selected))
        self.assertEqual(6, len(selected))

    def test_category_with_fewer_than_six_remaining_candidates(self):
        selected = select_contrastive_products(products(8), {1, 2, 3, 4})

        self.assertEqual(4, len(selected))
        self.assertEqual({5, 6, 7, 8}, {p["productId"] for p in selected})

    def test_very_small_category_returns_only_available_products(self):
        selected = select_contrastive_products(products(3), {1, 2})

        self.assertEqual([3], [p["productId"] for p in selected])


if __name__ == "__main__":
    unittest.main()
