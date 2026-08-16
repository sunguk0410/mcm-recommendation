import unittest

from src.avatar_look import (
    calculate_product_active_states,
    explicitly_removed_without_positive_state,
    select_category_anchors,
)
from src.affinity import Product


class AvatarActiveStateTest(unittest.TestCase):
    def state_for(self, *actions):
        interactions = [
            {"productId": 46, "interactionType": action}
            for action in actions
        ]
        return calculate_product_active_states(interactions)[46]

    def test_fitting_state_transitions(self):
        self.assertTrue(self.state_for("FITTING_ADD")["fittingActive"])
        self.assertFalse(
            self.state_for("FITTING_ADD", "FITTING_REMOVE")["fittingActive"]
        )
        self.assertTrue(
            self.state_for(
                "FITTING_ADD", "FITTING_REMOVE", "FITTING_ADD"
            )["fittingActive"]
        )

    def test_wishlist_state_transitions(self):
        self.assertTrue(self.state_for("WISHLIST_ADD")["wishlistActive"])
        self.assertFalse(
            self.state_for("WISHLIST_ADD", "WISHLIST_REMOVE")["wishlistActive"]
        )
        self.assertTrue(
            self.state_for(
                "WISHLIST_ADD", "WISHLIST_REMOVE", "WISHLIST_ADD"
            )["wishlistActive"]
        )

    def test_remove_does_not_clear_independent_positive_state(self):
        state = self.state_for(
            "PRODUCT_SELECT",
            "FITTING_ADD",
            "WISHLIST_ADD",
            "WISHLIST_REMOVE",
        )
        self.assertFalse(state["wishlistActive"])
        self.assertTrue(state["fittingActive"])
        self.assertTrue(state["productSelected"])

    def test_fully_removed_non_selected_product_is_excluded_from_complement(self):
        states = calculate_product_active_states([
            {"productId": 46, "interactionType": "WISHLIST_ADD"},
            {"productId": 46, "interactionType": "WISHLIST_REMOVE"},
        ])
        self.assertEqual({46}, explicitly_removed_without_positive_state(states))


class AvatarAnchorSelectionTest(unittest.TestCase):
    def setUp(self):
        self.products = {
            product_id: Product(
                product_id,
                f"P{product_id}",
                f"Product {product_id}",
                "UNISEX",
                "BAG",
                None,
                "NEW",
                None,
                None,
            )
            for product_id in (46, 53, 54)
        }

    def select(self, interactions, scores):
        states = calculate_product_active_states(interactions)
        return select_category_anchors(states, self.products, scores)["BAG"]

    def test_priority_is_wishlist_then_fitting_then_select(self):
        selected = self.select(
            [
                {"productId": 46, "interactionType": "WISHLIST_ADD"},
                {"productId": 53, "interactionType": "FITTING_ADD"},
                {"productId": 54, "interactionType": "PRODUCT_SELECT"},
            ],
            {46: 1.0, 53: 100.0, 54: 200.0},
        )
        self.assertEqual(46, selected["productId"])

    def test_equal_priority_uses_recrec_score_before_recency(self):
        selected = self.select(
            [
                {"productId": 46, "interactionType": "FITTING_ADD"},
                {"productId": 53, "interactionType": "FITTING_ADD"},
            ],
            {46: 5.0, 53: 4.0},
        )
        self.assertEqual(46, selected["productId"])

    def test_equal_priority_and_score_uses_recent_positive_interaction(self):
        selected = self.select(
            [
                {"productId": 46, "interactionType": "FITTING_ADD"},
                {"productId": 53, "interactionType": "FITTING_ADD"},
            ],
            {46: 5.0, 53: 5.0},
        )
        self.assertEqual(53, selected["productId"])


if __name__ == "__main__":
    unittest.main()
