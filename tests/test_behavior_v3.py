import random
import unittest

import torch

from src.affinity import Product
from src.dataset import (
    BEHAVIOR_TO_ID,
    ProductIdMapper,
    RawInteraction,
    build_product_episodes,
)
from src.inference import RecRecInference
from src.recrec import RecRec
from src.synthetic_generator_v3 import BEHAVIOR_PROFILES, generate_actions


class BehaviorVocabularyTest(unittest.TestCase):
    def test_v3_mapping_is_exact(self):
        self.assertEqual(
            {
                "PAD": 0,
                "PRODUCT_SELECT": 1,
                "FITTING_ADD": 2,
                "FITTING_REMOVE": 3,
                "WISHLIST_ADD": 4,
                "WISHLIST_REMOVE": 5,
            },
            BEHAVIOR_TO_ID,
        )
        self.assertNotIn("FITTING", BEHAVIOR_TO_ID)

    def test_behavior_embedding_includes_padding_and_five_actions(self):
        model = RecRec(num_products=3)
        self.assertEqual(6, model.behavior_embedding.num_embeddings)


class SequencePreservationTest(unittest.TestCase):
    def test_add_remove_events_are_preserved_in_order(self):
        interactions = [
            RawInteraction(46, "PRODUCT_SELECT", 1),
            RawInteraction(46, "FITTING_ADD", 2),
            RawInteraction(46, "FITTING_REMOVE", 3),
            RawInteraction(46, "WISHLIST_ADD", 4),
            RawInteraction(46, "WISHLIST_REMOVE", 5),
        ]
        episodes = build_product_episodes(interactions)
        self.assertEqual(1, len(episodes))
        self.assertEqual(
            [
                "PRODUCT_SELECT",
                "FITTING_ADD",
                "FITTING_REMOVE",
                "WISHLIST_ADD",
                "WISHLIST_REMOVE",
            ],
            [item.interaction_type for item in episodes[0].interactions],
        )

    def test_generated_remove_actions_always_follow_matching_add(self):
        random.seed(42)
        profile = BEHAVIOR_PROFILES["FASHION_EXPLORER"]
        for _ in range(2000):
            actions = generate_actions(random.random(), profile)
            fitting_active = False
            wishlist_active = False
            self.assertEqual("PRODUCT_SELECT", actions[0])
            for action in actions:
                if action == "FITTING_ADD":
                    self.assertFalse(fitting_active)
                    fitting_active = True
                elif action == "FITTING_REMOVE":
                    self.assertTrue(fitting_active)
                    fitting_active = False
                elif action == "WISHLIST_ADD":
                    self.assertFalse(wishlist_active)
                    wishlist_active = True
                elif action == "WISHLIST_REMOVE":
                    self.assertTrue(wishlist_active)
                    wishlist_active = False


class InferencePreprocessingTest(unittest.TestCase):
    def setUp(self):
        products = [
            Product(46, "P46", "Test", "UNISEX", "BAG", None, "NEW", None, None),
        ]
        self.inference = RecRecInference.__new__(RecRecInference)
        self.inference.device = torch.device("cpu")
        self.inference.mapper = ProductIdMapper(products)

    def test_fitting_add_and_remove_are_encoded(self):
        _, behavior_ids, attention_mask = self.inference._build_input(
            [
                {"productId": 46, "interactionType": "FITTING_ADD"},
                {"productId": 46, "interactionType": "FITTING_REMOVE"},
            ]
        )
        self.assertEqual([2, 3], behavior_ids[attention_mask].tolist())

    def test_legacy_fitting_is_not_a_valid_interaction(self):
        with self.assertRaisesRegex(ValueError, "No valid interactions"):
            self.inference._build_input(
                [{"productId": 46, "interactionType": "FITTING"}]
            )


if __name__ == "__main__":
    unittest.main()
