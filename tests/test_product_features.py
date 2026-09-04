from types import SimpleNamespace

import torch

from src.product_features import build_product_feature_matrix
from src.recrec import RecRec


class Mapper:
    num_products = 3

    @staticmethod
    def encode(product_id):
        return product_id


def product(product_id, name, color):
    return SimpleNamespace(
        product_id=product_id,
        name=name,
        gender="UNISEX",
        category="BAG",
        sub_category="SHOULDER",
        zone="CLASSIC",
        color=color,
    )


def test_metadata_features_are_deterministic_and_keep_pad_zero():
    products = [product(1, "Aren Bag", "BLACK"), product(2, "Aren Bag", "RED")]
    first = build_product_feature_matrix(products, Mapper(), feature_dim=64)
    second = build_product_feature_matrix(products, Mapper(), feature_dim=64)

    assert torch.equal(first, second)
    assert torch.equal(first[0], torch.zeros(64))
    assert not torch.equal(first[1], first[2])


def test_history_and_candidates_use_same_product_encoder():
    features = torch.zeros(3, 4)
    features[1, 0] = 1.0
    model = RecRec(
        num_products=3,
        embedding_dim=4,
        hidden_dim=8,
        num_refinement_steps=1,
        num_inner_steps=1,
        recursive_layers=2,
        metadata_features=features,
        dropout=0.0,
    )

    encoded_history = model.encode_products(torch.tensor([[1]]))[0, 0]
    encoded_candidate = model.encode_products(torch.arange(3))[1]

    assert torch.allclose(encoded_history, encoded_candidate)
