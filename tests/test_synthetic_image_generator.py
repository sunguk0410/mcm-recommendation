import random

import numpy as np
import pytest

from src.synthetic_image_generator import generate_session, validate_sessions


def fixture():
    products = [{"productId": i + 1, "category": "TOP", "gender": "UNISEX"} for i in range(12)]
    vectors = np.array([[1., 0.] if i < 6 else [0., 1.] for i in range(12)])
    return products, vectors


def generate(seed, count=100):
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    products, vectors = fixture()
    return [generate_session(i, "TOP", products, vectors, rng) for i in range(count)]


def test_reproducible_sessions_and_valid_action_states():
    first = generate(42)
    assert first == generate(42)
    assert first != generate(43)
    counts = validate_sessions(first, fixture()[0])
    assert counts["FITTING_REMOVE"] > 0
    assert counts["WISHLIST_REMOVE"] > 0


def test_visual_preference_affects_choices_without_forcing_nearest():
    sessions = generate(42, 300)
    matches = [((s["anchorProductId"] <= 6) == (s["selectedProductIds"][0] <= 6)) for s in sessions]
    assert .70 < np.mean(matches) < .99


def test_validation_rejects_remove_without_add():
    session = generate(42, 1)[0]
    session["interactions"][1] = {"productId": session["selectedProductIds"][0],
                                   "interactionType": "FITTING_REMOVE", "sequenceNo": 2}
    with pytest.raises(AssertionError):
        validate_sessions([session], fixture()[0])
