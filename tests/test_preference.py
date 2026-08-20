from types import SimpleNamespace

from src.inference import RecRecInference
from src.preference import build_zone_preferences


def test_zone_preferences_preserve_share_of_total_dwell():
    interactions = [
        SimpleNamespace(category="BAG", zone="TRAVEL", dwellSeconds=60),
        SimpleNamespace(category="BAG", zone="CLASSIC", dwellSeconds=20),
        SimpleNamespace(category="TOP", zone="NEW_COLLECTION", dwellSeconds=20),
    ]

    scores = build_zone_preferences(interactions)

    assert scores["BAG"]["TRAVEL"] == 0.6
    assert scores["BAG"]["CLASSIC"] == 0.2
    assert scores["TOP"]["NEW"] == 0.2


def test_rec_weight_increases_gradually():
    assert RecRecInference._rec_weight(2) == 0.70
    assert RecRecInference._rec_weight(3) == 0.75
    assert RecRecInference._rec_weight(6) == 0.80


def test_diverse_positions_limit_top_five_to_two_per_category():
    products = [
        SimpleNamespace(product_id=index, category=category)
        for index, category in enumerate(
            ["BAG", "BAG", "BAG", "BAG", "TOP", "TOP", "SHOES"],
            start=1,
        )
    ]
    scores = [1.0, 0.99, 0.98, 0.97, 0.90, 0.89, 0.88]

    positions = RecRecInference._diverse_positions(products, scores, 7)
    top_five_categories = [products[position].category for position in positions[:5]]

    assert top_five_categories.count("BAG") == 2
    assert top_five_categories.count("TOP") == 2
    assert top_five_categories.count("SHOES") == 1


def test_content_bonus_recognizes_collection_without_double_counting_wishlist():
    recommender = RecRecInference.__new__(RecRecInference)
    reference = SimpleNamespace(
        product_id=46,
        name="MCM x ATEEZ Ella 보스턴",
        category="BAG",
        sub_category=None,
        color="Cognac",
        zone="NEW",
    )
    candidate = SimpleNamespace(
        product_id=81,
        name="MCM x ATEEZ with Mingi 코튼 로고 티셔츠",
        category="TOP",
        sub_category=None,
        color="Black",
        zone="NEW",
    )
    recommender.product_by_id = {46: reference, 81: candidate}

    related_bonus = recommender._content_bonus(candidate, [], [46])
    exact_wishlist_bonus = recommender._content_bonus(reference, [], [46])

    assert related_bonus >= 0.07
    assert exact_wishlist_bonus == 0.0


def test_wishlist_products_are_treated_as_seen_candidates():
    excluded = RecRecInference._excluded_product_ids(
        [{"productId": 1, "interactionType": "FITTING"}],
        [2],
        True,
    )

    assert excluded == {1, 2}


def test_recommend_filters_candidates_by_gender_and_keeps_unisex():
    recommender = RecRecInference.__new__(RecRecInference)
    recommender.products = [
        SimpleNamespace(product_id=1, category="SHOES", gender="MALE"),
        SimpleNamespace(product_id=2, category="SHOES", gender="FEMALE"),
        SimpleNamespace(product_id=3, category="SHOES", gender="UNISEX"),
    ]
    recommender._initial_preference_score = lambda product, scores: float(product.product_id)

    recommendations = recommender.recommend(
        interactions=[],
        category="SHOES",
        gender="female",
        top_k=6,
    )

    assert [item["productId"] for item in recommendations] == [3, 2]


def test_recommend_does_not_filter_bags_by_gender():
    recommender = RecRecInference.__new__(RecRecInference)
    recommender.products = [
        SimpleNamespace(product_id=1, category="BAG", gender="MALE"),
        SimpleNamespace(product_id=2, category="BAG", gender="FEMALE"),
        SimpleNamespace(product_id=3, category="BAG", gender="UNISEX"),
    ]
    recommender._initial_preference_score = lambda product, scores: float(product.product_id)

    recommendations = recommender.recommend(
        interactions=[],
        category="BAG",
        gender="FEMALE",
        top_k=6,
    )

    assert [item["productId"] for item in recommendations] == [3, 2, 1]


def test_recommend_prioritizes_product_100_only_on_cold_start():
    recommender = RecRecInference.__new__(RecRecInference)
    recommender.products = [
        SimpleNamespace(product_id=1, category="BOTTOM", gender="FEMALE"),
        SimpleNamespace(product_id=100, category="BOTTOM", gender="FEMALE"),
        SimpleNamespace(product_id=105, category="BOTTOM", gender="FEMALE"),
    ]
    recommender._initial_preference_score = lambda product, scores: scores.get(
        product.product_id,
        0.0,
    )

    cold_start = recommender.recommend(
        interactions=[],
        category="BOTTOM",
        gender="FEMALE",
        top_k=2,
    )
    with_zone_history = recommender.recommend(
        interactions=[],
        zone_scores={1: 1.0, 100: 0.0},
        category="BOTTOM",
        gender="FEMALE",
        top_k=2,
    )

    assert [item["productId"] for item in cold_start] == [100, 105]
    assert [item["productId"] for item in with_zone_history] == [1, 100]


def test_recommend_prioritizes_bag_products_in_configured_order():
    recommender = RecRecInference.__new__(RecRecInference)
    recommender.products = [
        SimpleNamespace(product_id=53, category="BAG", gender="MALE"),
        SimpleNamespace(product_id=4, category="BAG", gender="FEMALE"),
        SimpleNamespace(product_id=62, category="BAG", gender="UNISEX"),
        SimpleNamespace(product_id=44, category="BAG", gender="FEMALE"),
        SimpleNamespace(product_id=75, category="BAG", gender="MALE"),
        SimpleNamespace(product_id=63, category="BAG", gender="FEMALE"),
    ]
    recommender._initial_preference_score = lambda product, scores: 0.0

    recommendations = recommender.recommend(
        interactions=[],
        category="BAG",
        gender="FEMALE",
        top_k=6,
    )

    assert [item["productId"] for item in recommendations] == [4, 44, 75, 62, 63, 53]


def test_recommend_prioritizes_top_products_in_configured_order():
    recommender = RecRecInference.__new__(RecRecInference)
    recommender.products = [
        SimpleNamespace(product_id=1, category="TOP", gender="FEMALE"),
        SimpleNamespace(product_id=81, category="TOP", gender="FEMALE"),
        SimpleNamespace(product_id=80, category="TOP", gender="FEMALE"),
    ]
    recommender._initial_preference_score = lambda product, scores: 0.0

    recommendations = recommender.recommend(
        interactions=[],
        category="TOP",
        gender="FEMALE",
        top_k=3,
    )

    assert [item["productId"] for item in recommendations] == [80, 81, 1]
