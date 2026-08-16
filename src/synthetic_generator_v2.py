import json
import math
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


# =========================================================
# Config
# =========================================================

SEED = 42

CATALOG_PATH = (
    "MCM_제품리스트_통합_추천모델용.xlsx"
)

OUTPUT_PATH = (
    "synthetic_interactions_v2.jsonl"
)

NUM_SESSIONS = 9000


# =========================================================
# Behavior Profiles
# =========================================================

BEHAVIOR_PROFILES = {

    "FASHION_EXPLORER": {
        "lambda_products": 13,
        "min_products": 8,
        "max_products": 20,

        "fitting_base": 0.45,
        "wishlist_add_base": 0.28,
        "wishlist_remove_base": 0.12,
    },

    "DECISIVE_BUYER": {
        "lambda_products": 8,
        "min_products": 5,
        "max_products": 14,

        "fitting_base": 0.78,
        "wishlist_add_base": 0.58,
        "wishlist_remove_base": 0.06,
    },

    "BROAD_EXPLORER": {
        "lambda_products": 18,
        "min_products": 10,
        "max_products": 28,

        "fitting_base": 0.18,
        "wishlist_add_base": 0.07,
        "wishlist_remove_base": 0.24,
    },
}


# =========================================================
# Zone Profiles
# =========================================================

ZONE_PROFILES = {

    "CLASSIC": {
        "CLASSIC": 0.65,
        "NEW": 0.25,
        "TRAVEL": 0.10,
    },

    "NEW": {
        "CLASSIC": 0.20,
        "NEW": 0.65,
        "TRAVEL": 0.15,
    },

    "TRAVEL": {
        "CLASSIC": 0.15,
        "NEW": 0.20,
        "TRAVEL": 0.65,
    },
}


# =========================================================
# Persona
# =========================================================

PERSONAS = [

    ("P1", "FASHION_EXPLORER", "CLASSIC"),
    ("P2", "FASHION_EXPLORER", "NEW"),
    ("P3", "FASHION_EXPLORER", "TRAVEL"),

    ("P4", "DECISIVE_BUYER", "CLASSIC"),
    ("P5", "DECISIVE_BUYER", "NEW"),
    ("P6", "DECISIVE_BUYER", "TRAVEL"),

    ("P7", "BROAD_EXPLORER", "CLASSIC"),
    ("P8", "BROAD_EXPLORER", "NEW"),
    ("P9", "BROAD_EXPLORER", "TRAVEL"),
]


# =========================================================
# Preference Update Strength
# =========================================================
#
# 여기서 v2의 핵심이 발생한다.
#
# SELECT
#   → 약한 관심
#
# FITTING
#   → 중간 수준 긍정
#
# WISHLIST_ADD
#   → 강한 긍정
#
# WISHLIST_REMOVE
#   → 부정 preference correction
#
# =========================================================

ACTION_STRENGTH = {

    "PRODUCT_SELECT": 0.15,

    "FITTING": 0.35,

    "WISHLIST_ADD": 0.60,

    "WISHLIST_REMOVE": -0.70,
}


# =========================================================
# Preference Memory
# =========================================================

PREFERENCE_DECAY = 0.97

MAX_DYNAMIC_PREFERENCE = 3.0

MIN_DYNAMIC_PREFERENCE = -3.0


# =========================================================
# Sampling
# =========================================================

SAMPLING_TEMPERATURE = 0.35


# =========================================================
# Helpers
# =========================================================

def sigmoid(x):

    return (
        1.0
        / (1.0 + math.exp(-x))
    )


def logit(p):

    p = min(
        max(p, 1e-6),
        1 - 1e-6,
    )

    return math.log(
        p / (1 - p)
    )


def softmax(values, temperature=1.0):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = (
        values
        / temperature
    )

    values = (
        values
        - np.max(values)
    )

    exp_values = np.exp(
        values
    )

    return (
        exp_values
        / exp_values.sum()
    )


def safe_str(value):

    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    return value


# =========================================================
# Load Products
# =========================================================

def load_products():

    df = pd.read_excel(
        CATALOG_PATH,
        sheet_name="Products",
    )

    products = []

    for _, row in df.iterrows():

        products.append({

            "productId": int(
                row["productId"]
            ),

            "productCode": safe_str(
                row["productCode"]
            ),

            "name": safe_str(
                row["name"]
            ),

            "gender": safe_str(
                row["gender"]
            ),

            "category": safe_str(
                row["category"]
            ),

            "subCategory": safe_str(
                row["subCategory"]
            ),

            "zone": safe_str(
                row["zone"]
            ),

            "color": safe_str(
                row["color"]
            ),
        })

    return products


# =========================================================
# Hidden Base Preference
# =========================================================

def create_base_preferences(
    products,
    zone_orientation,
):

    categories = sorted({
        product["category"]
        for product in products
        if product["category"]
    })

    category_values = np.random.dirichlet(
        np.ones(
            len(categories)
        ) * 0.7
    )

    category_pref = {
        category: float(value)
        for category, value
        in zip(
            categories,
            category_values,
        )
    }

    subcategory_pref = {}

    for category in categories:

        subcategories = sorted({

            product["subCategory"]

            for product in products

            if (
                product["category"]
                == category
                and product["subCategory"]
            )
        })

        if not subcategories:
            continue

        values = np.random.dirichlet(
            np.ones(
                len(subcategories)
            ) * 0.6
        )

        for subcategory, value in zip(
            subcategories,
            values,
        ):

            subcategory_pref[
                (
                    category,
                    subcategory,
                )
            ] = float(value)

    zone_pref = dict(
        ZONE_PROFILES[
            zone_orientation
        ]
    )

    return (
        category_pref,
        subcategory_pref,
        zone_pref,
    )


# =========================================================
# Dynamic Preference State
# =========================================================

def create_dynamic_state():

    return {
        "category": defaultdict(float),
        "subcategory": defaultdict(float),
        "zone": defaultdict(float),
    }


def decay_dynamic_state(
    dynamic_state,
):

    for state_name in (
        "category",
        "subcategory",
        "zone",
    ):

        state = dynamic_state[
            state_name
        ]

        for key in list(
            state.keys()
        ):

            state[key] *= (
                PREFERENCE_DECAY
            )


def clamp(value):

    return max(
        MIN_DYNAMIC_PREFERENCE,
        min(
            MAX_DYNAMIC_PREFERENCE,
            value,
        ),
    )


# =========================================================
# Product Scoring
# =========================================================

def calculate_base_score(
    product,
    category_pref,
    subcategory_pref,
    zone_pref,
):

    zone_score = zone_pref.get(
        product["zone"],
        0.0,
    )

    category_score = (
        category_pref.get(
            product["category"],
            0.0,
        )
    )

    subcategory = product[
        "subCategory"
    ]

    if subcategory:

        subcategory_score = (
            subcategory_pref.get(
                (
                    product["category"],
                    subcategory,
                ),
                0.0,
            )
        )

        score = (
            0.45 * zone_score
            + 0.35 * category_score
            + 0.20 * subcategory_score
        )

    else:

        # subCategory가 없으면
        # zone/category weight 재정규화
        score = (
            (0.45 / 0.80)
            * zone_score

            + (0.35 / 0.80)
            * category_score
        )

    return score


def calculate_dynamic_score(
    product,
    dynamic_state,
):

    category_score = (
        dynamic_state[
            "category"
        ].get(
            product["category"],
            0.0,
        )
    )

    zone_score = (
        dynamic_state[
            "zone"
        ].get(
            product["zone"],
            0.0,
        )
    )

    subcategory_score = 0.0

    if product["subCategory"]:

        subcategory_score = (
            dynamic_state[
                "subcategory"
            ].get(
                (
                    product["category"],
                    product["subCategory"],
                ),
                0.0,
            )
        )

    return (
        0.40 * category_score
        + 0.25 * zone_score
        + 0.35 * subcategory_score
    )


def calculate_product_score(
    product,
    category_pref,
    subcategory_pref,
    zone_pref,
    dynamic_state,
):

    base_score = (
        calculate_base_score(
            product=product,
            category_pref=category_pref,
            subcategory_pref=(
                subcategory_pref
            ),
            zone_pref=zone_pref,
        )
    )

    dynamic_score = (
        calculate_dynamic_score(
            product=product,
            dynamic_state=(
                dynamic_state
            ),
        )
    )

    # base preference가 여전히 주축이지만
    # 행동을 통해 형성된 preference도
    # 다음 상품 선택에 직접 영향
    return (
        base_score
        + 0.35 * dynamic_score
    )


# =========================================================
# Product Sampling
# =========================================================

def sample_next_product(
    candidates,
    category_pref,
    subcategory_pref,
    zone_pref,
    dynamic_state,
):

    scores = []

    for product in candidates:

        score = (
            calculate_product_score(
                product=product,
                category_pref=category_pref,
                subcategory_pref=(
                    subcategory_pref
                ),
                zone_pref=zone_pref,
                dynamic_state=(
                    dynamic_state
                ),
            )
        )

        scores.append(
            score
        )

    # min-max normalization
    minimum = min(
        scores
    )

    maximum = max(
        scores
    )

    if maximum > minimum:

        normalized = [
            (
                score
                - minimum
            )
            / (
                maximum
                - minimum
            )
            for score in scores
        ]

    else:

        normalized = [
            0.5
            for _ in scores
        ]

    probabilities = softmax(
        normalized,
        temperature=(
            SAMPLING_TEMPERATURE
        ),
    )

    index = np.random.choice(
        len(candidates),
        p=probabilities,
    )

    return (
        candidates[index],
        normalized[index],
    )


# =========================================================
# Behavior Probability
# =========================================================

def perturb_probability(
    probability,
):

    perturbed_logit = (
        logit(probability)
        + np.random.normal(
            0.0,
            0.30,
        )
    )

    return sigmoid(
        perturbed_logit
    )


def generate_actions(
    affinity,
    behavior_profile,
):

    fitting_base = perturb_probability(
        behavior_profile[
            "fitting_base"
        ]
    )

    wishlist_base = perturb_probability(
        behavior_profile[
            "wishlist_add_base"
        ]
    )

    remove_base = perturb_probability(
        behavior_profile[
            "wishlist_remove_base"
        ]
    )

    fitting_probability = sigmoid(
        logit(fitting_base)
        + 2.0 * (
            affinity - 0.5
        )
    )

    fitted = (
        random.random()
        < fitting_probability
    )

    wishlist_probability = sigmoid(
        logit(wishlist_base)
        + 2.5 * (
            affinity - 0.5
        )
        + (
            0.70
            if fitted
            else 0.0
        )
    )

    wishlist_added = (
        random.random()
        < wishlist_probability
    )

    wishlist_removed = False

    if wishlist_added:

        remove_probability = sigmoid(
            logit(remove_base)
            - 2.0 * (
                affinity - 0.5
            )
        )

        wishlist_removed = (
            random.random()
            < remove_probability
        )

    actions = [
        "PRODUCT_SELECT"
    ]

    if fitted:

        actions.append(
            "FITTING"
        )

    if wishlist_added:

        actions.append(
            "WISHLIST_ADD"
        )

    if wishlist_removed:

        actions.append(
            "WISHLIST_REMOVE"
        )

    return actions


# =========================================================
# Preference Update
# =========================================================

def update_preference(
    dynamic_state,
    product,
    action,
):

    strength = (
        ACTION_STRENGTH[
            action
        ]
    )

    # category
    category = product[
        "category"
    ]

    dynamic_state[
        "category"
    ][category] = clamp(

        dynamic_state[
            "category"
        ][category]

        + strength
    )

    # zone
    zone = product[
        "zone"
    ]

    dynamic_state[
        "zone"
    ][zone] = clamp(

        dynamic_state[
            "zone"
        ][zone]

        + 0.70 * strength
    )

    # subCategory
    subcategory = product[
        "subCategory"
    ]

    if subcategory:

        key = (
            category,
            subcategory,
        )

        dynamic_state[
            "subcategory"
        ][key] = clamp(

            dynamic_state[
                "subcategory"
            ][key]

            + 1.20 * strength
        )


# =========================================================
# Session Length
# =========================================================

def sample_num_products(
    profile,
):

    count = np.random.poisson(
        profile[
            "lambda_products"
        ]
    )

    return int(
        max(
            profile[
                "min_products"
            ],
            min(
                profile[
                    "max_products"
                ],
                count,
            ),
        )
    )


# =========================================================
# Generate Session
# =========================================================

def generate_session(
    session_index,
    products,
):

    (
        persona_id,
        behavior_type,
        zone_orientation,
    ) = random.choice(
        PERSONAS
    )

    behavior_profile = (
        BEHAVIOR_PROFILES[
            behavior_type
        ]
    )

    gender = random.choice(
        [
            "MALE",
            "FEMALE",
        ]
    )

    candidates = [

        product

        for product in products

        if (
            product["gender"]
            == gender

            or product["gender"]
            == "UNISEX"
        )
    ]

    (
        category_pref,
        subcategory_pref,
        zone_pref,
    ) = create_base_preferences(
        products=candidates,
        zone_orientation=(
            zone_orientation
        ),
    )

    dynamic_state = (
        create_dynamic_state()
    )

    num_products = (
        sample_num_products(
            behavior_profile
        )
    )

    num_products = min(
        num_products,
        len(candidates),
    )

    available_products = list(
        candidates
    )

    interactions = []

    sequence_no = 1

    selected_product_ids = []

    # =====================================================
    # 핵심:
    # 상품 하나 선택 → 행동 → preference update
    # → 그 다음 상품 선택
    # =====================================================

    for _ in range(
        num_products
    ):

        if not available_products:
            break

        # 이전 행동까지 반영된
        # 현재 preference state로
        # 다음 상품을 선택
        (
            product,
            affinity,
        ) = sample_next_product(

            candidates=(
                available_products
            ),

            category_pref=(
                category_pref
            ),

            subcategory_pref=(
                subcategory_pref
            ),

            zone_pref=(
                zone_pref
            ),

            dynamic_state=(
                dynamic_state
            ),
        )

        selected_product_ids.append(
            product[
                "productId"
            ]
        )

        # 같은 상품 중복 선택 방지
        available_products = [

            candidate

            for candidate
            in available_products

            if (
                candidate[
                    "productId"
                ]
                != product[
                    "productId"
                ]
            )
        ]

        actions = generate_actions(
            affinity=affinity,
            behavior_profile=(
                behavior_profile
            ),
        )

        # ---------------------------------------------
        # 해당 상품에 대한 모든 행동 기록
        # ---------------------------------------------

        for action in actions:

            interactions.append({

                "productId": (
                    product[
                        "productId"
                    ]
                ),

                "interactionType": (
                    action
                ),

                "sequenceNo": (
                    sequence_no
                ),
            })

            sequence_no += 1

            # -----------------------------------------
            # 이 행동으로 preference 변경
            # -----------------------------------------

            update_preference(
                dynamic_state=(
                    dynamic_state
                ),
                product=product,
                action=action,
            )

        # 오래된 선호는 조금씩 감소
        decay_dynamic_state(
            dynamic_state
        )

    return {

        "sessionId": (
            session_index
        ),

        # 아래 metadata는 생성 검증용.
        # RecRec dataset에서는 사용하지 않는다.
        "personaId": (
            persona_id
        ),

        "behaviorType": (
            behavior_type
        ),

        "zoneOrientation": (
            zone_orientation
        ),

        "gender": gender,

        "selectedProductIds": (
            selected_product_ids
        ),

        "interactions": (
            interactions
        ),
    }


# =========================================================
# Main
# =========================================================

def main():

    random.seed(
        SEED
    )

    np.random.seed(
        SEED
    )

    products = (
        load_products()
    )

    print(
        f"Loaded products: "
        f"{len(products)}"
    )

    sessions = []

    interaction_count = 0

    persona_counter = Counter()

    behavior_counter = Counter()

    for session_index in range(
        1,
        NUM_SESSIONS + 1,
    ):

        session = generate_session(
            session_index=(
                session_index
            ),
            products=products,
        )

        sessions.append(
            session
        )

        interaction_count += len(
            session[
                "interactions"
            ]
        )

        persona_counter[
            session["personaId"]
        ] += 1

        for interaction in session[
            "interactions"
        ]:

            behavior_counter[
                interaction[
                    "interactionType"
                ]
            ] += 1

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        for session in sessions:

            f.write(
                json.dumps(
                    session,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print()
    print(
        "================================="
    )

    print(
        "Synthetic Generator V2"
    )

    print(
        "================================="
    )

    print(
        f"Generated sessions: "
        f"{len(sessions)}"
    )

    print(
        f"Generated interactions: "
        f"{interaction_count}"
    )

    print(
        f"Saved: {OUTPUT_PATH}"
    )

    print()
    print(
        "Behavior counts:"
    )

    for action, count in (
        behavior_counter.most_common()
    ):

        print(
            f"  {action}: "
            f"{count}"
        )

    print()
    print(
        "Persona counts:"
    )

    for persona_id in sorted(
        persona_counter
    ):

        print(
            f"  {persona_id}: "
            f"{persona_counter[persona_id]}"
        )


if __name__ == "__main__":
    main()