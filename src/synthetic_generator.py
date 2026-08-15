import json
import math
from dataclasses import dataclass, asdict

import numpy as np

from persona import (
    PERSONAS,
    BEHAVIOR_PROFILES,
    Persona,
)

from affinity import (
    Product,
    load_products_from_excel,
    filter_products_by_gender,
    create_hidden_preference,
    calculate_normalized_affinities,
)


# =========================================================
# Interaction Types
# =========================================================

PRODUCT_SELECT = "PRODUCT_SELECT"
FITTING = "FITTING"
WISHLIST_ADD = "WISHLIST_ADD"
WISHLIST_REMOVE = "WISHLIST_REMOVE"


@dataclass
class Interaction:
    productId: int
    interactionType: str
    sequenceNo: int


@dataclass
class SyntheticSession:
    sessionId: int
    interactions: list[Interaction]


# =========================================================
# Math
# =========================================================

def clip_probability(
    probability: float,
) -> float:

    return float(
        np.clip(
            probability,
            0.01,
            0.99,
        )
    )


def logit(
    probability: float,
) -> float:

    probability = (
        clip_probability(
            probability
        )
    )

    return math.log(
        probability
        / (1.0 - probability)
    )


def sigmoid(
    value: float,
) -> float:

    return (
        1.0
        / (
            1.0
            + math.exp(-value)
        )
    )


# =========================================================
# Session-specific behavior tendency
# =========================================================

def perturb_probability(
    base: float,
    rng: np.random.Generator,
    sigma: float = 0.30,
) -> float:
    """
    probability 자체에 noise를 더하는 것보다
    logit 공간에서 noise를 추가하는 것이 안정적이다.
    """

    noisy_logit = (
        logit(base)
        + rng.normal(
            0.0,
            sigma,
        )
    )

    return clip_probability(
        sigmoid(
            noisy_logit
        )
    )


def behavior_probability(
    base: float,
    affinity: float,
    affinity_strength: float,
    extra_bias: float = 0.0,
) -> float:

    value = (
        logit(base)

        + affinity_strength
        * (
            affinity - 0.5
        )

        + extra_bias
    )

    return clip_probability(
        sigmoid(value)
    )


# =========================================================
# Number of Products
# =========================================================

def sample_num_products(
    persona: Persona,
    available_products: int,
    rng: np.random.Generator,
) -> int:

    profile = (
        BEHAVIOR_PROFILES[
            persona.behavior_type
        ]
    )

    count = int(
        rng.poisson(
            profile.lambda_products
        )
    )

    count = int(
        np.clip(
            count,
            profile.min_products,
            profile.max_products,
        )
    )

    return min(
        count,
        available_products,
    )


# =========================================================
# Product Sampling
# =========================================================

def sample_products(
    products: list[Product],
    affinities: dict[int, float],
    count: int,
    rng: np.random.Generator,
    temperature: float = 0.35,
) -> list[Product]:

    if not products:
        return []

    count = min(
        count,
        len(products),
    )

    scores = np.array(
        [
            affinities[
                product.product_id
            ]

            for product
            in products
        ],
        dtype=np.float64,
    )

    # Softmax sampling
    logits = (
        scores / temperature
    )

    # numerical stability
    logits = (
        logits
        - logits.max()
    )

    probabilities = np.exp(
        logits
    )

    probabilities = (
        probabilities
        / probabilities.sum()
    )

    indices = rng.choice(
        len(products),
        size=count,
        replace=False,
        p=probabilities,
    )

    return [
        products[index]
        for index
        in indices
    ]


# =========================================================
# Generate One Session
# =========================================================

def generate_session(
    session_id: int,
    persona: Persona,
    all_products: list[Product],
    rng: np.random.Generator,
) -> SyntheticSession:

    profile = (
        BEHAVIOR_PROFILES[
            persona.behavior_type
        ]
    )

    # -----------------------------------------
    # 1. Gender
    #
    # Persona의 일부가 아님.
    # AR session의 외부 조건이라고 본다.
    # -----------------------------------------

    session_gender = rng.choice(
        [
            "MALE",
            "FEMALE",
        ]
    )

    candidate_products = (
        filter_products_by_gender(
            products=all_products,
            gender=session_gender,
        )
    )

    if not candidate_products:
        raise ValueError(
            f"No products for gender: "
            f"{session_gender}"
        )

    # -----------------------------------------
    # 2. 개별 고객 행동 propensity
    # -----------------------------------------

    session_fitting_base = (
        perturb_probability(
            profile.fitting_base,
            rng,
        )
    )

    session_wishlist_base = (
        perturb_probability(
            profile.wishlist_add_base,
            rng,
        )
    )

    session_remove_base = (
        perturb_probability(
            profile.wishlist_remove_base,
            rng,
        )
    )

    # -----------------------------------------
    # 3. Hidden preference
    #
    # 모델에는 제공되지 않음
    # -----------------------------------------

    hidden_preference = (
        create_hidden_preference(
            persona=persona,
            products=candidate_products,
            rng=rng,
        )
    )

    # -----------------------------------------
    # 4. 모든 후보 상품 affinity
    # -----------------------------------------

    affinities = (
        calculate_normalized_affinities(
            products=candidate_products,
            preference=hidden_preference,
        )
    )

    # -----------------------------------------
    # 5. 세션 내 상품 수
    # -----------------------------------------

    product_count = (
        sample_num_products(
            persona=persona,
            available_products=len(
                candidate_products
            ),
            rng=rng,
        )
    )

    # -----------------------------------------
    # 6. 상품 선택
    # -----------------------------------------

    selected_products = (
        sample_products(
            products=candidate_products,
            affinities=affinities,
            count=product_count,
            rng=rng,
        )
    )

    interactions = []

    sequence_no = 1

    # =========================================
    # AR Interaction 생성
    # =========================================

    for product in selected_products:

        affinity = (
            affinities[
                product.product_id
            ]
        )

        # -------------------------------------
        # PRODUCT_SELECT
        #
        # 선택된 상품은 항상 SELECT부터 시작
        # -------------------------------------

        interactions.append(
            Interaction(
                productId=(
                    product.product_id
                ),
                interactionType=(
                    PRODUCT_SELECT
                ),
                sequenceNo=sequence_no,
            )
        )

        sequence_no += 1

        # -------------------------------------
        # FITTING
        # -------------------------------------

        p_fitting = (
            behavior_probability(
                base=(
                    session_fitting_base
                ),
                affinity=affinity,
                affinity_strength=2.0,
            )
        )

        fitted = (
            rng.random()
            < p_fitting
        )

        if fitted:

            interactions.append(
                Interaction(
                    productId=(
                        product.product_id
                    ),
                    interactionType=FITTING,
                    sequenceNo=sequence_no,
                )
            )

            sequence_no += 1

        # -------------------------------------
        # WISHLIST_ADD
        #
        # FITTING한 상품이면 찜 확률 증가
        # -------------------------------------

        fitting_bonus = (
            0.70
            if fitted
            else 0.0
        )

        p_wishlist = (
            behavior_probability(
                base=(
                    session_wishlist_base
                ),
                affinity=affinity,
                affinity_strength=2.5,
                extra_bias=fitting_bonus,
            )
        )

        wishlist_added = (
            rng.random()
            < p_wishlist
        )

        if wishlist_added:

            interactions.append(
                Interaction(
                    productId=(
                        product.product_id
                    ),
                    interactionType=(
                        WISHLIST_ADD
                    ),
                    sequenceNo=sequence_no,
                )
            )

            sequence_no += 1

            # ---------------------------------
            # REMOVE
            #
            # ADD 이후에만 발생
            #
            # affinity가 낮을수록
            # REMOVE 가능성 증가
            # ---------------------------------

            p_remove = (
                behavior_probability(
                    base=(
                        session_remove_base
                    ),
                    affinity=affinity,
                    affinity_strength=-2.0,
                )
            )

            removed = (
                rng.random()
                < p_remove
            )

            if removed:

                interactions.append(
                    Interaction(
                        productId=(
                            product.product_id
                        ),
                        interactionType=(
                            WISHLIST_REMOVE
                        ),
                        sequenceNo=sequence_no,
                    )
                )

                sequence_no += 1

    return SyntheticSession(
        sessionId=session_id,
        interactions=interactions,
    )


# =========================================================
# Generate Dataset
# =========================================================

def generate_dataset(
    products: list[Product],
    sessions_per_persona: int = 1000,
    seed: int = 42,
) -> list[SyntheticSession]:

    rng = np.random.default_rng(
        seed
    )

    sessions = []

    session_id = 1

    for persona in PERSONAS:

        for _ in range(
            sessions_per_persona
        ):

            session = generate_session(
                session_id=session_id,
                persona=persona,
                all_products=products,
                rng=rng,
            )

            sessions.append(
                session
            )

            session_id += 1

    # Persona 생성 순서가 데이터에서
    # 그대로 드러나는 것을 방지
    rng.shuffle(
        sessions
    )

    return sessions


# =========================================================
# Save JSONL
# =========================================================

def save_jsonl(
    sessions: list[SyntheticSession],
    output_path: str,
):

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        for session in sessions:

            payload = {
                "sessionId":
                    session.sessionId,

                "interactions": [
                    asdict(
                        interaction
                    )

                    for interaction
                    in session.interactions
                ],
            }

            file.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                )
                + "\n"
            )


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    EXCEL_PATH = (
        "MCM_제품리스트_통합_추천모델용.xlsx"
    )

    OUTPUT_PATH = (
        "synthetic_interactions.jsonl"
    )

    products = (
        load_products_from_excel(
            EXCEL_PATH
        )
    )

    print(
        f"Loaded products: "
        f"{len(products)}"
    )

    sessions = generate_dataset(
        products=products,

        # 9 Persona × 1000
        # = 9000 sessions
        sessions_per_persona=1000,

        seed=42,
    )

    save_jsonl(
        sessions=sessions,
        output_path=OUTPUT_PATH,
    )

    total_interactions = sum(
        len(
            session.interactions
        )

        for session
        in sessions
    )

    print(
        f"Generated sessions: "
        f"{len(sessions)}"
    )

    print(
        f"Generated interactions: "
        f"{total_interactions}"
    )

    print(
        f"Saved: {OUTPUT_PATH}"
    )