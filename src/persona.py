from dataclasses import dataclass
from enum import Enum


class ZoneType(str, Enum):
    CLASSIC = "CLASSIC"
    NEW = "NEW"
    TRAVEL = "TRAVEL"


class BehaviorType(str, Enum):
    # 여러 상품을 탐색하며 피팅/찜도 적당히 발생
    FASHION_EXPLORER = "FASHION_EXPLORER"

    # 상대적으로 적은 상품을 보지만 강한 행동이 많이 발생
    DECISIVE_BUYER = "DECISIVE_BUYER"

    # 매우 넓게 탐색하지만 강한 행동은 적음
    BROAD_EXPLORER = "BROAD_EXPLORER"


@dataclass(frozen=True)
class BehaviorProfile:
    # 세션당 탐색 상품 수
    lambda_products: float
    min_products: int
    max_products: int

    # 행동 baseline propensity
    fitting_base: float
    wishlist_add_base: float
    wishlist_remove_base: float


BEHAVIOR_PROFILES = {

    BehaviorType.FASHION_EXPLORER: BehaviorProfile(
        lambda_products=13,
        min_products=8,
        max_products=20,

        fitting_base=0.45,
        wishlist_add_base=0.28,
        wishlist_remove_base=0.12,
    ),

    BehaviorType.DECISIVE_BUYER: BehaviorProfile(
        lambda_products=8,
        min_products=5,
        max_products=14,

        fitting_base=0.78,
        wishlist_add_base=0.58,
        wishlist_remove_base=0.06,
    ),

    BehaviorType.BROAD_EXPLORER: BehaviorProfile(
        lambda_products=18,
        min_products=10,
        max_products=28,

        fitting_base=0.18,
        wishlist_add_base=0.07,
        wishlist_remove_base=0.24,
    ),
}


# =========================================================
# Zone Persona
# =========================================================
#
# Synthetic generator 내부에서만 사용하는 확률.
# RecRec 학습 입력에는 절대 들어가지 않음.
#
# 예:
# CLASSIC persona라도
# CLASSIC 상품만 고르는 것은 아님.
# =========================================================

ZONE_PROFILES = {

    ZoneType.CLASSIC: {
        "CLASSIC": 0.65,
        "NEW": 0.25,
        "TRAVEL": 0.10,
    },

    ZoneType.NEW: {
        "CLASSIC": 0.20,
        "NEW": 0.65,
        "TRAVEL": 0.15,
    },

    ZoneType.TRAVEL: {
        "CLASSIC": 0.15,
        "NEW": 0.20,
        "TRAVEL": 0.65,
    },
}


@dataclass(frozen=True)
class Persona:
    persona_id: str
    behavior_type: BehaviorType
    zone_type: ZoneType


# =========================================================
# 3 행동 성향 × 3 Zone = 9 Personas
# =========================================================

PERSONAS = [

    # 패션 탐색형
    Persona(
        "P1",
        BehaviorType.FASHION_EXPLORER,
        ZoneType.CLASSIC,
    ),
    Persona(
        "P2",
        BehaviorType.FASHION_EXPLORER,
        ZoneType.NEW,
    ),
    Persona(
        "P3",
        BehaviorType.FASHION_EXPLORER,
        ZoneType.TRAVEL,
    ),

    # 확신 구매형
    Persona(
        "P4",
        BehaviorType.DECISIVE_BUYER,
        ZoneType.CLASSIC,
    ),
    Persona(
        "P5",
        BehaviorType.DECISIVE_BUYER,
        ZoneType.NEW,
    ),
    Persona(
        "P6",
        BehaviorType.DECISIVE_BUYER,
        ZoneType.TRAVEL,
    ),

    # 탐색적 고객
    Persona(
        "P7",
        BehaviorType.BROAD_EXPLORER,
        ZoneType.CLASSIC,
    ),
    Persona(
        "P8",
        BehaviorType.BROAD_EXPLORER,
        ZoneType.NEW,
    ),
    Persona(
        "P9",
        BehaviorType.BROAD_EXPLORER,
        ZoneType.TRAVEL,
    ),
]