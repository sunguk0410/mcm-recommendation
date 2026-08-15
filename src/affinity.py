from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .persona import Persona, ZONE_PROFILES


# =========================================================
# Product
# =========================================================

@dataclass(frozen=True)
class Product:
    product_id: int
    product_code: str
    name: str

    gender: str

    category: str
    sub_category: Optional[str]

    zone: str
    color: Optional[str]

    url: Optional[str]


# =========================================================
# Excel Loader
# =========================================================

def load_products_from_excel(
    excel_path: str,
) -> list[Product]:

    df = pd.read_excel(
        excel_path,
        sheet_name="Products",
    )

    required_columns = {
        "productId",
        "productCode",
        "name",
        "gender",
        "category",
        "subCategory",
        "zone",
        "color",
        "url",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    products = []

    for _, row in df.iterrows():

        sub_category = (
            None
            if pd.isna(row["subCategory"])
            else str(row["subCategory"]).strip()
        )

        color = (
            None
            if pd.isna(row["color"])
            else str(row["color"]).strip()
        )

        url = (
            None
            if pd.isna(row["url"])
            else str(row["url"]).strip()
        )

        product = Product(
            product_id=int(
                row["productId"]
            ),
            product_code=str(
                row["productCode"]
            ).strip(),
            name=str(
                row["name"]
            ).strip(),

            gender=str(
                row["gender"]
            ).strip().upper(),

            category=str(
                row["category"]
            ).strip().upper(),

            sub_category=sub_category,

            zone=str(
                row["zone"]
            ).strip().upper(),

            color=color,
            url=url,
        )

        products.append(product)

    return products


# =========================================================
# Hidden Preference
# =========================================================
#
# Synthetic customer에게만 존재하는 실제 취향.
#
# RecRec은 아래 값을 절대 볼 수 없음.
# =========================================================

@dataclass
class HiddenPreference:

    zone_probs: dict[str, float]

    category_probs: dict[str, float]

    subcategory_probs: dict[
        str,
        dict[str, float]
    ]


def create_hidden_preference(
    persona: Persona,
    products: list[Product],
    rng: np.random.Generator,
) -> HiddenPreference:

    # -----------------------------------------
    # 1. Zone
    # -----------------------------------------

    zone_probs = (
        ZONE_PROFILES[
            persona.zone_type
        ].copy()
    )

    # -----------------------------------------
    # 2. 개인 Category 취향
    # -----------------------------------------

    categories = sorted({
        product.category
        for product in products
    })

    # alpha < 1
    # → 일부 카테고리가 상대적으로 강한 분포 생성
    category_distribution = (
        rng.dirichlet(
            np.full(
                len(categories),
                0.7,
            )
        )
    )

    category_probs = {
        category: float(probability)
        for category, probability
        in zip(
            categories,
            category_distribution,
        )
    }

    # -----------------------------------------
    # 3. 개인 SubCategory 취향
    # -----------------------------------------

    subcategories_by_category = {}

    for category in categories:

        subcategories = sorted({
            product.sub_category
            for product in products

            if (
                product.category == category
                and product.sub_category
            )
        })

        if subcategories:
            subcategories_by_category[
                category
            ] = subcategories

    subcategory_probs = {}

    for (
        category,
        subcategories
    ) in subcategories_by_category.items():

        distribution = rng.dirichlet(
            np.full(
                len(subcategories),
                0.6,
            )
        )

        subcategory_probs[
            category
        ] = {

            subcategory: float(probability)

            for (
                subcategory,
                probability
            )

            in zip(
                subcategories,
                distribution,
            )
        }

    return HiddenPreference(
        zone_probs=zone_probs,
        category_probs=category_probs,
        subcategory_probs=(
            subcategory_probs
        ),
    )


# =========================================================
# Product Affinity
# =========================================================

def calculate_raw_affinity(
    product: Product,
    preference: HiddenPreference,
) -> float:

    zone_score = (
        preference
        .zone_probs
        .get(
            product.zone,
            0.0,
        )
    )

    category_score = (
        preference
        .category_probs
        .get(
            product.category,
            0.0,
        )
    )

    # -----------------------------------------
    # SubCategory가 있는 상품
    # -----------------------------------------

    if product.sub_category:

        subcategory_score = (
            preference
            .subcategory_probs
            .get(
                product.category,
                {},
            )
            .get(
                product.sub_category,
                0.0,
            )
        )

        return float(
            0.45 * zone_score
            + 0.35 * category_score
            + 0.20 * subcategory_score
        )

    # -----------------------------------------
    # SubCategory가 없는 상품
    #
    # 기존:
    # Zone     = 0.45
    # Category = 0.35
    #
    # 합 = 0.80
    #
    # 다시 합이 1이 되도록 normalize
    # -----------------------------------------

    zone_weight = (
        0.45 / 0.80
    )

    category_weight = (
        0.35 / 0.80
    )

    return float(
        zone_weight * zone_score
        + category_weight * category_score
    )


def calculate_normalized_affinities(
    products: list[Product],
    preference: HiddenPreference,
) -> dict[int, float]:

    raw_scores = {

        product.product_id:
            calculate_raw_affinity(
                product,
                preference,
            )

        for product in products
    }

    values = np.array(
        list(
            raw_scores.values()
        ),
        dtype=np.float64,
    )

    minimum = values.min()
    maximum = values.max()

    score_range = (
        maximum - minimum
    )

    # 모든 상품 점수가 같은 극단적인 상황
    if score_range < 1e-8:

        return {
            product_id: 0.5
            for product_id
            in raw_scores
        }

    return {

        product_id:
            float(
                (score - minimum)
                / score_range
            )

        for (
            product_id,
            score
        )
        in raw_scores.items()
    }


# =========================================================
# Gender Filter
# =========================================================

def filter_products_by_gender(
    products: list[Product],
    gender: str,
) -> list[Product]:

    gender = gender.upper()

    return [

        product

        for product
        in products

        if (
            product.gender == gender
            or product.gender == "UNISEX"
        )
    ]