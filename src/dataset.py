import json
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from torch.utils.data import Dataset

from .affinity import (
    Product,
    load_products_from_excel,
)


# =========================================================
# Interaction Types
# =========================================================

PRODUCT_SELECT = "PRODUCT_SELECT"
FITTING = "FITTING"
WISHLIST_ADD = "WISHLIST_ADD"
WISHLIST_REMOVE = "WISHLIST_REMOVE"


# 0 = PAD
BEHAVIOR_TO_ID = {

    "PAD": 0,

    PRODUCT_SELECT: 1,

    FITTING: 2,

    WISHLIST_ADD: 3,

    WISHLIST_REMOVE: 4,
}


# =========================================================
# Data Classes
# =========================================================

@dataclass
class RawInteraction:
    product_id: int
    interaction_type: str
    sequence_no: int


@dataclass
class ProductEpisode:
    product_id: int

    interactions: list[
        RawInteraction
    ]


@dataclass
class TrainingSample:

    history: list[
        RawInteraction
    ]

    target_product_id: int


# =========================================================
# JSONL
# =========================================================

def load_sessions(
    jsonl_path: str,
) -> list[dict]:

    sessions = []

    with open(
        jsonl_path,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            sessions.append(
                json.loads(
                    line
                )
            )

    return sessions


# =========================================================
# Session Split
# =========================================================
#
# 중요:
#
# 하나의 session에서 여러 training sample이 나온다.
#
# sample을 만든 다음 train/test split하면
# 같은 고객 session의 prefix가
# train과 test 양쪽에 들어가는 leakage 발생 가능.
#
# 따라서 반드시 SESSION 단위로 먼저 나눈다.
# =========================================================

def split_sessions(
    sessions: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
):

    rng = np.random.default_rng(
        seed
    )

    indices = np.arange(
        len(sessions)
    )

    rng.shuffle(
        indices
    )

    train_end = int(
        len(indices)
        * train_ratio
    )

    val_end = int(
        len(indices)
        * (
            train_ratio
            + val_ratio
        )
    )

    train_sessions = [
        sessions[index]
        for index
        in indices[:train_end]
    ]

    val_sessions = [
        sessions[index]
        for index
        in indices[
            train_end:val_end
        ]
    ]

    test_sessions = [
        sessions[index]
        for index
        in indices[val_end:]
    ]

    return (
        train_sessions,
        val_sessions,
        test_sessions,
    )


# =========================================================
# Product Episode
# =========================================================

def build_product_episodes(
    interactions: list[RawInteraction],
) -> list[ProductEpisode]:

    interactions = sorted(
        interactions,
        key=lambda interaction:
            interaction.sequence_no,
    )

    episodes = []

    current_episode: Optional[
        ProductEpisode
    ] = None

    for interaction in interactions:

        # 새로운 상품 선택
        if (
            interaction.interaction_type
            == PRODUCT_SELECT
        ):

            if (
                current_episode
                is not None
            ):

                episodes.append(
                    current_episode
                )

            current_episode = (
                ProductEpisode(
                    product_id=(
                        interaction.product_id
                    ),
                    interactions=[
                        interaction
                    ],
                )
            )

            continue

        # SELECT 이전 행동은 비정상
        if current_episode is None:
            continue

        # 현재 선택된 상품과 다른 ID면
        # malformed interaction
        if (
            interaction.product_id
            != current_episode.product_id
        ):
            continue

        current_episode.interactions.append(
            interaction
        )

    if (
        current_episode
        is not None
    ):

        episodes.append(
            current_episode
        )

    return episodes


# =========================================================
# Training Samples
# =========================================================
#
# 예:
#
# Episode 1
# P12 SELECT
# P12 FITTING
# P12 WISHLIST_ADD
#
# Episode 2
# P31 SELECT
# P31 FITTING
#
# Episode 3
# P08 SELECT
#
#
# Sample 1
# [P12 전체 episode]
# -> P31
#
# Sample 2
# [P12 + P31 전체 episode]
# -> P08
#
#
# WISHLIST_REMOVE가 있더라도
# target 상품에서 제외하지 않는다.
#
# 이유:
# "다음 SELECT 상품" 예측에서는
# 실제 다음 선택 상품이 맞기 때문.
# =========================================================

def build_training_samples(
    episodes: list[ProductEpisode],
    min_history_products: int = 1,
) -> list[TrainingSample]:

    samples = []

    if (
        len(episodes)
        <= min_history_products
    ):
        return samples

    for target_index in range(
        min_history_products,
        len(episodes),
    ):

        history_interactions = []

        for episode in episodes[
            :target_index
        ]:

            history_interactions.extend(
                episode.interactions
            )

        if not history_interactions:
            continue

        target_episode = (
            episodes[
                target_index
            ]
        )

        samples.append(
            TrainingSample(
                history=(
                    history_interactions
                ),
                target_product_id=(
                    target_episode.product_id
                ),
            )
        )

    return samples


# =========================================================
# Product ID Mapper
# =========================================================

class ProductIdMapper:

    def __init__(
        self,
        products: list[Product],
    ):

        product_ids = sorted({
            product.product_id
            for product in products
        })

        # PAD = 0
        self.product_to_index = {

            product_id:
                index + 1

            for (
                index,
                product_id
            )

            in enumerate(
                product_ids
            )
        }

        self.index_to_product = {

            index:
                product_id

            for (
                product_id,
                index
            )

            in self.product_to_index.items()
        }

    def encode(
        self,
        product_id: int,
    ) -> int:

        return (
            self.product_to_index[
                product_id
            ]
        )

    def decode(
        self,
        index: int,
    ) -> int:

        return (
            self.index_to_product[
                index
            ]
        )

    @property
    def num_products(
        self,
    ) -> int:

        # +1 = PAD
        return (
            len(
                self.product_to_index
            )
            + 1
        )


# =========================================================
# RecRec Dataset
# =========================================================

class RecRecDataset(Dataset):

    def __init__(
        self,
        sessions: list[dict],
        product_mapper: ProductIdMapper,
        max_seq_len: int = 64,
        min_history_products: int = 1,
        use_behavior: bool = True,
    ):

        self.max_seq_len = max_seq_len
        self.product_mapper = product_mapper
        self.use_behavior = use_behavior

        self.samples = []

        for session in sessions:

            raw_interactions = []

            for item in session["interactions"]:

                interaction_type = item["interactionType"]

                # 정의되지 않은 interaction은 무시
                if interaction_type not in BEHAVIOR_TO_ID:
                    continue

                # PAD는 실제 행동이 아니므로 무시
                if interaction_type == "PAD":
                    continue

                product_id = int(
                    item["productId"]
                )

                # catalog에 없는 product 방어
                if (
                    product_id
                    not in product_mapper.product_to_index
                ):
                    continue

                raw_interactions.append(
                    RawInteraction(
                        product_id=product_id,
                        interaction_type=interaction_type,
                        sequence_no=int(
                            item["sequenceNo"]
                        ),
                    )
                )

            # sequenceNo 기준 episode 생성
            episodes = build_product_episodes(
                raw_interactions
            )

            # prefix -> next product 학습 샘플 생성
            session_samples = build_training_samples(
                episodes=episodes,
                min_history_products=min_history_products,
            )

            self.samples.extend(
                session_samples
            )

    def __len__(self):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        index: int,
    ):

        sample = self.samples[
            index
        ]

        # 최근 interaction만 유지
        history = sample.history[
            -self.max_seq_len:
        ]

        product_ids = []
        behavior_ids = []

        for interaction in history:

            # -----------------------------
            # Product ID
            # -----------------------------

            product_ids.append(
                self.product_mapper.encode(
                    interaction.product_id
                )
            )

            # -----------------------------
            # Behavior ID
            # -----------------------------
            #
            # use_behavior=True
            #   PRODUCT_SELECT  -> 1
            #   FITTING         -> 2
            #   WISHLIST_ADD    -> 3
            #   WISHLIST_REMOVE -> 4
            #
            # use_behavior=False
            #   모든 실제 행동 -> 1
            #
            # Product-only ablation
            # -----------------------------

            if self.use_behavior:

                behavior_id = BEHAVIOR_TO_ID[
                    interaction.interaction_type
                ]

            else:

                behavior_id = 1

            behavior_ids.append(
                behavior_id
            )

        sequence_length = len(
            product_ids
        )

        padding_length = (
            self.max_seq_len
            - sequence_length
        )

        # -----------------------------
        # Left Padding
        # -----------------------------

        product_ids = (
            [0] * padding_length
            + product_ids
        )

        behavior_ids = (
            [0] * padding_length
            + behavior_ids
        )

        attention_mask = (
            [0] * padding_length
            + [1] * sequence_length
        )

        # -----------------------------
        # Target
        # -----------------------------

        target_product_id = (
            self.product_mapper.encode(
                sample.target_product_id
            )
        )

        return {

            "product_ids": torch.tensor(
                product_ids,
                dtype=torch.long,
            ),

            "behavior_ids": torch.tensor(
                behavior_ids,
                dtype=torch.long,
            ),

            "attention_mask": torch.tensor(
                attention_mask,
                dtype=torch.bool,
            ),

            "target_product_id": torch.tensor(
                target_product_id,
                dtype=torch.long,
            ),
        }


# =========================================================
# Dataset Factory
# =========================================================

def create_datasets(
    jsonl_path: str,
    catalog_path: str,
    max_seq_len: int = 64,
    seed: int = 42,
    use_behavior: bool = True,
):

    # -----------------------------------------------------
    # Product Catalog
    # -----------------------------------------------------

    products = load_products_from_excel(
        catalog_path
    )

    mapper = ProductIdMapper(
        products
    )

    # -----------------------------------------------------
    # Sessions
    # -----------------------------------------------------

    sessions = load_sessions(
        jsonl_path
    )

    (
        train_sessions,
        val_sessions,
        test_sessions,
    ) = split_sessions(
        sessions=sessions,
        seed=seed,
    )

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    train_dataset = RecRecDataset(
        sessions=train_sessions,
        product_mapper=mapper,
        max_seq_len=max_seq_len,
        use_behavior=use_behavior,
    )

    val_dataset = RecRecDataset(
        sessions=val_sessions,
        product_mapper=mapper,
        max_seq_len=max_seq_len,
        use_behavior=use_behavior,
    )

    test_dataset = RecRecDataset(
        sessions=test_sessions,
        product_mapper=mapper,
        max_seq_len=max_seq_len,
        use_behavior=use_behavior,
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        mapper,
    )


# =========================================================
# Test
# =========================================================

if __name__ == "__main__":

    JSONL_PATH = (
        "synthetic_interactions_v2.jsonl"
    )

    CATALOG_PATH = (
        "MCM_제품리스트_통합_추천모델용.xlsx"
    )

    (
        train_dataset,
        val_dataset,
        test_dataset,
        mapper,
    ) = create_datasets(
        jsonl_path=JSONL_PATH,
        catalog_path=CATALOG_PATH,
        max_seq_len=64,
        seed=42,
    )

    print(
        "============================="
    )

    print(
        f"Products: "
        f"{mapper.num_products - 1}"
    )

    print(
        f"Train samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_dataset)}"
    )

    print(
        f"Test samples: "
        f"{len(test_dataset)}"
    )

    print(
        "============================="
    )

    if len(train_dataset) > 0:

        sample = (
            train_dataset[0]
        )

        print(
            sample
        )

        internal_target = int(
            sample[
                "target_product_id"
            ].item()
        )

        original_product_id = (
            mapper.decode(
                internal_target
            )
        )

        print(
            "Original target productId:",
            original_product_id,
        )