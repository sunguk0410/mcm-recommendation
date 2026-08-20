import torch
import torch.nn.functional as F
import pandas as pd
import re

from .recrec import RecRec
from .dataset import (
    ProductIdMapper,
    BEHAVIOR_TO_ID,
    load_products_from_excel,
)


CHECKPOINT_PATH = "checkpoints/recrec_v2_best.pt"
CATALOG_PATH = "MCM_제품리스트_통합_추천모델용.xlsx"

MAX_SEQ_LEN = 64

COMPLEMENTARY_CATEGORY_BONUSES = {
    "BAG": {"ACCESSORIES": 0.08, "SHOES": 0.06, "TOP": 0.04},
    "BOTTOM": {"TOP": 0.10, "SHOES": 0.07, "ACCESSORIES": 0.05},
    "TOP": {"BOTTOM": 0.10, "ACCESSORIES": 0.06, "SHOES": 0.04},
    "SHOES": {"BOTTOM": 0.08, "BAG": 0.06, "TOP": 0.04},
    "ACCESSORIES": {"BAG": 0.08, "TOP": 0.06, "BOTTOM": 0.04},
}

MAX_DIVERSE_PRODUCTS_PER_CATEGORY = 2
DEFAULT_DIVERSITY_WINDOW = 5
INITIAL_FEATURED_PRODUCTS = {
    "BAG": [4, 44, 75, 62, 63, 53],
    "BOTTOM": [100, 105],
    "TOP": [80, 81],
}
CONTENT_STOP_WORDS = {
    "mcm", "x", "비세토스", "모노그램", "레더", "가죽", "코튼",
    "로고", "프린트", "백", "티셔츠", "셔츠", "팬츠", "블랙",
}

class RecRecInference:

    def __init__(self):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Inference device: {self.device}"
        )

        # =================================================
        # Product Mapper
        # =================================================

        products = load_products_from_excel(
            CATALOG_PATH
        )

        self.products = products

        self.product_by_id = {
            product.product_id: product
            for product in products
        }

        self.mapper = ProductIdMapper(
            products
        )

        # =================================================
        # Checkpoint
        # =================================================

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=self.device,
        )

        config = checkpoint[
            "model_config"
        ]

        self.model = RecRec(
            num_products=config[
                "num_products"
            ],
            embedding_dim=config[
                "embedding_dim"
            ],
            hidden_dim=config[
                "hidden_dim"
            ],
            num_behavior_types=config[
                "num_behavior_types"
            ],
            num_refinement_steps=config[
                "num_refinement_steps"
            ],
            num_inner_steps=config[
                "num_inner_steps"
            ],
            recursive_layers=config[
                "recursive_layers"
            ],
            update_scale=config[
                "update_scale"
            ],
            temperature=config[
                "temperature"
            ],
            dropout=config[
                "dropout"
            ],
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        print(
            f"Loaded checkpoint epoch: "
            f"{checkpoint['epoch']}"
        )

    # =====================================================
    # Input Tensor
    # =====================================================

    def _build_input(
        self,
        interactions,
    ):

        product_ids = []
        behavior_ids = []

        for interaction in interactions:

            product_id = int(
                interaction["productId"]
            )

            interaction_type = (
                interaction[
                    "interactionType"
                ]
            )

            if (
                product_id
                not in self.mapper.product_to_index
            ):
                continue

            if (
                interaction_type
                not in BEHAVIOR_TO_ID
            ):
                continue

            product_ids.append(
                self.mapper.encode(
                    product_id
                )
            )

            behavior_ids.append(
                BEHAVIOR_TO_ID[
                    interaction_type
                ]
            )

        # 최근 64개
        product_ids = product_ids[
            -MAX_SEQ_LEN:
        ]

        behavior_ids = behavior_ids[
            -MAX_SEQ_LEN:
        ]

        sequence_length = len(
            product_ids
        )

        if sequence_length == 0:
            raise ValueError(
                "No valid interactions"
            )

        padding_length = (
            MAX_SEQ_LEN
            - sequence_length
        )

        product_ids = (
            [0] * padding_length
            + product_ids
        )

        behavior_ids = (
            [0] * padding_length
            + behavior_ids
        )

        attention_mask = (
            [False] * padding_length
            + [True] * sequence_length
        )

        return (
            torch.tensor(
                [product_ids],
                dtype=torch.long,
                device=self.device,
            ),
            torch.tensor(
                [behavior_ids],
                dtype=torch.long,
                device=self.device,
            ),
            torch.tensor(
                [attention_mask],
                dtype=torch.bool,
                device=self.device,
            ),
        )

    # =====================================================
    # Recommend
    # =====================================================

    @torch.no_grad()
    def build_wishlist_preference_scores(self, interactions):
        wishlist_embeddings = []

        for interaction in interactions:
            product_id = int(interaction.productId)
            if product_id not in self.mapper.product_to_index:
                continue

            product_index = self.mapper.encode(product_id)
            wishlist_embeddings.append(
                self.model.product_embedding.weight[product_index]
            )

        if not wishlist_embeddings:
            raise ValueError("No valid memberInteractions")

        member_preference = torch.stack(wishlist_embeddings).mean(dim=0)
        member_preference = F.normalize(member_preference, dim=0)
        product_embeddings = F.normalize(
            self.model.product_embedding.weight,
            dim=-1,
        )
        similarities = product_embeddings @ member_preference

        return {
            product.product_id: float(
                (similarities[self.mapper.encode(product.product_id)] + 1.0) / 2.0
            )
            for product in self.products
        }

    @staticmethod
    def _initial_preference_score(product, preference_scores):
        if product.product_id in preference_scores:
            return preference_scores[product.product_id]
        return preference_scores.get(product.zone, 0.0)

    @staticmethod
    def _rec_weight(interaction_count):
        if interaction_count <= 2:
            return 0.70
        if interaction_count <= 5:
            return 0.75
        return 0.80

    @staticmethod
    def _excluded_product_ids(
        interactions,
        preference_product_ids,
        exclude_seen,
    ):
        if not exclude_seen:
            return set()
        return {
            int(interaction["productId"])
            for interaction in interactions
        } | {
            int(product_id)
            for product_id in (preference_product_ids or [])
        }

    def _complementary_bonus(self, candidate, interactions):
        source_categories = []
        for interaction in interactions:
            product = self.product_by_id.get(int(interaction["productId"]))
            if product is not None:
                source_categories.append(product.category)

        bonus = 0.0
        for recency_index, source_category in enumerate(reversed(source_categories)):
            recency_factor = max(0.5, 1.0 - 0.15 * recency_index)
            category_bonus = COMPLEMENTARY_CATEGORY_BONUSES.get(
                source_category,
                {},
            ).get(candidate.category, 0.0)
            bonus = max(bonus, category_bonus * recency_factor)
        return bonus

    @staticmethod
    def _name_tokens(name):
        return {
            token
            for token in re.findall(r"[0-9a-zA-Z가-힣]+", name.casefold())
            if len(token) > 1 and token not in CONTENT_STOP_WORDS
        }

    def _content_bonus(
        self,
        candidate,
        interactions,
        preference_product_ids,
    ):
        preference_product_ids = {
            int(product_id) for product_id in (preference_product_ids or [])
        }
        reference_ids = [
            int(interaction["productId"])
            for interaction in interactions
        ] + list(preference_product_ids)
        candidate_tokens = self._name_tokens(candidate.name)
        best_bonus = 0.0

        for product_id in reference_ids:
            reference = self.product_by_id.get(product_id)
            if reference is None or reference.product_id == candidate.product_id:
                continue

            shared_tokens = candidate_tokens & self._name_tokens(reference.name)
            bonus = min(0.12, 0.06 * len(shared_tokens))
            if (
                candidate.sub_category
                and reference.sub_category
                and candidate.sub_category == reference.sub_category
            ):
                bonus += 0.04
            if candidate.color and candidate.color == reference.color:
                bonus += 0.015
            if candidate.zone == reference.zone:
                bonus += 0.015
            if candidate.category == reference.category:
                bonus += 0.02
            best_bonus = max(best_bonus, min(0.18, bonus))

        return best_bonus

    @staticmethod
    def _diverse_positions(products, scores, result_count):
        ranked_positions = sorted(
            range(len(products)),
            key=lambda position: (-float(scores[position]), products[position].product_id),
        )
        diversity_window = min(DEFAULT_DIVERSITY_WINDOW, result_count)
        selected = []
        deferred = []
        category_counts = {}

        for position in ranked_positions:
            category = products[position].category
            if (
                len(selected) < diversity_window
                and category_counts.get(category, 0)
                < MAX_DIVERSE_PRODUCTS_PER_CATEGORY
            ):
                selected.append(position)
                category_counts[category] = category_counts.get(category, 0) + 1
            else:
                deferred.append(position)

        if len(selected) < diversity_window:
            needed = diversity_window - len(selected)
            selected.extend(deferred[:needed])
            deferred = deferred[needed:]

        return (selected + deferred)[:result_count]

    @torch.no_grad()
    def _recommend_legacy(
        self,
        interactions,
        top_k=6,
        category=None,
        exclude_seen=True,
    ):

        (
            product_ids,
            behavior_ids,
            attention_mask,
        ) = self._build_input(
            interactions
        )

        outputs = self.model(
            product_ids=product_ids,
            behavior_ids=behavior_ids,
            attention_mask=attention_mask,
        )

        logits = outputs[
            "final_logits"
        ][0]

        # PAD 제외
        logits[0] = float(
            "-inf"
        )

        # =====================================================
        # Category Filtering
        # =====================================================

        if category is not None:

            for product_id, index in (
                self.mapper.product_to_index.items()
            ):

                product = self.product_by_id[
                    product_id
                ]

                if product.category != category:

                    logits[index] = float(
                        "-inf"
                    )

        # 이미 본 상품 제외
        if exclude_seen:

            seen_products = {
                int(
                    interaction[
                        "productId"
                    ]
                )
                for interaction
                in interactions
            }

            for product_id in seen_products:

                if (
                    product_id
                    in self.mapper.product_to_index
                ):

                    index = self.mapper.encode(
                        product_id
                    )

                    logits[
                        index
                    ] = float(
                        "-inf"
                    )

        # Top K
        values, indices = torch.topk(
            logits,
            k=top_k,
        )

        recommendations = []

        for score, index in zip(
            values.tolist(),
            indices.tolist(),
        ):

            product_id = (
                self.mapper.decode(
                    index
                )
            )

            recommendations.append({
                "productId": (
                    product_id
                ),
                "score": float(
                    score
                ),
            })

        return recommendations

    @torch.no_grad()
    def recommend(
        self,
        interactions,
        zone_scores=None,
        top_k=6,
        category=None,
        gender=None,
        exclude_seen=True,
        diversify=False,
        preference_product_ids=None,
    ):
        zone_scores = zone_scores or {}
        category = category.strip().upper() if category else None
        gender = gender.strip().upper() if gender else None

        candidates = [
            product for product in self.products
            if (category is None or product.category == category)
            and (
                category == "BAG"
                or
                gender is None
                or product.gender == gender
                or product.gender == "UNISEX"
            )
        ]
        if not candidates:
            raise ValueError(f"Unknown category: {category}")

        seen_products = self._excluded_product_ids(
            interactions,
            preference_product_ids,
            exclude_seen,
        )
        candidates = [
            product for product in candidates
            if product.product_id not in seen_products
        ]
        if not candidates:
            return []

        # Initial recommendations do not invoke RecRec.
        if not interactions:
            is_cold_start = not zone_scores and not preference_product_ids
            featured_ranks = {
                product_id: rank
                for rank, product_id in enumerate(
                    INITIAL_FEATURED_PRODUCTS.get(category, [])
                )
            }
            recommendations = [
                {
                    "productId": product.product_id,
                    "score": float(self._initial_preference_score(product, zone_scores)),
                }
                for product in candidates
            ]
            recommendations.sort(
                key=lambda item: (
                    0
                    if is_cold_start
                    and item["productId"] in featured_ranks
                    else 1,
                    featured_ranks.get(item["productId"], 0),
                    -item["score"],
                    item["productId"],
                )
            )
            return recommendations[:top_k]

        product_ids, behavior_ids, attention_mask = self._build_input(
            interactions
        )
        outputs = self.model(
            product_ids=product_ids,
            behavior_ids=behavior_ids,
            attention_mask=attention_mask,
        )
        logits = outputs["final_logits"][0]
        candidate_indices = torch.tensor(
            [
                self.mapper.encode(product.product_id)
                for product in candidates
            ],
            dtype=torch.long,
            device=self.device,
        )
        candidate_logits = logits[candidate_indices]

        # Preserve the original raw RecRec scores when zone data is absent.
        if zone_scores:
            minimum = candidate_logits.min()
            score_range = candidate_logits.max() - minimum
            if score_range.item() > 0:
                rec_scores = (candidate_logits - minimum) / score_range
            else:
                rec_scores = torch.zeros_like(candidate_logits)

            rec_weight = self._rec_weight(len(interactions))
            candidate_zone_scores = torch.tensor(
                [
                    self._initial_preference_score(product, zone_scores)
                    for product in candidates
                ],
                dtype=rec_scores.dtype,
                device=self.device,
            )
            final_scores = (
                rec_scores * rec_weight
                + candidate_zone_scores * (1.0 - rec_weight)
            )
        else:
            final_scores = candidate_logits

        content_bonuses = torch.tensor(
            [
                self._content_bonus(
                    product,
                    interactions,
                    preference_product_ids,
                )
                for product in candidates
            ],
            dtype=final_scores.dtype,
            device=self.device,
        )
        final_scores = final_scores + content_bonuses

        if diversify and category is None:
            complementary_bonuses = torch.tensor(
                [
                    self._complementary_bonus(product, interactions)
                    for product in candidates
                ],
                dtype=final_scores.dtype,
                device=self.device,
            )
            final_scores = final_scores + complementary_bonuses

        result_count = min(top_k, len(candidates))
        if diversify and category is None:
            positions = self._diverse_positions(
                candidates,
                final_scores,
                result_count,
            )
            values = [float(final_scores[position]) for position in positions]
        else:
            tensor_values, tensor_positions = torch.topk(
                final_scores,
                k=result_count,
            )
            values = tensor_values.tolist()
            positions = tensor_positions.tolist()
        return [
            {
                "productId": candidates[position].product_id,
                "score": float(score),
            }
            for score, position in zip(values, positions)
        ]
