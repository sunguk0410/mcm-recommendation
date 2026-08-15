import torch
import pandas as pd

from .recrec import RecRec
from .dataset import (
    ProductIdMapper,
    BEHAVIOR_TO_ID,
    load_products_from_excel,
)


CHECKPOINT_PATH = "checkpoints/recrec_v2_best.pt"
CATALOG_PATH = "MCM_제품리스트_통합_추천모델용.xlsx"

MAX_SEQ_LEN = 64

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
        exclude_seen=True,
    ):
        zone_scores = zone_scores or {}
        category = category.strip().upper() if category else None

        candidates = [
            product for product in self.products
            if category is None or product.category == category
        ]
        if not candidates:
            raise ValueError(f"Unknown category: {category}")

        seen_products = {
            int(interaction["productId"])
            for interaction in interactions
        } if exclude_seen else set()
        candidates = [
            product for product in candidates
            if product.product_id not in seen_products
        ]
        if not candidates:
            return []

        # Initial recommendations do not invoke RecRec.
        if not interactions:
            recommendations = [
                {
                    "productId": product.product_id,
                    "score": float(zone_scores.get(product.zone, 0.0)),
                }
                for product in candidates
            ]
            recommendations.sort(
                key=lambda item: (-item["score"], item["productId"])
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

            rec_weight = 0.7 if len(interactions) <= 2 else 0.9
            candidate_zone_scores = torch.tensor(
                [
                    zone_scores.get(product.zone, 0.0)
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

        result_count = min(top_k, len(candidates))
        values, positions = torch.topk(final_scores, k=result_count)
        return [
            {
                "productId": candidates[position].product_id,
                "score": float(score),
            }
            for score, position in zip(
                values.tolist(),
                positions.tolist(),
            )
        ]
