import torch
import torch.nn as nn
import torch.nn.functional as F


class RecursiveBlock(nn.Module):
    """
    논문의 shared recursive function f_phi.

    입력:
        [x || y || z]

    출력:
        새로운 latent z

    같은 block을 inner recursion과
    preference update에서 반복해서 재사용한다.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        input_dim = embedding_dim * 3

        layers = []

        # 첫 layer
        layers.append(
            nn.Linear(
                input_dim,
                hidden_dim,
            )
        )
        layers.append(
            nn.GELU()
        )
        layers.append(
            nn.Dropout(dropout)
        )

        # 중간 layers
        for _ in range(
            num_layers - 2
        ):
            layers.append(
                nn.Linear(
                    hidden_dim,
                    hidden_dim,
                )
            )
            layers.append(
                nn.GELU()
            )
            layers.append(
                nn.Dropout(dropout)
            )

        # embedding dimension으로 복귀
        layers.append(
            nn.Linear(
                hidden_dim,
                embedding_dim,
            )
        )

        self.network = nn.Sequential(
            *layers
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.network(x)


class RecRec(nn.Module):
    """
    RecRec:
    Recursive Refinement for Sequential Recommendation

    MCM 확장:
        item embedding
        +
        behavior embedding

    behavior:
        0 = PAD
        1 = PRODUCT_SELECT
        2 = FITTING
        3 = WISHLIST_ADD
        4 = WISHLIST_REMOVE
    """

    def __init__(
        self,
        num_products: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_behavior_types: int = 5,

        # 논문의 outer refinement T
        num_refinement_steps: int = 6,

        # 논문의 inner recursion n
        num_inner_steps: int = 3,

        recursive_layers: int = 3,

        update_scale: float = 0.1,
        temperature: float = 0.07,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_products = (
            num_products
        )

        self.embedding_dim = (
            embedding_dim
        )

        self.num_refinement_steps = (
            num_refinement_steps
        )

        self.num_inner_steps = (
            num_inner_steps
        )

        self.update_scale = (
            update_scale
        )

        self.temperature = (
            temperature
        )

        # =====================================================
        # Product Embedding
        # =====================================================
        #
        # 0 = PAD
        #
        # 논문의 Random embedding variant에 대응.
        # =====================================================

        self.product_embedding = (
            nn.Embedding(
                num_embeddings=num_products,
                embedding_dim=embedding_dim,
                padding_idx=0,
            )
        )

        # =====================================================
        # Behavior Embedding
        # =====================================================
        #
        # MCM-specific extension
        #
        # SELECT / FITTING /
        # WISHLIST_ADD / REMOVE
        # =====================================================

        self.behavior_embedding = (
            nn.Embedding(
                num_embeddings=(
                    num_behavior_types
                ),
                embedding_dim=embedding_dim,
                padding_idx=0,
            )
        )

        # interaction representation 정규화
        self.interaction_norm = (
            nn.LayerNorm(
                embedding_dim
            )
        )

        self.interaction_dropout = (
            nn.Dropout(dropout)
        )

        # =====================================================
        # Shared recursive block f_phi
        # =====================================================

        self.recursive_block = (
            RecursiveBlock(
                embedding_dim=(
                    embedding_dim
                ),
                hidden_dim=hidden_dim,
                num_layers=recursive_layers,
                dropout=dropout,
            )
        )

        # =====================================================
        # Correction Gates
        #
        # 논문:
        #
        # g_t = sigmoid(
        #     W_t [x || y_t]
        # )
        #
        # W_t는 refinement step별 learnable parameter.
        # =====================================================

        self.correction_gates = (
            nn.ModuleList([
                nn.Linear(
                    embedding_dim * 2,
                    embedding_dim,
                )

                for _ in range(
                    num_refinement_steps
                )
            ])
        )

        self.preference_norm = (
            nn.LayerNorm(
                embedding_dim
            )
        )

        self._initialize_weights()

    # =========================================================
    # Initialization
    # =========================================================

    def _initialize_weights(
        self,
    ):

        nn.init.normal_(
            self.product_embedding.weight,
            mean=0.0,
            std=0.02,
        )

        nn.init.normal_(
            self.behavior_embedding.weight,
            mean=0.0,
            std=0.02,
        )

        # PAD embedding은 0 유지
        with torch.no_grad():

            self.product_embedding.weight[
                0
            ].zero_()

            self.behavior_embedding.weight[
                0
            ].zero_()

    # =========================================================
    # Interaction Encoder
    # =========================================================

    def encode_interactions(
        self,
        product_ids: torch.Tensor,
        behavior_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        product_ids:
            [batch, seq_len]

        behavior_ids:
            [batch, seq_len]

        output:
            [batch, seq_len, embedding_dim]
        """

        product_embeddings = (
            self.product_embedding(
                product_ids
            )
        )

        behavior_embeddings = (
            self.behavior_embedding(
                behavior_ids
            )
        )

        # MCM extension
        interaction_embeddings = (
            product_embeddings
            + behavior_embeddings
        )

        interaction_embeddings = (
            self.interaction_norm(
                interaction_embeddings
            )
        )

        interaction_embeddings = (
            self.interaction_dropout(
                interaction_embeddings
            )
        )

        return interaction_embeddings

    # =========================================================
    # Static Context x
    # =========================================================

    def masked_mean_pooling(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        논문의:

        x = masked mean(history embeddings)

        embeddings:
            [B, L, D]

        attention_mask:
            [B, L]

        output:
            [B, D]
        """

        mask = (
            attention_mask
            .unsqueeze(-1)
            .float()
        )

        masked_embeddings = (
            embeddings * mask
        )

        summed = (
            masked_embeddings.sum(
                dim=1
            )
        )

        counts = (
            mask.sum(
                dim=1
            )
            .clamp(
                min=1.0
            )
        )

        context = (
            summed / counts
        )

        return context

    # =========================================================
    # Recursive Refinement
    # =========================================================

    def refine(
        self,
        context: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        논문 상태:

        x   = static context
        y_t = preference state
        z_t = latent state

        초기:

        y_0 = x
        z_0 = 0

        반환:
            각 outer refinement step의 y
            deep supervision에 사용
        """

        x = context

        # y0 = x
        y = x

        # z0 = 0
        z = torch.zeros_like(
            x
        )

        preference_states = []

        # =====================================================
        # Outer refinement loop
        # =====================================================

        for step in range(
            self.num_refinement_steps
        ):

            # -------------------------------------------------
            # Inner recursion
            #
            # z_t^(j)
            # =
            # f_phi(
            #   [x || y_t || z_t^(j-1)]
            # )
            # -------------------------------------------------

            z_inner = z

            for _ in range(
                self.num_inner_steps
            ):

                recursive_input = (
                    torch.cat(
                        [
                            x,
                            y,
                            z_inner,
                        ],
                        dim=-1,
                    )
                )

                z_inner = (
                    self.recursive_block(
                        recursive_input
                    )
                )

            # -------------------------------------------------
            # Evidence-anchored Correction Gate
            #
            # g_t =
            # sigmoid(
            #   W_t [x || y_t]
            # )
            # -------------------------------------------------

            gate_input = (
                torch.cat(
                    [
                        x,
                        y,
                    ],
                    dim=-1,
                )
            )

            gate = torch.sigmoid(
                self.correction_gates[
                    step
                ](
                    gate_input
                )
            )

            # -------------------------------------------------
            # corrected z
            #
            # z_t =
            # (1 - g_t) * z_inner
            # +
            # g_t * x
            # -------------------------------------------------

            z = (
                (1.0 - gate)
                * z_inner
                +
                gate * x
            )

            # -------------------------------------------------
            # Preference Refinement
            #
            # Delta_t =
            # tanh(
            #   f_phi(
            #     [x || y_t || z_t]
            #   )
            # )
            #
            # y_(t+1) =
            # y_t + L * Delta_t
            # -------------------------------------------------

            preference_input = (
                torch.cat(
                    [
                        x,
                        y,
                        z,
                    ],
                    dim=-1,
                )
            )

            delta = torch.tanh(
                self.recursive_block(
                    preference_input
                )
            )

            y = (
                y
                + self.update_scale
                * delta
            )

            y = (
                self.preference_norm(
                    y
                )
            )

            preference_states.append(
                y
            )

        return preference_states

    # =========================================================
    # Product Scoring
    # =========================================================

    def score_products(
        self,
        preference: torch.Tensor,
    ) -> torch.Tensor:
        """
        preference:
            [B, D]

        output:
            [B, num_products]

        preference와 모든 상품 embedding 간 dot-product.
        """

        product_embeddings = (
            self.product_embedding.weight
        )

        # normalize해서 cosine-like score
        preference = F.normalize(
            preference,
            dim=-1,
        )

        product_embeddings = (
            F.normalize(
                product_embeddings,
                dim=-1,
            )
        )

        logits = (
            preference
            @ product_embeddings.T
        )

        logits = (
            logits
            / self.temperature
        )

        # PAD 상품은 절대 추천되지 않도록
        logits[:, 0] = (
            torch.finfo(
                logits.dtype
            ).min
        )

        return logits

    # =========================================================
    # Forward
    # =========================================================

    def forward(
        self,
        product_ids: torch.Tensor,
        behavior_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict:
        """
        dataset.py에서 그대로 들어오는 값.

        product_ids:
            [B, L]

        behavior_ids:
            [B, L]

        attention_mask:
            [B, L]
        """

        # -----------------------------------------
        # Interaction Embedding
        # -----------------------------------------

        interaction_embeddings = (
            self.encode_interactions(
                product_ids,
                behavior_ids,
            )
        )

        # -----------------------------------------
        # Static evidence context x
        # -----------------------------------------

        context = (
            self.masked_mean_pooling(
                interaction_embeddings,
                attention_mask,
            )
        )

        # -----------------------------------------
        # Recursive refinement
        # -----------------------------------------

        preference_states = (
            self.refine(
                context
            )
        )

        # -----------------------------------------
        # Deep supervision용
        # step별 logits
        # -----------------------------------------

        logits_per_step = [

            self.score_products(
                preference
            )

            for preference
            in preference_states
        ]

        return {
            "context":
                context,

            "preference_states":
                preference_states,

            "logits_per_step":
                logits_per_step,

            "final_logits":
                logits_per_step[-1],

            "final_preference":
                preference_states[-1],
        }


# =========================================================
# Deep Supervision Loss
# =========================================================

def recrec_loss(
    outputs: dict,
    target_product_ids: torch.Tensor,
) -> torch.Tensor:
    """
    논문:

    L_total
    =
    1/T Σ CE(y_t, target)

    모든 recursive refinement step에
    supervision을 걸어준다.
    """

    logits_per_step = (
        outputs[
            "logits_per_step"
        ]
    )

    losses = []

    for logits in logits_per_step:

        loss = F.cross_entropy(
            logits,
            target_product_ids,
        )

        losses.append(
            loss
        )

    total_loss = torch.stack(
        losses
    ).mean()

    return total_loss


# =========================================================
# Simple Test
# =========================================================

if __name__ == "__main__":

    batch_size = 4
    seq_len = 64

    # dataset.py 기준
    # 실제 134개 상품 + PAD
    num_products = 135

    model = RecRec(
        num_products=num_products,

        embedding_dim=128,
        hidden_dim=256,

        num_refinement_steps=6,
        num_inner_steps=3,

        recursive_layers=3,

        update_scale=0.1,
        temperature=0.07,
    )

    # 테스트용 fake batch
    product_ids = torch.zeros(
        batch_size,
        seq_len,
        dtype=torch.long,
    )

    behavior_ids = torch.zeros(
        batch_size,
        seq_len,
        dtype=torch.long,
    )

    attention_mask = torch.zeros(
        batch_size,
        seq_len,
        dtype=torch.bool,
    )

    # 실제 history 있다고 가정
    product_ids[:, -3:] = torch.tensor(
        [8, 8, 8]
    )

    behavior_ids[:, -3:] = torch.tensor(
        [1, 2, 3]
    )

    attention_mask[:, -3:] = True

    targets = torch.tensor(
        [29, 10, 51, 82],
        dtype=torch.long,
    )

    outputs = model(
        product_ids=product_ids,
        behavior_ids=behavior_ids,
        attention_mask=attention_mask,
    )

    loss = recrec_loss(
        outputs=outputs,
        target_product_ids=targets,
    )

    print(
        "context:",
        outputs["context"].shape,
    )

    print(
        "refinement steps:",
        len(
            outputs[
                "preference_states"
            ]
        ),
    )

    print(
        "final logits:",
        outputs[
            "final_logits"
        ].shape,
    )

    print(
        "loss:",
        loss.item(),
    )