import torch
from torch.utils.data import DataLoader

from dataset import create_datasets
from recrec import RecRec


# =========================================================
# Config
# =========================================================

JSONL_PATH = "synthetic_interactions_v2.jsonl"
CATALOG_PATH = "MCM_제품리스트_통합_추천모델용.xlsx"

CHECKPOINT_PATH = "checkpoints/recrec_v2_best.pt"

BATCH_SIZE = 256
MAX_SEQ_LEN = 64


# =========================================================
# Device
# =========================================================

def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# =========================================================
# Evaluation
# =========================================================

@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
):

    model.eval()

    ks = [1, 5, 10]

    hit_sums = {
        k: 0.0
        for k in ks
    }

    ndcg_sums = {
        k: 0.0
        for k in ks
    }

    total_samples = 0

    for batch in dataloader:

        product_ids = (
            batch["product_ids"]
            .to(device)
        )

        behavior_ids = (
            batch["behavior_ids"]
            .to(device)
        )

        attention_mask = (
            batch["attention_mask"]
            .to(device)
        )

        targets = (
            batch["target_product_id"]
            .to(device)
        )

        outputs = model(
            product_ids=product_ids,
            behavior_ids=behavior_ids,
            attention_mask=attention_mask,
        )

        logits = outputs[
            "final_logits"
        ]

        # -----------------------------------------
        # 최대 Top-10 뽑기
        # -----------------------------------------

        _, topk_indices = torch.topk(
            logits,
            k=max(ks),
            dim=1,
        )

        batch_size = (
            targets.size(0)
        )

        total_samples += (
            batch_size
        )

        # =========================================
        # 각 sample 평가
        # =========================================

        for i in range(
            batch_size
        ):

            target = int(
                targets[i].item()
            )

            ranking = (
                topk_indices[i]
                .tolist()
            )

            # -------------------------------------
            # target rank 찾기
            #
            # rank는 1부터 시작
            # -------------------------------------

            target_rank = None

            for rank, product_id in enumerate(
                ranking,
                start=1,
            ):

                if product_id == target:

                    target_rank = rank
                    break

            # -------------------------------------
            # HR@K + NDCG@K
            # -------------------------------------

            for k in ks:

                if (
                    target_rank is not None
                    and target_rank <= k
                ):

                    # Hit
                    hit_sums[k] += 1.0

                    # NDCG
                    ndcg_sums[k] += (
                        1.0
                        / torch.log2(
                            torch.tensor(
                                target_rank + 1,
                                dtype=torch.float32,
                            )
                        ).item()
                    )

    # =====================================================
    # 평균
    # =====================================================

    metrics = {}

    for k in ks:

        metrics[
            f"HR@{k}"
        ] = (
            hit_sums[k]
            / total_samples
        )

        metrics[
            f"NDCG@{k}"
        ] = (
            ndcg_sums[k]
            / total_samples
        )

    return metrics


# =========================================================
# Main
# =========================================================

def main():

    device = get_device()

    print(
        "================================="
    )

    print(
        f"Device: {device}"
    )

    print(
        "================================="
    )

    # -----------------------------------------
    # Dataset
    # -----------------------------------------

    (
        train_dataset,
        val_dataset,
        test_dataset,
        mapper,
    ) = create_datasets(
        jsonl_path=JSONL_PATH,
        catalog_path=CATALOG_PATH,
        max_seq_len=MAX_SEQ_LEN,
        seed=42,
    )

    print(
        f"Test samples: "
        f"{len(test_dataset)}"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # -----------------------------------------
    # Checkpoint
    # -----------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    config = checkpoint[
        "model_config"
    ]

    # -----------------------------------------
    # Model 복원
    # -----------------------------------------

    model = RecRec(
        num_products=(
            config[
                "num_products"
            ]
        ),

        embedding_dim=(
            config[
                "embedding_dim"
            ]
        ),

        hidden_dim=(
            config[
                "hidden_dim"
            ]
        ),

        num_behavior_types=(
            config[
                "num_behavior_types"
            ]
        ),

        num_refinement_steps=(
            config[
                "num_refinement_steps"
            ]
        ),

        num_inner_steps=(
            config[
                "num_inner_steps"
            ]
        ),

        recursive_layers=(
            config[
                "recursive_layers"
            ]
        ),

        update_scale=(
            config[
                "update_scale"
            ]
        ),

        temperature=(
            config[
                "temperature"
            ]
        ),

        dropout=(
            config[
                "dropout"
            ]
        ),
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        device
    )

    print(
        f"Loaded checkpoint "
        f"from epoch "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Best val loss: "
        f"{checkpoint['val_loss']:.4f}"
    )

    # -----------------------------------------
    # Evaluate
    # -----------------------------------------

    metrics = evaluate(
        model=model,
        dataloader=test_loader,
        device=device,
    )

    print(
        "\n================================="
    )

    print(
        "Evaluation Result"
    )

    print(
        "================================="
    )

    print(
        f"HR@1    : "
        f"{metrics['HR@1']:.4f}"
    )

    print(
        f"HR@5    : "
        f"{metrics['HR@5']:.4f}"
    )

    print(
        f"HR@10   : "
        f"{metrics['HR@10']:.4f}"
    )

    print(
        f"NDCG@1  : "
        f"{metrics['NDCG@1']:.4f}"
    )

    print(
        f"NDCG@5  : "
        f"{metrics['NDCG@5']:.4f}"
    )

    print(
        f"NDCG@10 : "
        f"{metrics['NDCG@10']:.4f}"
    )

    print(
        "================================="
    )


if __name__ == "__main__":
    main()