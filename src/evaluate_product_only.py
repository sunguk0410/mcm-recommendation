import math

import torch
from torch.utils.data import DataLoader

from dataset import create_datasets
from recrec import RecRec


# =========================================================
# Config
# =========================================================

JSONL_PATH = (
    "synthetic_interactions_v2.jsonl"
)

CATALOG_PATH = (
    "MCM_제품리스트_통합_추천모델용.xlsx"
)

CHECKPOINT_PATH = (
    "checkpoints/"
    "recrec_v2_product_only_best.pt"
)

MAX_SEQ_LEN = 64

BATCH_SIZE = 256

KS = [
    1,
    5,
    10,
]


# =========================================================
# Device
# =========================================================

def get_device():

    if torch.cuda.is_available():

        return torch.device(
            "cuda"
        )

    return torch.device(
        "cpu"
    )


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

    hit_sums = {
        k: 0.0
        for k in KS
    }

    ndcg_sums = {
        k: 0.0
        for k in KS
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

        # PAD index 제외
        logits[:, 0] = float(
            "-inf"
        )

        topk_indices = (
            torch.topk(
                logits,
                k=max(KS),
                dim=1,
            )
            .indices
        )

        batch_size = (
            targets.size(0)
        )

        total_samples += (
            batch_size
        )

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

            target_rank = None

            for rank, product_id in enumerate(
                ranking,
                start=1,
            ):

                if (
                    product_id
                    == target
                ):

                    target_rank = (
                        rank
                    )

                    break

            for k in KS:

                if (
                    target_rank
                    is not None
                    and target_rank <= k
                ):

                    hit_sums[
                        k
                    ] += 1.0

                    ndcg_sums[
                        k
                    ] += (
                        1.0
                        / math.log2(
                            target_rank
                            + 1
                        )
                    )

    metrics = {}

    for k in KS:

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
        "Product-Only Evaluation"
    )

    print(
        "================================="
    )

    print(
        f"Device: {device}"
    )

    print(
        "================================="
    )

    # =====================================================
    # Dataset
    # =====================================================

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

        # 핵심
        use_behavior=False,
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

        pin_memory=(
            device.type == "cuda"
        ),
    )

    # =====================================================
    # Load Checkpoint
    # =====================================================

    checkpoint = torch.load(

        CHECKPOINT_PATH,

        map_location=device,
    )

    config = checkpoint[
        "model_config"
    ]

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

    # =====================================================
    # Evaluate
    # =====================================================

    metrics = evaluate(

        model=model,

        dataloader=test_loader,

        device=device,
    )

    print()
    print(
        "================================="
    )

    print(
        "Product Only Result"
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