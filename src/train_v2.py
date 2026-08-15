import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import create_datasets
from recrec import RecRec, recrec_loss


# =========================================================
# Config
# =========================================================

SEED = 42

JSONL_PATH = "synthetic_interactions_v2.jsonl"
CATALOG_PATH = "MCM_제품리스트_통합_추천모델용.xlsx"

CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "recrec_v2_best.pt",
)

# Dataset
MAX_SEQ_LEN = 64

# Training
BATCH_SIZE = 128
NUM_EPOCHS = 30

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

GRAD_CLIP_NORM = 1.0

# Early stopping
PATIENCE = 5

# Model
EMBEDDING_DIM = 128
HIDDEN_DIM = 256

NUM_REFINEMENT_STEPS = 6
NUM_INNER_STEPS = 3

RECURSIVE_LAYERS = 3

UPDATE_SCALE = 0.1
TEMPERATURE = 0.07
DROPOUT = 0.1


# =========================================================
# Seed
# =========================================================

def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 재현성을 조금 더 높이기 위한 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# Device
# =========================================================

def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    # Apple Silicon 대응
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


# =========================================================
# Train One Epoch
# =========================================================

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
):

    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch_idx, batch in enumerate(
        dataloader
    ):

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

        # -----------------------------------------
        # Gradient 초기화
        # -----------------------------------------

        optimizer.zero_grad()

        # -----------------------------------------
        # Forward
        # -----------------------------------------

        outputs = model(
            product_ids=product_ids,
            behavior_ids=behavior_ids,
            attention_mask=attention_mask,
        )

        loss = recrec_loss(
            outputs=outputs,
            target_product_ids=targets,
        )

        # -----------------------------------------
        # Backpropagation
        # -----------------------------------------

        loss.backward()

        # -----------------------------------------
        # Gradient Clipping
        #
        # recursive 구조에서 gradient 폭주 방지
        # -----------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRAD_CLIP_NORM,
        )

        optimizer.step()

        # -----------------------------------------
        # Loss 기록
        # -----------------------------------------

        batch_size = (
            product_ids.size(0)
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += (
            batch_size
        )

        # 중간 진행 상황
        if (
            batch_idx + 1
        ) % 100 == 0:

            current_avg = (
                total_loss
                / total_samples
            )

            print(
                f"  Batch "
                f"{batch_idx + 1}"
                f"/{len(dataloader)}"
                f" | loss="
                f"{current_avg:.4f}"
            )

    average_loss = (
        total_loss
        / total_samples
    )

    return average_loss


# =========================================================
# Validation
# =========================================================

@torch.no_grad()
def validate(
    model,
    dataloader,
    device,
):

    model.eval()

    total_loss = 0.0
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

        loss = recrec_loss(
            outputs=outputs,
            target_product_ids=targets,
        )

        batch_size = (
            product_ids.size(0)
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += (
            batch_size
        )

    average_loss = (
        total_loss
        / total_samples
    )

    return average_loss


# =========================================================
# Save Checkpoint
# =========================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    val_loss,
    mapper,
    path,
):

    checkpoint = {

        "epoch":
            epoch,

        "val_loss":
            val_loss,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        # 모델 구조 복원용
        "model_config": {
            "num_products":
                mapper.num_products,

            "embedding_dim":
                EMBEDDING_DIM,

            "hidden_dim":
                HIDDEN_DIM,

            "num_behavior_types":
                5,

            "num_refinement_steps":
                NUM_REFINEMENT_STEPS,

            "num_inner_steps":
                NUM_INNER_STEPS,

            "recursive_layers":
                RECURSIVE_LAYERS,

            "update_scale":
                UPDATE_SCALE,

            "temperature":
                TEMPERATURE,

            "dropout":
                DROPOUT,
        },

        # 추론 시 productId mapping 복원용
        "product_to_index":
            mapper.product_to_index,

        "index_to_product":
            mapper.index_to_product,
    }

    torch.save(
        checkpoint,
        path,
    )


# =========================================================
# Main
# =========================================================

def main():

    # -----------------------------------------
    # Seed
    # -----------------------------------------

    set_seed(SEED)

    # -----------------------------------------
    # Device
    # -----------------------------------------

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
        seed=SEED,
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

    # -----------------------------------------
    # DataLoader
    # -----------------------------------------

    train_loader = DataLoader(
        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=0,

        pin_memory=(
            device.type == "cuda"
        ),
    )

    val_loader = DataLoader(
        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0,

        pin_memory=(
            device.type == "cuda"
        ),
    )

    # test는 나중 evaluate.py에서 사용
    _ = test_dataset

    # -----------------------------------------
    # Model
    # -----------------------------------------

    model = RecRec(
        num_products=(
            mapper.num_products
        ),

        embedding_dim=(
            EMBEDDING_DIM
        ),

        hidden_dim=(
            HIDDEN_DIM
        ),

        num_behavior_types=5,

        num_refinement_steps=(
            NUM_REFINEMENT_STEPS
        ),

        num_inner_steps=(
            NUM_INNER_STEPS
        ),

        recursive_layers=(
            RECURSIVE_LAYERS
        ),

        update_scale=(
            UPDATE_SCALE
        ),

        temperature=(
            TEMPERATURE
        ),

        dropout=(
            DROPOUT
        ),
    )

    model = model.to(
        device
    )

    # -----------------------------------------
    # Optimizer
    # -----------------------------------------

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),

            lr=LEARNING_RATE,

            weight_decay=(
                WEIGHT_DECAY
            ),
        )
    )

    # -----------------------------------------
    # Learning Rate Scheduler
    #
    # validation loss가 정체되면
    # learning rate 감소
    # -----------------------------------------

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,

            mode="min",

            factor=0.5,

            patience=2,

            min_lr=1e-6,
        )
    )

    # -----------------------------------------
    # Checkpoint Directory
    # -----------------------------------------

    Path(
        CHECKPOINT_DIR
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------
    # Training
    # -----------------------------------------

    best_val_loss = (
        float("inf")
    )

    patience_counter = 0

    print(
        "\nStart Training\n"
    )

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):

        print(
            "================================="
        )

        print(
            f"Epoch "
            f"{epoch}"
            f"/{NUM_EPOCHS}"
        )

        print(
            "================================="
        )

        # -------------------------------------
        # Train
        # -------------------------------------

        train_loss = (
            train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
            )
        )

        # -------------------------------------
        # Validation
        # -------------------------------------

        val_loss = (
            validate(
                model=model,
                dataloader=val_loader,
                device=device,
            )
        )

        # -------------------------------------
        # Scheduler
        # -------------------------------------

        scheduler.step(
            val_loss
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        print()

        print(
            f"Train Loss : "
            f"{train_loss:.4f}"
        )

        print(
            f"Val Loss   : "
            f"{val_loss:.4f}"
        )

        print(
            f"Learning Rate : "
            f"{current_lr:.8f}"
        )

        # -------------------------------------
        # Best Model
        # -------------------------------------

        if (
            val_loss
            < best_val_loss
        ):

            best_val_loss = (
                val_loss
            )

            patience_counter = 0

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                mapper=mapper,
                path=BEST_MODEL_PATH,
            )

            print(
                f"Best model saved -> "
                f"{BEST_MODEL_PATH}"
            )

        else:

            patience_counter += 1

            print(
                f"No improvement "
                f"({patience_counter}"
                f"/{PATIENCE})"
            )

        # -------------------------------------
        # Early Stopping
        # -------------------------------------

        if (
            patience_counter
            >= PATIENCE
        ):

            print(
                "\nEarly stopping."
            )

            break

    print(
        "\n================================="
    )

    print(
        "Training Finished"
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.4f}"
    )

    print(
        f"Best model: "
        f"{BEST_MODEL_PATH}"
    )

    print(
        "================================="
    )


if __name__ == "__main__":
    main()