import random
import math
from collections import Counter

from dataset import create_datasets


JSONL_PATH = "synthetic_interactions.jsonl"
CATALOG_PATH = "MCM_제품리스트_통합_추천모델용.xlsx"

MAX_SEQ_LEN = 64
SEED = 42

KS = [1, 5, 10]


def calculate_metrics(rankings, targets):
    hit_sums = {k: 0.0 for k in KS}
    ndcg_sums = {k: 0.0 for k in KS}

    total = len(targets)

    for ranking, target in zip(rankings, targets):

        target_rank = None

        for rank, item_id in enumerate(ranking, start=1):
            if item_id == target:
                target_rank = rank
                break

        for k in KS:
            if target_rank is not None and target_rank <= k:

                hit_sums[k] += 1.0

                ndcg_sums[k] += (
                    1.0 / math.log2(target_rank + 1)
                )

    metrics = {}

    for k in KS:
        metrics[f"HR@{k}"] = hit_sums[k] / total
        metrics[f"NDCG@{k}"] = ndcg_sums[k] / total

    return metrics


def print_metrics(name, metrics):

    print()
    print("=================================")
    print(name)
    print("=================================")

    for k in KS:
        print(f"HR@{k:<2}   : {metrics[f'HR@{k}']:.4f}")

    for k in KS:
        print(f"NDCG@{k:<2} : {metrics[f'NDCG@{k}']:.4f}")


def main():

    random.seed(SEED)

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

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples : {len(test_dataset)}")

    # target은 이미 model 내부 index
    train_targets = [
        int(train_dataset[i]["target_product_id"].item())
        for i in range(len(train_dataset))
    ]

    test_targets = [
        int(test_dataset[i]["target_product_id"].item())
        for i in range(len(test_dataset))
    ]

    # PAD=0 제외
    num_products = len(mapper.product_to_index)

    product_indices = list(
        range(1, num_products + 1)
    )

    # =====================================================
    # 1. Random baseline
    # =====================================================

    random_rankings = []

    for _ in test_targets:

        ranking = random.sample(
            product_indices,
            k=min(10, len(product_indices))
        )

        random_rankings.append(ranking)

    random_metrics = calculate_metrics(
        random_rankings,
        test_targets
    )

    # =====================================================
    # 2. Most Popular baseline
    # =====================================================

    popularity = Counter(train_targets)

    popular_ranking = [
        product_id
        for product_id, count
        in popularity.most_common()
    ]

    # 혹시 train에서 target으로 등장하지 않은 상품 보충
    remaining = [
        product_id
        for product_id in product_indices
        if product_id not in popularity
    ]

    popular_ranking.extend(remaining)

    popular_top10 = popular_ranking[:10]

    popularity_rankings = [
        popular_top10
        for _ in test_targets
    ]

    popularity_metrics = calculate_metrics(
        popularity_rankings,
        test_targets
    )

    # =====================================================
    # Result
    # =====================================================

    print_metrics(
        "Random Baseline",
        random_metrics
    )

    print_metrics(
        "Most Popular Baseline",
        popularity_metrics
    )


if __name__ == "__main__":
    main()