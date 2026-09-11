"""Generate single-category sessions from fixed CLIP visual preferences."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import shutil

import numpy as np
import torch

from .synthetic_generator import BEHAVIOR_PROFILES, generate_actions, load_products


def load_visual_features(path, products, image_dir):
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    rows = artifact["products"]
    mapping = {row["product_id"]: i for i, row in enumerate(rows)}
    if len(mapping) != len(rows) or set(mapping) != {p["productId"] for p in products}:
        raise ValueError("Embedding and catalog product IDs must match exactly")
    order = []
    for product in products:
        index = mapping[product["productId"]]
        row = rows[index]
        if row["product_code"] != product["productCode"] or row["category"] != product["category"]:
            raise ValueError("Embedding metadata differs from catalog; regenerate embeddings")
        image_path = image_dir / (product["productCode"] + ".png")
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != row["image_sha256"]:
            raise ValueError(f"Image changed: {image_path}; regenerate embeddings")
        order.append(index)
    features = artifact["features"].float()
    if features.ndim != 2 or features.shape[0] != len(rows) or not torch.isfinite(features).all():
        raise ValueError("Invalid embedding matrix")
    if (features.norm(dim=1) < 1e-8).any():
        raise ValueError("Zero embedding vector")
    features = torch.nn.functional.normalize(features[order], dim=1).numpy()
    return features, artifact


def generate_session(session_id, category, products, features, rng):
    genders = [g for g in ("MALE", "FEMALE") if sum(
        p["category"] == category and p["gender"] in (g, "UNISEX") for p in products
    ) >= 2]
    if not genders:
        raise ValueError(f"Category {category} needs at least two compatible products")
    gender = str(rng.choice(genders))
    pool = np.array([i for i, p in enumerate(products)
                     if p["category"] == category and p["gender"] in (gender, "UNISEX")])
    behavior = str(rng.choice(list(BEHAVIOR_PROFILES)))
    profile = BEHAVIOR_PROFILES[behavior]
    exploration = {"DECISIVE_BUYER": .10, "FASHION_EXPLORER": .20, "BROAD_EXPLORER": .35}[behavior]
    temperature = float(rng.uniform(.16, .28))
    anchor = int(rng.choice(pool))
    taste = features[anchor]
    similarity = features[pool] @ taste
    # Fixed session scaling: removing candidates must not redefine their affinity.
    low, high = float(similarity.min()), float(similarity.max())
    span = max(high - low, 1e-6)
    affinity = np.clip((similarity - low) / span, 0., 1.)
    if high - low < 1e-6:
        affinity[:] = .5
    # Leave alternatives unseen instead of forcing every item into each session.
    maximum = max(2, min(14, int(np.ceil(len(pool) * .7))))
    count = int(np.clip(rng.poisson(profile["lambda_products"] * .65), 2, maximum))
    available = list(range(len(pool)))
    interactions, selected = [], []
    feedback = np.zeros(features.shape[1], dtype=np.float32)
    previous = None
    for _ in range(count):
        a = np.array(available)
        recent = np.full(len(a), .5) if previous is None else np.clip(
            (features[pool[a]] @ features[previous] - low) / span, 0., 1.
        )
        feedback_score = np.clip(features[pool[a]] @ feedback, -.5, .5)
        scores = .80 * affinity[a] + .20 * recent + .15 * feedback_score
        weights = np.exp((scores - scores.max()) / temperature)
        probabilities = (1 - exploration) * weights / weights.sum() + exploration / len(a)
        local = int(rng.choice(a, p=probabilities))
        index = int(pool[local])
        product = products[index]
        actions = generate_actions(float(affinity[local]), profile)
        for action in actions:
            interactions.append({"productId": product["productId"],
                                 "interactionType": action, "sequenceNo": len(interactions) + 1})
        # Final retained state: removals cancel additions instead of reinforcing them.
        fitted = "FITTING_ADD" in actions and "FITTING_REMOVE" not in actions
        wished = "WISHLIST_ADD" in actions and "WISHLIST_REMOVE" not in actions
        strength = .3 * fitted + .6 * wished
        if "FITTING_REMOVE" in actions:
            strength -= .15
        if "WISHLIST_REMOVE" in actions:
            strength -= .3
        feedback = .8 * feedback + strength * features[index]
        feedback /= max(1., float(np.linalg.norm(feedback)))
        previous = index
        selected.append(product["productId"])
        available.remove(local)
    return {"sessionId": session_id, "generator": "clip_single_category_v1",
            "category": category, "gender": gender, "behaviorType": behavior,
            # Diagnostics only; dataset.py feeds only interactions to RecRec.
            "anchorProductId": products[anchor]["productId"],
            "explorationProbability": exploration, "temperature": temperature,
            "selectedProductIds": selected, "interactions": interactions}


def validate_sessions(sessions, products):
    by_id = {p["productId"]: p for p in products}
    counts = Counter()
    for session in sessions:
        selected, current, states = [], None, set()
        for seq, event in enumerate(session["interactions"], 1):
            pid, action = event["productId"], event["interactionType"]
            p = by_id[pid]
            assert event["sequenceNo"] == seq
            assert p["category"] == session["category"]
            assert p["gender"] in (session["gender"], "UNISEX")
            if action == "PRODUCT_SELECT":
                assert pid not in selected
                selected.append(pid)
                current, states = pid, set()
            else:
                assert current == pid
                assert action in {"FITTING_ADD", "FITTING_REMOVE", "WISHLIST_ADD", "WISHLIST_REMOVE"}
                kind, operation = action.rsplit("_", 1)
                if operation == "ADD":
                    assert kind not in states
                    states.add(kind)
                else:
                    assert kind in states
                    states.remove(kind)
            counts[action] += 1
        assert selected == session["selectedProductIds"] and len(selected) >= 2
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, default=Path("artifacts/image_similarity/embeddings.pt"))
    parser.add_argument("--images", type=Path, default=Path("images"))
    parser.add_argument("--output", type=Path, default=Path("synthetic_interactions.jsonl"))
    parser.add_argument("--sessions", type=int, default=9000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.sessions < 1:
        parser.error("sessions must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    products = load_products()
    features, artifact = load_visual_features(args.embeddings, products, args.images)
    categories = sorted({p["category"] for p in products})
    schedule = [categories[i % len(categories)] for i in range(args.sessions)]
    rng.shuffle(schedule)
    sessions = [generate_session(i + 1, c, products, features, rng) for i, c in enumerate(schedule)]
    counts = validate_sessions(sessions, products)
    encoded = ''.join(json.dumps(s, ensure_ascii=False) + '\n' for s in sessions).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    backup = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        previous = hashlib.sha256(args.output.read_bytes()).hexdigest()
        if previous != digest:
            backup = Path("artifacts/synthetic_backups") / (previous + ".jsonl")
            backup.parent.mkdir(parents=True, exist_ok=True)
            if not backup.exists():
                shutil.copy2(args.output, backup)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(args.output)
    summary = {"generator": "clip_single_category_v1", "seed": args.seed,
               "sessions": len(sessions), "interactions": sum(counts.values()), "actions": counts,
               "categories": dict(Counter(s["category"] for s in sessions)),
               "products_selected": len({p for s in sessions for p in s["selectedProductIds"]}),
               "training_targets": sum(len(s["selectedProductIds"]) - 1 for s in sessions),
               "model": artifact["model"], "model_revision": artifact["revision"],
               "embedding_sha256": hashlib.sha256(args.embeddings.read_bytes()).hexdigest(),
               "output_sha256": digest, "backup": str(backup) if backup else None,
               "note": "Simulation assumptions, not observed customer preferences. No model retraining performed."}
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
