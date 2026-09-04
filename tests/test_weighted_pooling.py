import torch

from src.recrec import RecRec


def build_model(recency_decay=0.25):
    return RecRec(
        num_products=8,
        embedding_dim=2,
        hidden_dim=4,
        num_behavior_types=6,
        num_refinement_steps=1,
        num_inner_steps=1,
        recursive_layers=2,
        dropout=0.0,
        recency_decay=recency_decay,
    )


def test_weighted_pooling_ignores_left_padding():
    model = build_model()
    embeddings = torch.tensor([[[99.0, 99.0], [1.0, 0.0], [0.0, 1.0]]])
    behavior_ids = torch.tensor([[0, 1, 1]])
    attention_mask = torch.tensor([[False, True, True]])

    context = model.weighted_pooling(
        embeddings,
        behavior_ids,
        attention_mask,
    )

    assert context[0, 0] < context[0, 1]
    assert torch.allclose(context.sum(dim=-1), torch.ones(1))


def test_recent_interaction_has_more_weight():
    model = build_model(recency_decay=0.5)
    embeddings = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    behavior_ids = torch.tensor([[1, 1]])
    attention_mask = torch.tensor([[True, True]])

    context = model.weighted_pooling(
        embeddings,
        behavior_ids,
        attention_mask,
    )

    assert context[0, 1] > context[0, 0]


def test_action_strength_changes_context():
    model = build_model(recency_decay=0.0)
    embeddings = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    behavior_ids = torch.tensor([[1, 4]])  # SELECT, WISHLIST_ADD
    attention_mask = torch.tensor([[True, True]])

    context = model.weighted_pooling(
        embeddings,
        behavior_ids,
        attention_mask,
    )

    assert torch.allclose(context, torch.tensor([[1.0 / 3.0, 2.0 / 3.0]]))


def test_single_interaction_is_unchanged():
    model = build_model()
    embeddings = torch.tensor([[[0.0, 0.0], [2.0, 3.0]]])
    behavior_ids = torch.tensor([[0, 2]])
    attention_mask = torch.tensor([[False, True]])

    context = model.weighted_pooling(
        embeddings,
        behavior_ids,
        attention_mask,
    )

    assert torch.allclose(context, torch.tensor([[2.0, 3.0]]))


def test_all_padding_returns_zero_context():
    model = build_model()
    embeddings = torch.ones(1, 3, 2)
    behavior_ids = torch.zeros(1, 3, dtype=torch.long)
    attention_mask = torch.zeros(1, 3, dtype=torch.bool)

    context = model.weighted_pooling(
        embeddings,
        behavior_ids,
        attention_mask,
    )

    assert torch.equal(context, torch.zeros(1, 2))
