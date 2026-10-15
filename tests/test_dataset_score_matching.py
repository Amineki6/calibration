import torch

from methods.score_matching_dataset import DatasetScoreMatchingLoss


def test_binary_hand_computed():
    # Group 0: pos 0.8, neg 0.2; group 1: pos 0.6, neg 0.4 -> pos and neg std are both 0.2 / sqrt(2)
    loss_fn = DatasetScoreMatchingLoss(4, min_subgroup_count=1, device="cpu")
    probs = torch.tensor([[0.8], [0.2], [0.6], [0.4]])
    loss = loss_fn(probs, torch.tensor([1, 0, 1, 0]), torch.tensor([0, 0, 1, 1]), torch.arange(4))
    assert torch.isclose(loss, torch.tensor(0.2 / 2**0.5))


def test_multiclass_matches_reference():
    torch.manual_seed(0)
    n, k, n_groups = 90, 3, 3
    probs = torch.rand(n, k).softmax(dim=1)
    labels = torch.arange(n) % k
    groups = (torch.arange(n) // k) % n_groups
    loss_fn = DatasetScoreMatchingLoss(n, min_subgroup_count=1, device="cpu", num_classes=k)
    loss = loss_fn(probs, labels, groups, torch.arange(n))

    # E[p_j | y=c, g] for all (c, j, g); std over g, mean over (c, j)
    means = torch.stack([
        torch.stack([probs[(labels == c) & (groups == g)].mean(dim=0) for g in range(n_groups)])
        for c in range(k)
    ])  # (c, g, j)
    assert torch.isclose(loss, means.std(dim=1).mean())


def test_multiclass_gradient_only_through_batch():
    n, k = 12, 3
    labels, groups = torch.arange(n) % k, (torch.arange(n) // k) % 2
    loss_fn = DatasetScoreMatchingLoss(n, min_subgroup_count=1, device="cpu", num_classes=k)
    loss_fn.update(torch.rand(n, k).softmax(dim=1), labels, groups, torch.arange(n))
    logits = torch.randn(2, k, requires_grad=True)
    loss_fn(logits.softmax(dim=1), labels[:2], groups[:2], torch.arange(2)).backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    assert not loss_fn.score_buffer.requires_grad
