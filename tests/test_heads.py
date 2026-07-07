import torch

from models.heads import TwoHotSymlogRewardHead, symexp, symlog


def test_symlog_symexp_roundtrip():
    x = torch.tensor([-100.0, -1.0, -0.1, 0.0, 0.1, 1.0, 100.0])
    assert torch.allclose(symexp(symlog(x)), x, atol=1e-4)


def test_two_hot_encoding_weights_and_expectation():
    head = TwoHotSymlogRewardHead(feat_dim=8, num_bins=11, low=-5.0, high=5.0)
    target = torch.tensor([0.0, -1.0, 1.0])
    two_hot = head._two_hot(target)
    assert two_hot.shape == (3, 11)
    # Exactly two adjacent bins active (or one, when the target hits a bin),
    # weights sum to 1, and the expectation reproduces symlog(target).
    assert torch.allclose(two_hot.sum(-1), torch.ones(3))
    assert (two_hot > 0).sum(-1).max() <= 2
    recovered = symexp((two_hot * head.bins).sum(-1))
    assert torch.allclose(recovered, target, atol=1e-5)


def test_two_hot_clamps_out_of_range():
    head = TwoHotSymlogRewardHead(feat_dim=8, num_bins=5, low=-1.0, high=1.0)
    two_hot = head._two_hot(torch.tensor([1e6, -1e6]))
    assert torch.allclose(two_hot.sum(-1), torch.ones(2))
    assert two_hot[0].argmax() == 4  # clamped to the top bin
    assert two_hot[1].argmax() == 0


def test_zero_init_predicts_zero_reward():
    head = TwoHotSymlogRewardHead(feat_dim=8, num_bins=255)
    pred = head.prediction(torch.randn(4, 8))
    assert torch.allclose(pred, torch.zeros(4), atol=1e-5)


def test_loss_decreases_toward_correct_bin():
    """CE against the two-hot target must be lower for a matching prediction."""
    head = TwoHotSymlogRewardHead(feat_dim=4, num_bins=11, low=-5.0, high=5.0)
    feat = torch.randn(16, 4)
    target = torch.full((16,), -1.0)
    with torch.no_grad():
        base = head.loss(feat, target).mean()
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    for _ in range(50):
        loss = head.loss(feat, target).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < base.item()
    assert torch.allclose(head.prediction(feat), target, atol=0.2)
