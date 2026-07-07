import torch

from train.lambda_returns import discount_weights, lambda_returns


def test_lambda_returns_hand_computed_h3():
    """H=3, one batch row, checked against a manual backward recursion."""
    reward = torch.tensor([[1.0, 1.0, 1.0]])
    discount = torch.tensor([[0.9, 0.9, 0.9]])
    value = torch.tensor([[2.0, 3.0, 4.0]])  # V(s_1), V(s_2), V(s_3)
    lam = 0.95

    r2 = 1.0 + 0.9 * 4.0  # boundary: r_3 + gamma_3 * v_3
    r1 = 1.0 + 0.9 * ((1 - lam) * 3.0 + lam * r2)
    r0 = 1.0 + 0.9 * ((1 - lam) * 2.0 + lam * r1)

    out = lambda_returns(reward, discount, value, lam=lam)
    assert out.shape == (1, 3)
    assert torch.allclose(out, torch.tensor([[r0, r1, r2]]), atol=1e-6)


def test_lambda_zero_is_one_step_td():
    reward = torch.rand(4, 5)
    discount = torch.full((4, 5), 0.99)
    value = torch.rand(4, 5)
    out = lambda_returns(reward, discount, value, lam=0.0)
    expected = reward + discount * value  # R_t = r_{t+1} + gamma * V(s_{t+1})
    assert torch.allclose(out, expected, atol=1e-6)


def test_lambda_one_is_monte_carlo():
    reward = torch.rand(2, 4)
    discount = torch.full((2, 4), 0.9)
    value = torch.rand(2, 4)
    out = lambda_returns(reward, discount, value, lam=1.0)
    # Discounted sum of rewards plus discounted bootstrap at the end.
    expected = torch.zeros(2)
    for t in reversed(range(4)):
        boot = value[:, -1] if t == 3 else expected
        expected = reward[:, t] + discount[:, t] * boot
        if t == 0:
            break
    # Recompute fully for column 0.
    mc = reward[:, 0] + 0.9 * (reward[:, 1] + 0.9 * (reward[:, 2] + 0.9 * (
        reward[:, 3] + 0.9 * value[:, -1])))
    assert torch.allclose(out[:, 0], mc, atol=1e-5)


def test_zero_discount_cuts_the_future():
    """A predicted episode end (cont ~ 0) must block all later reward."""
    reward = torch.tensor([[0.0, 0.0, 100.0]])
    discount = torch.tensor([[0.9, 0.0, 0.9]])  # imagined death after step 1
    value = torch.zeros(1, 3)
    out = lambda_returns(reward, discount, value, lam=0.95)
    assert out[0, 0].abs() < 1e-6  # the +100 at t=2 never reaches t=0


def test_discount_weights_cumulative_product():
    discount = torch.tensor([[0.9, 0.8, 0.7]])
    w = discount_weights(discount)
    assert torch.allclose(w, torch.tensor([[1.0, 0.9, 0.72]]), atol=1e-6)
    assert not w.requires_grad
