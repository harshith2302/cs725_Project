"""FGSM and PGD, operating in raw [0,1] pixel space.

The threat model is Madry et al. (2018) for MNIST: L-inf, eps=0.3, alpha=0.01,
40 steps, random start inside the eps-ball, clamped to [0,1] after every step.

These attacks assume the model consumes raw pixels -- normalisation lives inside
the network (``src/models.py:InputNormalize``). ``assert_pixel_space_model``
checks that assumption rather than trusting it, because the failure is silent:
an attack against a model with normalisation in the dataloader runs at an
effective eps of 0.3*0.3081 ~ 0.09 and reports robustness that does not exist.
"""

import torch
import torch.nn as nn

EPS_DEFAULT = 0.3
ALPHA_DEFAULT = 0.01
STEPS_DEFAULT = 40


def assert_pixel_space_model(model) -> None:
    """Fail loudly if the model does not own its normalisation."""
    if not any(m.__class__.__name__ == "InputNormalize" for m in model.modules()):
        raise ValueError(
            "Model has no InputNormalize layer. Attacks operate on raw [0,1] pixels; "
            "a model expecting pre-normalised input would be attacked at the wrong "
            "epsilon and report meaningless robust accuracy."
        )


def _grad_wrt_input(model, x, y, loss_fn):
    x = x.clone().detach().requires_grad_(True)
    loss = loss_fn(model(x), y)
    (grad,) = torch.autograd.grad(loss, x)
    return grad


def fgsm(model, x, y, eps: float = EPS_DEFAULT, loss_fn=None):
    """Single-step sign-of-gradient attack. Cheap sanity check on PGD."""
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    was_training = model.training
    model.eval()
    grad = _grad_wrt_input(model, x, y, loss_fn)
    adv = (x + eps * grad.sign()).clamp(0.0, 1.0)
    model.train(was_training)
    return adv.detach()


def pgd(
    model,
    x,
    y,
    eps: float = EPS_DEFAULT,
    alpha: float = ALPHA_DEFAULT,
    steps: int = STEPS_DEFAULT,
    random_start: bool = True,
    loss_fn=None,
):
    """PGD-L-inf. Returns adversarial examples in [0,1]."""
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    was_training = model.training
    model.eval()

    x_orig = x.clone().detach()
    if random_start:
        adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
        adv = adv.clamp(0.0, 1.0)
    else:
        adv = x_orig.clone()

    for _ in range(steps):
        grad = _grad_wrt_input(model, adv, y, loss_fn)
        adv = adv.detach() + alpha * grad.sign()
        # Project back into the eps-ball, then into the valid pixel range.
        adv = x_orig + (adv - x_orig).clamp(-eps, eps)
        adv = adv.clamp(0.0, 1.0)

    model.train(was_training)
    return adv.detach()


ATTACKS = {"fgsm": fgsm, "pgd": pgd}


@torch.enable_grad()
def adversarial_accuracy(
    model,
    loader,
    device,
    attack: str = "pgd",
    eps: float = EPS_DEFAULT,
    max_batches: int = None,
    **kwargs,
) -> dict:
    """Accuracy under attack, plus the perturbation-budget audit.

    ``max_linf`` must not exceed eps and inputs must stay in [0,1]; both are
    measured rather than assumed, so a projection bug shows up as a number in
    the JSON record instead of as inflated robustness.
    """
    assert_pixel_space_model(model)
    attack_fn = ATTACKS[attack]
    model.eval()

    correct = clean_correct = n = 0
    max_linf = 0.0
    lo, hi = 1.0, 0.0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        adv = attack_fn(model, x, y, eps=eps, **kwargs)

        max_linf = max(max_linf, (adv - x).abs().max().item())
        lo, hi = min(lo, adv.min().item()), max(hi, adv.max().item())

        with torch.no_grad():
            correct += (model(adv).argmax(1) == y).sum().item()
            clean_correct += (model(x).argmax(1) == y).sum().item()
        n += y.size(0)

    # Tolerance covers float32 rounding in the projection, nothing more.
    if max_linf > eps + 1e-5:
        raise AssertionError(f"perturbation exceeded budget: {max_linf:.6f} > {eps}")
    if lo < -1e-6 or hi > 1 + 1e-6:
        raise AssertionError(f"adversarial inputs left [0,1]: [{lo:.6f}, {hi:.6f}]")

    return {
        "attack": attack,
        "eps": eps,
        "robust_acc": 100.0 * correct / n,
        "clean_acc": 100.0 * clean_correct / n,
        "n_images": n,
        "max_linf": max_linf,
        "adv_range": [lo, hi],
        **{k: v for k, v in kwargs.items() if isinstance(v, (int, float, bool))},
    }


def epsilon_sweep(model, loader, device, epsilons=(0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4), **kwargs):
    """Robustness curve for the dense-vs-pruned figure."""
    return [adversarial_accuracy(model, loader, device, eps=e, **kwargs) for e in epsilons]
