"""The sender-sparsity term (fix 2, ``.claude/comparison.md``): scale-free, lowest when one gene enters the tokens, refused without a learned embedding."""

import pytest

torch = pytest.importorskip("torch")

from yacs.config import CfgNode as CN

from interscale.module.base._step_output import StepOutput, ViewOutput
from interscale.train.aux_losses import AUX_LOSSES, SenderSparsity


def _cfg(embedding="Linear"):
    cfg = CN()
    cfg.model = CN()
    cfg.model.global_component = CN()
    cfg.model.global_component.parameters = CN()
    cfg.model.global_component.parameters.type_gex_embedding = embedding
    return cfg


class _Owner(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.input_embedding = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        with torch.no_grad():
            self.input_embedding.weight.copy_(weight)


def test_bounds_scale_invariance_and_concentration():
    gen = torch.Generator().manual_seed(0)
    spread = torch.randn(8, 50, generator=gen)
    uniform = torch.ones(8, 50)
    one = torch.zeros(8, 50)
    one[:, 3] = 1.0
    assert SenderSparsity.value(uniform).item() == pytest.approx(1.0)
    assert SenderSparsity.value(one).item() == pytest.approx(50**-0.5)
    assert SenderSparsity.value(spread * 0.01).item() == pytest.approx(SenderSparsity.value(spread * 100.0).item(), rel=1e-5)
    sparse = spread.clone()
    sparse[:, 10:] *= 0.05
    assert SenderSparsity.value(sparse) < SenderSparsity.value(spread)


def test_reads_the_owner_and_has_a_gradient_that_concentrates():
    assert "sender_sparsity" in AUX_LOSSES
    gen = torch.Generator().manual_seed(1)
    owner = _Owner(torch.randn(8, 20, generator=gen))
    term = SenderSparsity(_cfg(), owner)
    out = StepOutput(y_pred=torch.zeros(3, 20), y_true=torch.zeros(3, 20), entry_mask=None, views=[ViewOutput()])
    value = term(out, None)["sender_sparsity"]
    value.backward()
    grad = owner.input_embedding.weight.grad
    norms = owner.input_embedding.weight.detach().norm(dim=0)
    big, small = int(norms.argmax()), int(norms.argmin())
    # a step down the gradient grows the largest column relative to the smallest
    w = owner.input_embedding.weight.detach() - 0.5 * grad
    assert (w[:, big].norm() / w[:, small].norm()) > (norms[big] / norms[small])


def test_refused_without_a_learned_embedding():
    with pytest.raises(ValueError, match="type_gex_embedding Linear"):
        SenderSparsity(_cfg(None))
