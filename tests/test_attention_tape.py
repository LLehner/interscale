"""The frozen-attention control and closed-form checks of the pairwise interaction (weekend_plan.md A1 / A4, 2026-11-28).

* :class:`AttentionTape` reaches every softmax the toy models use, the vendored dense attention included (its ``softmax`` is ``F.softmax``,
  which dispatches to the patched ``Tensor.softmax``: guarded here, so a torch change cannot silently skip it), replays exactly what it
  recorded, refuses a changed step, and puts every patched name back.
* With the attention held, the read-out model (a weighted sum of per-sender values and a linear decoder) is additive in its senders to float
  precision: the pairwise interaction vanishes, while unfrozen it does not (the softmax re-weighting is the read-out's whole interaction).
* Closed form: the same read-out with a known non-additive head ``f(c) = (c . u1)(c . u2)`` on its context. With the attention held the
  context moves by ``delta_j`` per removed sender, so ``I_jk = -((delta_j . u1)(delta_k . u2) + (delta_k . u1)(delta_j . u2))``, and
  ``delta_j . u`` is exactly what a linear head ``u`` reads as ``D_j``. The instrument must reproduce it; the linear heads must read 0.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_spec = importlib.util.spec_from_file_location("attr_methods_base", Path(__file__).with_name("test_attribution_methods.py"))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

from interscale.evaluation import attribution as at  # noqa: E402


@pytest.fixture(scope="module")
def ir(tmp_path_factory):
    return base._model(tmp_path_factory.mktemp("tape_ir"), "ir")


@pytest.fixture(scope="module")
def stream(tmp_path_factory):
    return base._model(tmp_path_factory.mktemp("tape_stream"), "stream")


def _names():
    return [(obj, name, getattr(obj, name)) for obj, name in at._attention_patch_points()]


def _read(module, graph, receivers, w, tokens):
    with torch.no_grad():
        return at._Reader(module, graph, receivers, w, "global")(tokens.clone()).cpu().numpy()


def test_the_tape_reaches_the_vendored_dense_path_replays_exactly_and_restores_every_name(stream, ir):
    from interscale.module.global_modules import transformer_utils as tu

    before = _names()
    for module, graph, labels in (stream, ir):
        receivers = np.array([4, 17])
        w = at.gene_targets([0, 1], 2, base.G)
        tokens = at._actual_tokens(module, graph)
        plain = _read(module, graph, receivers, w, tokens)
        tape = at.AttentionTape()
        with tape.record():
            recorded = _read(module, graph, receivers, w, tokens)
        assert tape.calls >= module.num_layers if hasattr(module, "num_layers") else tape.calls >= 1
        np.testing.assert_allclose(recorded, plain, rtol=1e-6, atol=1e-7)  # recording changes nothing
        with tape.replay():
            replayed = _read(module, graph, receivers, w, tokens)
        np.testing.assert_allclose(replayed, plain, rtol=1e-6, atol=1e-7)  # the same input: the same weights, the same output
        # A changed input: unfrozen the weights move, frozen they do not, so the two reads differ.
        moved = tokens.clone()
        rows = torch.arange(0, graph.num_nodes, 3)
        moved[rows] += torch.randn(moved[rows].shape, generator=torch.Generator().manual_seed(0))  # not a constant: LayerNorm takes that out
        free = _read(module, graph, receivers, w, moved)
        with tape.replay():
            held = _read(module, graph, receivers, w, moved)
        assert np.abs(free - held).max() > 1e-6
        # Replay must use every recorded call, in order, with matching shapes.
        tape.weights.append(tape.weights[-1])
        with pytest.raises(ValueError, match="recorded"):
            with tape.replay():
                _read(module, graph, receivers, w, tokens)
        tape.weights = tape.weights[:-2]
        with pytest.raises(ValueError, match="no recorded weights"):
            with tape.replay():
                _read(module, graph, receivers, w, tokens)
    # the vendored path was reached (stream: MultiHeadAttentionWithEdits), and every name is back
    assert type(stream[0].transformer_encoder.layers[0].self_attn).__name__ == "MultiHeadAttentionWithEdits"
    assert [(o, n, f) for o, n, f in before] == _names()
    assert tu.softmax is torch.nn.functional.softmax and tu.scaled_dot_product_attention is torch.nn.functional.scaled_dot_product_attention


def test_the_tape_sees_the_vendored_softmax_calls(stream, monkeypatch):
    """A mutant: without the ``Tensor`` methods the vendored model's softmax is not reached, records nothing, and the tape refuses it."""
    module, graph, labels = stream
    points = at._attention_patch_points
    monkeypatch.setattr(at, "_attention_patch_points", lambda: [(o, n) for o, n in points() if o is not torch.Tensor])
    w = at.gene_targets([0], 1, base.G)
    tokens = at._actual_tokens(module, graph)
    with pytest.raises(ValueError, match="recorded no attention weights"):
        with at.AttentionTape().record():
            _read(module, graph, np.array([4]), w, tokens)


def test_every_softmax_name_is_reached_exactly_once():
    from interscale.module.global_modules import transformer_utils as tu

    seen = []
    x = torch.randn(3, 5)
    with at._patched_attention(lambda w: (seen.append(1), w)[1]):
        for fn in (torch.softmax, torch.nn.functional.softmax, tu.softmax, lambda t, dim: t.softmax(dim), torch.log_softmax,
                   torch.nn.functional.log_softmax):  # fmt: skip
            before = len(seen)
            fn(x, dim=-1)
            assert len(seen) == before + 1, fn
        before = len(seen)
        q = torch.randn(1, 2, 4, 3)
        out = tu.scaled_dot_product_attention(q, q, q)
        assert len(seen) == before + 1
    torch.testing.assert_close(out, torch.nn.functional.scaled_dot_product_attention(q, q, q))


def test_frozen_readout_is_additive_in_its_senders_and_unfrozen_is_not(ir):
    module, graph, labels = ir
    receivers = np.array([5, 20, 40])
    w = at.gene_targets([0, 2], 3, base.G)
    cand = at.near_candidates(graph.pos.numpy(), receivers, np.ones(graph.num_nodes, bool), k=8)
    free = at.pairwise_interactions(module, graph.clone(), receivers, w, cand, pairs=4, seed=1)
    held = at.pairwise_interactions(module, graph.clone(), receivers, w, cand, pairs=4, seed=1, freeze_attention=True)
    assert np.nanmax(np.abs(held["interaction"])) < 1e-5 * max(np.nanmax(held["single"]), 1e-12) + 1e-6
    assert np.nanmax(np.abs(free["interaction"])) > 1e2 * np.nanmax(np.abs(held["interaction"]))
    assert np.nanmedian(held["single"]) > 0  # the senders still act: only their interaction is gone
    with pytest.raises(ValueError, match="freeze_attention"):
        at.pairwise_interactions(module, graph.clone(), receivers, w, cand, pairs=1, removal="mask", freeze_attention=True)


class _KnownHead(torch.nn.Module):
    """Columns: ``c . u1``, ``c . u2``, ``(c . u1)(c . u2)``, then zeros."""

    def __init__(self, n_embed: int, n_output: int, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.u = torch.nn.Parameter(torch.randn(2, n_embed, generator=g), requires_grad=False)
        self.n_output = n_output

    def forward(self, c):
        a, b = c @ self.u[0], c @ self.u[1]
        out = c.new_zeros(c.shape[0], self.n_output)
        out[:, 0], out[:, 1], out[:, 2] = a, b, a * b
        return out


def test_closed_form_interaction_of_a_known_head(ir, monkeypatch):
    module, graph, labels = ir
    monkeypatch.setattr(module, "decoder", _KnownHead(module.decoder.decoder.in_features, base.G))
    receivers = np.array([5, 20, 40])
    cand = at.near_candidates(graph.pos.numpy(), receivers, np.ones(graph.num_nodes, bool), k=8)
    given = np.stack([cand[:, :2], cand[:, 2:4]], axis=1)  # [Q, 2 pairs, 2]
    run = {g: at.pairwise_interactions(module, graph.clone(), receivers, at.gene_targets([g], 3, base.G), None, given_pairs=given,
                                       freeze_attention=True) for g in (0, 1, 2)}  # fmt: skip
    for g in (0, 1):  # linear heads: additive, and D_j is delta_j . u exactly
        assert np.nanmax(np.abs(run[g]["interaction"])) < 1e-5
    a_j, a_k, b_j, b_k = run[0]["d_j"], run[0]["d_k"], run[1]["d_j"], run[1]["d_k"]
    expected = -(a_j * b_k + a_k * b_j)
    np.testing.assert_allclose(run[2]["interaction"], expected, rtol=1e-3, atol=1e-6)
    assert np.nanmax(np.abs(expected)) > 1e-4  # not vacuous
