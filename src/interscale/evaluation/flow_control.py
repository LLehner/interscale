"""Controls for the net-attention-flow readout: does flow follow signalling, or co-occurrence?

The probes in :mod:`~interscale.evaluation.online_probes` measure what is *in the embedding*.
This measures what the *attention* says. They can disagree: a contrastive term can raise every
probe while making the flow map less interpretable, because nothing in a probe score constrains
which cells the transformer attends to.

The control is a pair of cell types that **co-occur spatially but do not interact**. Attention
that merely tracks who-is-near-whom shows flow on such a pair; attention that tracks signalling
does not. Without one, a strong flow map is unfalsifiable -- every method produces a picture, and
co-occurrence alone is enough to produce a plausible one.

Nothing here knows any cell type, niche or dataset. Pairs are supplied by the caller (or from
``cfg.probe.flow_null_pairs`` / ``flow_signal_pairs``) as ``"sender>receiver"`` strings, and the
flow matrix is a plain DataFrame, so this works on any net-flow computation and any annotation.

Orientation
-----------
``flow.loc[sender, receiver]`` is the net flow **from** sender **to** receiver, positive in that
direction -- where "flow" means *information*, which
:func:`~interscale.evaluation.net_streams.compute_hierarchical_net_flow` treats as the **opposite
of attention**. Concretely: if type A's cells attend to type B's, that function reports
``flow.loc["B", "A"] = +1``, i.e. B is the information sender. So a configured pair
``"senderA>receiverA"`` claims *information flows senderA to receiverA*, and the corresponding
attention runs receiverA -> senderA. Getting this backwards inverts every interpretation without
changing a single magnitude, which is why ``test_flow_control.py`` pins it against the real
function rather than restating it here.

Net flow is antisymmetric, so ``flow.loc[b, a] == -flow.loc[a, b]``; the two kinds of pair are
therefore read differently, and deliberately so:

* a **null** pair is violated by flow in *either* direction, so it is scored on ``abs``;
* a **signal** pair is satisfied only by flow in the *configured* direction, so it is scored
  signed, and ``"a>b"`` and ``"b>a"`` are different claims.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PAIR_SEPARATOR = ">"


def parse_pairs(pairs: list[str]) -> list[tuple[str, str]]:
    """``["senderB>receiverA"]`` -> ``[("senderB", "receiverA")]``.

    Strings rather than tuples because these come from YAML through yacs, which takes a list of
    strings cleanly and a list of pairs badly.

    Raises
    ------
    ValueError
        If an entry does not contain exactly one separator, naming the entry.
    """
    parsed = []
    for pair in pairs:
        parts = [p.strip() for p in pair.split(PAIR_SEPARATOR)]
        if len(parts) != 2 or not all(parts):
            raise ValueError(
                f"Flow pair {pair!r} is not of the form 'sender{PAIR_SEPARATOR}receiver'."
            )
        parsed.append((parts[0], parts[1]))
    return parsed


def _lookup(flow: pd.DataFrame, pairs: list[tuple[str, str]], kind: str) -> np.ndarray:
    """Values of ``flow`` at each pair, with a failure that names what is actually available."""
    missing = sorted({name for pair in pairs for name in pair if name not in flow.index})
    if missing:
        raise KeyError(
            f"{kind} flow pairs name cell types {missing} that are not in the flow matrix. "
            f"Available: {sorted(flow.index)}. These come from config, so this is usually a "
            "pair list written for a different dataset."
        )
    return np.array([flow.loc[sender, receiver] for sender, receiver in pairs], dtype=float)


def _offdiagonal(flow: pd.DataFrame) -> np.ndarray:
    """Every ordered pair of distinct types, as a flat array of flows."""
    values = flow.to_numpy(dtype=float)
    return values[~np.eye(len(flow), dtype=bool)]


def net_flow_control(
    flow: pd.DataFrame,
    null_pairs: list[str],
    signal_pairs: list[str] | None = None,
) -> dict[str, float]:
    """Score a net-flow matrix against pairs that should and should not carry flow.

    Parameters
    ----------
    flow
        Square DataFrame of net flows, senders on the index and receivers on the columns. See
        the module docstring for the orientation this assumes.
    null_pairs
        ``"sender>receiver"`` strings for pairs expected to carry **no** flow -- types that
        co-occur without interacting. Scored on absolute value: flow either way is a violation.
    signal_pairs
        Optional ``"sender>receiver"`` strings for pairs expected to carry flow **in that
        direction**. Scored signed.

    Returns
    -------
    dict
        ``null_flow``
            Mean |net flow| over the null pairs. Lower is better; this is the control.
        ``null_percentile``
            Where ``null_flow`` sits in the distribution of |flow| over all off-diagonal pairs,
            in [0, 1], tied values taking their midrank. **The scale-free summary, and the one
            that needs no signal pairs**: a null pair should sit low among all pairs, whatever
            the absolute magnitudes are, and those magnitudes are not comparable across runs,
            datasets or normalisations. Read it against 0.5, not against 0.
        ``signal_flow``, ``separation``
            Present only when ``signal_pairs`` is given. Mean *signed* flow over them, and
            ``signal_flow - null_flow`` -- the quantity the Stage 2 gate is about: genuine pairs
            strengthening while the null pair stays flat.
        ``n_null``, ``n_signal``
            How many pairs each number is averaged over, so a summary is never read without
            knowing it came from one pair.

    Raises
    ------
    ValueError
        If ``null_pairs`` is empty; a control with nothing to control for would silently report
        a perfect ``null_flow`` of zero.
    KeyError
        If a pair names a type absent from ``flow``.
    """
    if not null_pairs:
        raise ValueError(
            "net_flow_control needs at least one null pair; with none, null_flow is 0.0 by "
            "construction and reads as a passing control. Set cfg.probe.flow_null_pairs."
        )
    if flow.shape[0] != flow.shape[1] or list(flow.index) != list(flow.columns):
        raise ValueError("flow must be square with identical index and columns (senders x receivers).")

    null_values = np.abs(_lookup(flow, parse_pairs(null_pairs), "null"))
    null_flow = float(null_values.mean())

    # Percentile rank with a MIDRANK for ties, not `<=`. A flow matrix is often sparse, and a
    # perfectly clean null pair sits at 0 alongside every other unconnected pair -- under `<=`
    # that puts it at the TOP of the tie group and reports ~0.9, i.e. the control reads as
    # violated precisely when it passes. The midrank places it in the middle of its ties, so a
    # clean null pair lands below 0.5 and a violated one above its own magnitude's rank.
    all_magnitudes = np.abs(_offdiagonal(flow))
    if all_magnitudes.size:
        below = float((all_magnitudes < null_flow).mean())
        at_or_below = float((all_magnitudes <= null_flow).mean())
        percentile = 0.5 * (below + at_or_below)
    else:
        percentile = float("nan")

    result = {
        "null_flow": null_flow,
        "null_percentile": percentile,
        "n_null": float(len(null_values)),
    }

    if signal_pairs:
        signal_values = _lookup(flow, parse_pairs(signal_pairs), "signal")
        result["signal_flow"] = float(signal_values.mean())
        result["separation"] = result["signal_flow"] - null_flow
        result["n_signal"] = float(len(signal_values))

    return result


def net_flow_control_report(
    flows: dict[str, pd.DataFrame | dict[str, pd.DataFrame]],
    null_pairs: list[str],
    signal_pairs: list[str] | None = None,
) -> pd.DataFrame:
    """Apply :func:`net_flow_control` to each group of a flow dict, as one table.

    Accepts what :func:`~interscale.evaluation.net_streams.compute_hierarchical_net_flow` returns
    -- ``{group: {"mean": df, "std": df}}`` -- as well as a plain ``{group: df}``, so the control
    does not depend on which of those a caller has.

    Returns
    -------
    pandas.DataFrame
        One row per group, columns as in :func:`net_flow_control`.
    """
    rows = {}
    for group, value in flows.items():
        frame = value["mean"] if isinstance(value, dict) else value
        rows[group] = net_flow_control(frame, null_pairs, signal_pairs)
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("group")


def net_flow_control_from_adata(
    adata,
    cfg,
    *,
    window_key: str | None = None,
    compute_net: bool = True,
) -> pd.DataFrame:
    """Run the control on a model-output ``adata``, taking every key from ``cfg``.

    The one-call entry point: it resolves the annotation columns from ``cfg.dataset`` and the
    pairs from ``cfg.probe``, computes the net flow, and scores it. Nothing dataset-specific is
    assumed -- which column carries the cell type, which the sample, which the condition, and
    which pairs are null all come from config.

    Parameters
    ----------
    adata
        The object produced by ``get_model_output`` / ``save_evaluation_results``, carrying the
        per-window attention in ``obsm["_attn_matrix"]``.
    cfg
        Reads ``dataset.celltype_key``, ``dataset.sample_key``, ``dataset.condition_key``,
        ``probe.flow_null_pairs`` and ``probe.flow_signal_pairs``.
    window_key
        Column identifying the attention windows. Defaults to the sample key, which is correct
        whenever one graph is one sample -- the usual layout here. Pass it explicitly for a
        sliding-window run, where several windows share a sample.
    compute_net
        Passed through: net flow (sender minus receiver) rather than raw aggregated attention.

    Returns
    -------
    pandas.DataFrame
        One row per condition (or one ``all_samples`` row when no condition key is set), with the
        columns of :func:`net_flow_control`.

    Raises
    ------
    ValueError
        If ``dataset.celltype_key`` is unset, since the flow matrix is indexed by cell type and
        there is nothing sensible to fall back to.
    """
    from interscale.evaluation.net_streams import compute_hierarchical_net_flow

    celltype_key = cfg.dataset.celltype_key
    if celltype_key is None:
        raise ValueError(
            "net_flow_control_from_adata needs dataset.celltype_key: the flow matrix is indexed "
            "by cell type, and the control's pairs name values of that column."
        )

    sample_key = cfg.dataset.sample_key[0] if cfg.dataset.sample_key else None
    if sample_key is None:
        raise ValueError("net_flow_control_from_adata needs dataset.sample_key to group windows.")

    flows = compute_hierarchical_net_flow(
        adata,
        window_key=window_key if window_key is not None else sample_key,
        sample_key=sample_key,
        condition_key=cfg.dataset.condition_key,
        cell_type_col=celltype_key,
        compute_net=compute_net,
    )
    return net_flow_control_report(
        flows,
        null_pairs=list(cfg.probe.flow_null_pairs),
        signal_pairs=list(cfg.probe.flow_signal_pairs) or None,
    )
