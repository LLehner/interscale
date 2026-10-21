"""One results file per sweep: every analysis of every arm as a section, easy to read and to re-read.

The file is Markdown::

    # <title>
    <legend>
    # Summary                      <- rebuilt by write_report, cross-arm tables as text
    ### <what the table shows>
    ```text
    ...
    ```
    # Sections
    ## <arm> | <analysis>
    <one line on how to read it>
    ```csv index=<k>
    <the table; its first k columns are the index>
    ```

:func:`append_sections` adds an arm's tables as they are produced -- from any job, in any order,
with any subset of analyses -- so a sweep that is missing a result simply has no such section.
:func:`read_sections` reads every section back; when an arm and analysis appear twice (an arm
evaluated again), the later one wins. :func:`write_report` rewrites the file: the summary over the
arms present, then every section once. Appends are single writes on an append-mode descriptor;
a rewrite racing an append from another process could drop that append, so rebuild the report when
no job is writing.

Runs of one arm at several seeds are named ``<arm>_s<seed>`` (``sweep_common.py``): the summary then
shows each such arm once, as the mean over its seeds, with the standard deviation and the value per
run in tables of their own (:func:`average_over_seeds`); the sections keep every run.

:func:`compact_tables` turns the evaluation's tables (``masked_probing``, ``attention_diagnostics``,
``receptive_field``, ``representations``) into the compact wide tables the sections hold: one row
per depth bin, feature set, layer or head, one column per depth group or distance bin. Tables it
does not know pass through unchanged. What each analysis answers: ``.claude/linear_probing_docs.md``.
"""

from __future__ import annotations

import io
import os
import re
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

#: How the sections header the arm and the analysis.
SECTION = re.compile(r"^## (?P<arm>.+) \| (?P<analysis>[\w.-]+)\s*$")
#: The opening fence of a section's table, with the number of index columns.
FENCE = re.compile(r"^```csv(?: index=(?P<index>\d+))?\s*$")
#: Floats in the sections: enough digits to compare arms, short enough to read.
FLOAT_FORMAT = "%.4g"
#: Count columns of the head scores, kept in the compact tables and shown once in the summary.
COUNTS = {"n_cells": "hidden spots", "n_hidden": "hidden entries"}
#: The model's own heads among the head-score columns; every other column is a reference.
REFERENCE_PREFIX = "ref_"
#: The feature set of model-free averages of the visible cells (``masked_probing.SMOOTHER``).
SMOOTHER = "smoother"
#: The feature set of a cell's place on its slide (``masked_probing.POSITION``).
POSITION = "position"
#: The model-free probe feature sets, shown once as references in the summary.
PROBE_REFERENCES = (SMOOTHER, POSITION, f"{SMOOTHER}+{POSITION}", "majority")
#: The flow summary's metrics, in reading order (``attention_flow.flow_summary``).
FLOW_METRICS = (
    "reach_um",
    "reach_null_um",
    "self_share",
    "self_share_null",
    "kl_null",
    "kl_null_floor",
    "shift_um",
    "shift_floor_um",
    "above_null",
    "n_sources",
)
#: Attention metrics kept per head as well as for the mean over heads.
PER_HEAD_METRICS = ("visible_lift", "span_rel", "effective_rel")

#: One line under each section's header: how to read its table.
NOTES = {
    "heads_cosine_by_depth": "centred cosine of every head and reference (ref_*) on hidden spots, by depth "
    "(µm to the nearest visible spot); higher is better, the bar is the best reference",
    "heads_sce_by_depth": "raw SCE (1 - cosine) on hidden spots, by depth; lower is better",
    "heads_by_group": "centred cosine pooled over each gene group's hidden entries",
    "heads_by_group_cells": "the same, over the hidden entries of the cells that respond (a programme's "
    "receivers) alone: the response where there is one to predict, apart from telling which cells respond",
    "heads_group_by_depth": "centred cosine pooled over each gene group's hidden entries, by depth; rows per group: "
    "the hidden entries' count, every head, every reference (ref_*)",
    "context_labels": "balanced accuracy of a label probe from hidden spots, by depth; smoother = references as "
    "features, position = the spot's place on its slide (offset, patches), smoother+position+E - "
    "smoother+position = what E knows beyond both; NaN = one class in the bin",
    "source_distances": "centred cosine of the z-scored source distances predicted from hidden spots, by depth",
    "representations": "linear CKA (1: the same up to rotation and scale) and participation ratio, by query group",
    "attention_by_query": "masked pass, per layer and head (mean: over heads), by query depth: visible_lift, "
    "span_rel, effective_rel read 1 for uniform attention",
    "attention_profile": "head-mean attention per distance bin (µm) over uniform attention's (1: uniform)",
    "attention_edges": "planted pairs with a hidden receiver: coverage, matched_rank (0.5: blind to the edge)",
    "erf_by_query": "receptive field of hidden spots: influence of the visible spots (0: none reaches the "
    "prediction), span_rel and effective_rel against uniform weights",
    "erf_profile": "influence per distance bin (µm) over uniform weights (1: uniform)",
    "erf_edges": "planted pairs: the sender's rank in the receiver's influence among visible spots as far (0.5: blind)",
    "response_edges": "programme masking: the planted sender's rank in its receiver's attribution (integrated "
    "gradients on the summed response genes) among cells as far from the receiver (0.5: blind to the sender)",
    "response_profile": "programme masking: attribution of the sampled receivers per distance bin (µm) over "
    "uniform weights (1: uniform)",
    "response_check": "programme masking: share of the response's change from the baseline cell that the "
    "contributions fail to add up to (0: they add up; over 0.1: the integration is too coarse)",
    "attention_check": "every layer's attention of a masked pass recomputed from its own inputs: reference = the "
    "definition written out, torch = torch's stock MultiheadAttention, torch_layer = torch's stock encoder layer, "
    "same weights; largest absolute difference (max_abs: the expected tensor's scale; inf: blocked entries differ)",
    "readout_edges": "the planted sender's rank among cells as far from its receiver, per way of reading attention "
    "(attention_check.READOUTS) and range class (0.5: blind to the sender; read against the untrained twin)",
    "readout_agreement": "mean Spearman correlation between two readouts' scores over a query's cells",
    "readout_span": "each readout's score-weighted distance to the query (span_um) and its share within each radius",
    "flow_summary": "where the cells of a patch draw their predictions from -- influence (gradient flow) and "
    "rollout (attention composed over layers with the residual path) -- against the slide with the visible "
    "cells' expression permuted: reach in µm, share kept in the source patch, KL from the permutations and "
    "the centre-of-mass shift (each next to the floor permutations reach among themselves), share of the "
    "flow more than 2 null SDs above the permutations",
    "flow_labels": "the same by label (query label drawing from key label): observed share, permuted share, "
    "log2 observed / permuted and z against the permutations",
    "training": "the last epoch's throughput and peak memory, the final validation metrics",
    "timing": "seconds each step of the evaluation took",
    "phases": "staged training, each phase's end: drift_<component> = relative weight change (0: frozen), "
    "and the phase's last validation losses",
}

LEGEND = """\
One section per arm and analysis, `## <arm> | <analysis>`; a later section replaces an earlier one
with the same header. `<arm>_untrained` is the arm's model at initialisation: only a gain over it is
learned. Hidden spots lie in seeded discs; depth is the distance to the nearest visible spot, and
the GCN's reach is about 290 µm (more at slide edges). Bars: the best `ref_*` reference for head
scores, `smoother+position` (else `smoother`) for probes, 1 for every metric relative to uniform
weights, 0.5 for matched ranks. `blend_<head>` averages a head with one reference: above both, the
head holds something the reference lacks. What each analysis answers:
`.claude/linear_probing_docs.md` in the InterScale fork."""


# ------------------------------------------------------------------------------- column order


def _left_edge(label) -> float | None:
    """The left edge of an interval label such as ``"(150.0, 290.0]"``; None for any other label."""
    match = re.fullmatch(r"\(\s*([-+\w.]+),\s*([-+\w.]+)\]", str(label))
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def ordered(columns) -> list:
    """``all``, ``visible``, ``hidden`` first, then interval labels from the lowest edge, then the rest."""
    columns = list(columns)
    first = [c for c in ("all", "visible", "hidden") if c in columns]
    intervals = sorted((c for c in columns if _left_edge(c) is not None), key=_left_edge)
    rest = [c for c in columns if c not in first and c not in intervals]
    return first + intervals + rest


def deepest(labels) -> str | None:
    """The interval label with the highest left edge among ``labels``: the deepest depth bin."""
    intervals = [label for label in set(labels) if _left_edge(label) is not None]
    return max(intervals, key=_left_edge) if intervals else None


# ------------------------------------------------------------------------------- compact tables


def _head_columns(table: pd.DataFrame, suffix: str) -> dict[str, str]:
    """``{column: head}`` for the columns ``<head><suffix>``: the count first, model heads, then references.

    The ``global - local`` differences are left out; they follow from the two heads.
    """
    found = {c: c[: -len(suffix)] for c in table.columns if c.endswith(suffix) and " - " not in c}
    heads = [c for c in found if not found[c].startswith(REFERENCE_PREFIX)]
    counts = {c: c for c in table.columns if c in COUNTS}
    return counts | {c: found[c] for c in heads + [c for c in found if c not in heads]}


def _heads_by_depth(cells: pd.DataFrame | None, by_depth: pd.DataFrame | None, suffix: str) -> pd.DataFrame | None:
    parts = []
    if cells is not None and len(cells):
        parts.append(cells.assign(depth_um="all"))
    if by_depth is not None and len(by_depth):
        parts.append(by_depth)
    if not parts:
        return None
    table = pd.concat(parts, ignore_index=True)
    columns = _head_columns(table, suffix)
    if not columns:
        return None
    out = table.set_index("depth_um")[list(columns)].rename(columns=columns)
    return out.loc[ordered(out.index)]


def _pivot(table: pd.DataFrame, index, columns: str, values: str = "value") -> pd.DataFrame:
    out = table.pivot_table(index=index, columns=columns, values=values, aggfunc="first", sort=False)
    out.columns.name = None
    return out[ordered(out.columns)]


def compact_tables(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """The evaluation's tables as the compact wide tables of the sections (see :data:`NOTES`).

    Known inputs: ``heads_cells``, ``heads_by_depth``, ``heads_by_group``, ``context_labels``,
    ``source_distances``, ``representations``, ``attention_by_query``, ``attention_profile``,
    ``attention_edges``, ``erf_by_query``, ``erf_profile``, ``erf_edges``, ``heads_by_group_cells``,
    ``response_edges``, ``response_profile``, ``response_check``, ``attention_check``, ``readout_edges``,
    ``readout_agreement``, ``readout_span`` and ``training``. The
    per-gene head scores are left out (``heads_by_group`` summarises them); any other table passes
    through unchanged.
    """
    tables = {name: table for name, table in tables.items() if table is not None and len(table)}
    out = {}
    for name, suffix in (("heads_cosine_by_depth", "_cosine_centred"), ("heads_sce_by_depth", "_sce")):
        table = _heads_by_depth(tables.get("heads_cells"), tables.get("heads_by_depth"), suffix)
        if table is not None:
            out[name] = table
    if "heads_by_group" in tables:
        groups = tables["heads_by_group"]
        columns = _head_columns(groups, "_cosine")
        out["heads_by_group"] = groups.set_index("group")[list(columns)].rename(columns=columns)
    if "heads_by_group_cells" in tables:
        groups = tables["heads_by_group_cells"]
        columns = _head_columns(groups, "_cosine")
        out["heads_by_group_cells"] = groups.set_index("group")[list(columns)].rename(columns=columns)
    if "heads_by_group_depth" in tables:
        groups = tables["heads_by_group_depth"]
        columns = _head_columns(groups.drop(columns="n_genes", errors="ignore"), "_cosine")
        long = groups.melt(id_vars=["group", "depth_um"], value_vars=list(columns), var_name="head")
        long["head"] = long["head"].map(columns)
        # Groups in their order, within each the count, the model heads, then the references.
        out["heads_group_by_depth"] = _pivot(long, ["group", "head"], "depth_um")
    if "context_labels" in tables:
        labels = tables["context_labels"]
        labels = labels[labels["metric"] == "balanced_accuracy"]
        labels = labels.assign(depth_um=labels["depth_um"] if "depth_um" in labels else "all")
        out["context_labels"] = _pivot(labels, ["target", "feature_set"], "depth_um")
    if "source_distances" in tables:
        distances = tables["source_distances"]
        distances = distances.assign(depth_um=distances["depth_um"] if "depth_um" in distances else "all")
        out["source_distances"] = _pivot(distances, "feature_set", "depth_um")
    if "representations" in tables:
        reps = tables["representations"]
        b = reps["b"].fillna("").astype(str)
        measure = np.where(
            reps["metric"] == "cka", "cka " + reps["a"].astype(str) + "~" + b, "pr " + reps["a"].astype(str)
        )
        out["representations"] = _pivot(reps.assign(measure=measure), "measure", "query")
    if "attention_by_query" in tables:
        attention = tables["attention_by_query"]
        attention = attention.assign(head=attention["head"].astype(str))
        keep = (attention["head"] == "mean") | attention["metric"].isin(PER_HEAD_METRICS)
        table = _pivot(attention[keep], ["layer", "head", "metric"], "query")
        out["attention_by_query"] = table.sort_index(level=["layer", "head", "metric"], sort_remaining=False)
    if "attention_profile" in tables:
        out["attention_profile"] = _pivot(tables["attention_profile"], ["layer", "query"], "distance_um", "lift")
    if "attention_edges" in tables:
        edges = tables["attention_edges"]
        table = _pivot(edges, ["layer", "range_class"], "metric")
        coverage = edges[edges["metric"] == "coverage"].set_index(["layer", "range_class"])["n_pairs"]
        out["attention_edges"] = table.assign(n_pairs=coverage.reindex(table.index).to_numpy())
    if "erf_by_query" in tables:
        out["erf_by_query"] = _pivot(tables["erf_by_query"], ["head", "metric"], "query")
    if "erf_profile" in tables:
        out["erf_profile"] = _pivot(tables["erf_profile"], ["head", "query"], "distance_um", "lift")
    if "erf_edges" in tables:
        out["erf_edges"] = (
            tables["erf_edges"]
            .set_index(["head", "range_class"])[["value", "n_pairs"]]
            .rename(columns={"value": "matched_rank"})
        )
    if "flow_summary" in tables:
        flow = tables["flow_summary"]
        table = flow.pivot_table(index="kind", columns="metric", values="value", aggfunc="first", sort=False)
        table.columns.name = None
        out["flow_summary"] = table[[m for m in FLOW_METRICS if m in table.columns]]
    if "flow_labels" in tables:
        labels = tables["flow_labels"].copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            labels["log2_ratio"] = np.log2(labels["observed"] / labels["null_mean"])
            labels["z"] = (labels["observed"] - labels["null_mean"]) / labels["null_std"]
        long = labels.melt(
            id_vars=["kind", "query_label", "key_label"],
            value_vars=["observed", "null_mean", "log2_ratio", "z"],
            var_name="measure",
        )
        table = _pivot(long, ["kind", "measure", "query_label"], "key_label")
        out["flow_labels"] = table.sort_index(level=["kind", "measure"], sort_remaining=False)
    if "response_edges" in tables:
        out["response_edges"] = (
            tables["response_edges"]
            .set_index(["head", "range_class", "program"])[["value", "n_pairs"]]
            .rename(columns={"value": "matched_rank"})
        )
    if "response_profile" in tables:
        out["response_profile"] = _pivot(tables["response_profile"], ["head", "range_class"], "distance_um", "lift")
    if "response_check" in tables:
        out["response_check"] = (
            tables["response_check"]
            .set_index(["head", "program"])[["value", "n_queries"]]
            .rename(columns={"value": "completeness_gap"})
        )
    if "attention_check" in tables:
        out["attention_check"] = tables["attention_check"].set_index(["layer", "check", "quantity"])
    if "readout_edges" in tables:
        out["readout_edges"] = _pivot(tables["readout_edges"], "readout", "range_class", "matched_rank")
    if "readout_agreement" in tables:
        out["readout_agreement"] = _pivot(tables["readout_agreement"], "readout_a", "readout_b", "spearman")
    if "readout_span" in tables:
        out["readout_span"] = _pivot(tables["readout_span"], "readout", "metric")
    if "training" in tables:
        out["training"] = tables["training"].set_index("metric")[["value"]]
    known = {
        "heads_cells",
        "heads_by_depth",
        "heads_per_gene",
        "heads_by_group",
        "heads_by_group_cells",
        "heads_by_group_depth",
        "context_labels",
        "source_distances",
        "representations",
        "attention_by_query",
        "attention_profile",
        "attention_edges",
        "erf_by_query",
        "erf_profile",
        "erf_edges",
        "flow_summary",
        "flow_labels",
        "response_edges",
        "response_profile",
        "response_check",
        "attention_check",
        "readout_edges",
        "readout_agreement",
        "readout_span",
        "training",
    }
    out.update({name: table for name, table in tables.items() if name not in known})
    return out


# ------------------------------------------------------------------------------- the file


def _section(arm: str, analysis: str, table: pd.DataFrame) -> str:
    """One section: header, note, and the table as CSV with its index columns counted in the fence."""
    named = table.copy()
    plain = isinstance(named.index, pd.RangeIndex) and named.index.name is None
    if not plain and all(name is None for name in named.index.names):
        levels = named.index.nlevels
        named.index = named.index.set_names(["index"] if levels == 1 else [f"index_{k}" for k in range(levels)])
    body = named.to_csv(index=not plain, float_format=FLOAT_FORMAT, lineterminator="\n")
    note = NOTES.get(analysis)
    lines = [f"## {arm} | {analysis}", ""]
    if note:
        lines += [note, ""]
    lines += [f"```csv index={0 if plain else named.index.nlevels}", body.rstrip("\n"), "```", "", ""]
    return "\n".join(lines)


def _header(title: str) -> str:
    return f"# {title}\n\n{LEGEND}\n\n# Sections\n\n"


def append_sections(path, arm: str, tables: dict[str, pd.DataFrame], *, title: str = "Evaluation report") -> None:
    """Append ``tables`` -- ``{analysis: table}`` -- to the results file at ``path`` as ``arm``'s sections.

    The file and its header are created when missing. Each table is written with its index; the
    whole append is one write on an append-mode descriptor.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(_section(arm, analysis, table) for analysis, table in tables.items() if table is not None)
    if not path.exists():
        text = _header(title) + text
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(descriptor, text.encode("utf-8"))
    finally:
        os.close(descriptor)


def read_sections(path) -> dict[tuple[str, str], pd.DataFrame]:
    """Every section of the results file, ``{(arm, analysis): table}``, in order of first appearance.

    A later section with the same arm and analysis replaces the earlier one. A missing file reads empty.
    """
    path = Path(path)
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    sections: dict[tuple[str, str], pd.DataFrame] = {}
    k = 0
    while k < len(lines):
        match = SECTION.match(lines[k])
        k += 1
        if match is None:
            continue
        while k < len(lines) and FENCE.match(lines[k]) is None and SECTION.match(lines[k]) is None:
            k += 1
        if k >= len(lines) or SECTION.match(lines[k]):
            continue
        n_index = int(FENCE.match(lines[k]).group("index") or 0)
        k += 1
        start = k
        while k < len(lines) and lines[k].strip() != "```":
            k += 1
        csv = "\n".join(lines[start:k])
        k += 1
        table = pd.read_csv(io.StringIO(csv), index_col=list(range(n_index)) if n_index else None)
        sections[(match["arm"], match["analysis"])] = table
    return sections


def read_parts(folder, sections: dict | None = None) -> dict[tuple[str, str], pd.DataFrame]:
    """``sections`` updated with the section files of ``folder``, oldest file first (so the newest wins).

    Jobs that run at the same time write one file each (``<folder>/<arm>.md``, by
    :func:`append_sections`) instead of appending to one shared file, so no write can interleave with
    another; :func:`write_report` then merges them. A missing folder changes nothing.
    """
    merged = dict(sections or {})
    folder = Path(folder)
    if folder.is_dir():
        for part in sorted(folder.glob("*.md"), key=lambda f: (f.stat().st_mtime, f.name)):
            merged.update(read_sections(part))
    return merged


def write_report(
    path,
    *,
    title: str = "Evaluation report",
    arms: list[str] | None = None,
    groups: list[str] | None = None,
    parts=None,
    bar: str | None = None,
    programmes: str = "interaction=",
) -> str:
    """Rewrite the results file: header, the summary over its arms, then every section once.

    ``arms`` orders the arms (e.g. the sweep's, each followed by its ``_untrained`` twin); arms in
    the file but not in the list follow in order of appearance. ``groups`` limits the summary's
    gene-group tables to those groups; the sections keep every group. ``parts`` is a folder of
    per-arm section files (:func:`read_parts`) merged over the file's own sections. ``bar`` and
    ``programmes`` add the tables of programmes won, tied and lost against a reference
    (:func:`summary_tables`). Returns the summary as text.
    """
    path = Path(path)
    sections = read_sections(path)
    if parts is not None:
        sections = read_parts(parts, sections)
    present = list(dict.fromkeys(arm for arm, _ in sections))
    order = [arm for arm in (arms or []) if arm in present] + [arm for arm in present if arm not in (arms or [])]
    summary = render_summary(summary_tables(sections, order, groups, bar=bar, programmes=programmes))
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    text = f"# {title}\n\n{LEGEND}\n\n# Summary\n\n_Rebuilt {stamp} from {len(order)} arms: {', '.join(order)}._\n\n"
    text += summary + "\n# Sections\n\n"
    for arm in order:
        text += "".join(_section(arm, analysis, table) for (a, analysis), table in sections.items() if a == arm)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
    return summary


# ------------------------------------------------------------------------------- seeds

#: An arm run at one seed: ``<arm>_s<seed>``.
SEED_ARM = re.compile(r"^(?P<base>.+)_s(?P<seed>\d+)$")


def seed_groups(arms: list[str]) -> dict[str, list[str]]:
    """``{base: [arm runs]}`` for the arms present at two or more seeds, in the order of ``arms``."""
    found: dict[str, list[str]] = {}
    for arm in arms:
        match = SEED_ARM.match(arm)
        if match is not None:
            found.setdefault(match["base"], []).append(arm)
    return {base: runs for base, runs in found.items() if len(runs) >= 2}


def _mean_sd(tables: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean and sample standard deviation (``ddof=1``) over tables of one analysis, entry by entry.

    Tables with the same index and columns are averaged entry by entry (text columns from the first);
    otherwise rows are matched by index and only numeric columns are kept, each averaged over the runs
    that have it.
    """
    first = tables[0]
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        if all(t.index.equals(first.index) and list(t.columns) == list(first.columns) for t in tables):
            numeric = [c for c in first.columns if all(pd.api.types.is_numeric_dtype(t[c]) for t in tables)]
            stack = np.stack([t[numeric].to_numpy(dtype=float) for t in tables])
            mean, sd = first.copy(), first.copy()
            if numeric:
                mean[numeric] = np.nanmean(stack, axis=0)
                sd[numeric] = np.nanstd(stack, axis=0, ddof=1)
            return mean, sd
        big = pd.concat(tables, keys=range(len(tables)))
        levels = list(range(1, big.index.nlevels))
        grouped = big.groupby(level=levels, sort=False)
        return grouped.mean(numeric_only=True), grouped.std(numeric_only=True, ddof=1)


def average_over_seeds(sections, arms: list[str]):
    """The sections plus, for every arm that ran at several seeds, its mean and its spread.

    Returns ``(mean, sd, shown, groups)``: ``mean`` and ``sd`` are ``{(arm, analysis): table}``; the
    groups appear in them as ``<base> [<n> seeds]``. ``shown`` lists the arms a summary reads, in the
    order of ``arms`` with each seed group once, at its first run's place (arms at one seed and every
    other arm keep their names), and ``groups`` is :func:`seed_groups`. ``sd`` holds the groups only.
    """
    groups = seed_groups(arms)
    label = {base: f"{base} [{len(runs)} seeds]" for base, runs in groups.items()}
    member = {run: base for base, runs in groups.items() for run in runs}
    mean = dict(sections)  # the runs stay (nothing looks them up: they are not in `shown`); the groups' means are added
    sd: dict = {}
    for base, runs in groups.items():
        for analysis in dict.fromkeys(a for (arm, a) in sections if arm in runs):
            tables = [sections[(run, analysis)] for run in runs if (run, analysis) in sections]
            if not tables:
                continue
            mean[(label[base], analysis)], sd[(label[base], analysis)] = _mean_sd(tables)
    shown = []
    for arm in arms:
        name = label[member[arm]] if arm in member else arm
        if name not in shown:
            shown.append(name)
    return mean, sd, shown, groups


def _without_shared(table: pd.DataFrame | None) -> pd.DataFrame | None:
    """A summary table without its count and reference rows: they hold no spread, and are shown once elsewhere."""
    if table is None:
        return None
    keep = [i for i, row in enumerate(table.index) if (row[0] if isinstance(row, tuple) else row) not in ("count", "reference")]
    return table.iloc[keep]


def _seed_tables(sections, mean, sd, groups, shown, group_filter) -> list[tuple[str, pd.DataFrame]]:
    """The spread of the key head scores over seeds, and each run's own value, beside the mean tables."""
    out = []
    spread = [a for a in shown if a in {k[0] for k in sd}]

    def add(title, table):
        if table is not None and len(table) and table.notna().any().any():
            out.append((title, table))

    n = ", ".join(f"{base}: {len(runs)}" for base, runs in groups.items())
    cosine = _per_arm(sd, "heads_cosine_by_depth", spread)
    add(f"centred cosine on hidden spots by depth: standard deviation over seeds ({n})", _without_shared(_heads_summary(cosine)))
    by_group = _per_arm(sd, "heads_by_group", spread)
    if group_filter is not None:
        by_group = {arm: t.loc[[g for g in group_filter if g in t.index]] for arm, t in by_group.items()}
    add("centred cosine per gene group: standard deviation over seeds", _without_shared(_heads_summary(by_group)))
    runs = [run for members in groups.values() for run in members]
    per_run = _heads_summary(_per_arm(sections, "heads_cosine_by_depth", runs))
    if per_run is not None and "all" in per_run.columns:
        add("centred cosine on all hidden spots, every run", _without_shared(per_run[["all"]]))
    rows = []
    for arm, table in _per_arm(sd, "response_edges", spread).items():
        overall = table.xs("any", level="program")["matched_rank"]
        overall.index = pd.MultiIndex.from_tuples(
            [(arm, head, rc) for head, rc in overall.index], names=["arm", "head", "range_class"]
        )
        rows.append(overall)
    if rows:
        add("programme masking, sender's rank: standard deviation over seeds", pd.concat(rows).unstack("range_class"))
    return out


# ------------------------------------------------------------------------------- the summary


def _per_arm(sections, analysis: str, arms: list[str]) -> dict[str, pd.DataFrame]:
    return {arm: sections[(arm, analysis)] for arm in arms if (arm, analysis) in sections}


def _rows(rows: list[pd.Series], names: list[str]) -> pd.DataFrame | None:
    if not rows:
        return None
    table = pd.DataFrame(rows)
    table.index = pd.MultiIndex.from_tuples(table.index, names=names)
    return table[ordered(table.columns)]


def _heads_summary(per_arm: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    """Rows ``(arm, head)`` for the model heads; the counts first and the references last, once each.

    Counts and references depend on the data and the discs only, so they are the same in every arm.
    """
    rows, shared, source = [], [], None
    for arm, table in per_arm.items():
        for column in table.columns:
            name = str(column)
            if name in COUNTS or name.startswith(REFERENCE_PREFIX):
                if source in (None, arm):
                    source = arm
                    label = ("count", COUNTS[name]) if name in COUNTS else ("reference", name[len(REFERENCE_PREFIX) :])
                    shared.append(table[column].rename(label))
            else:
                rows.append(table[column].rename((arm, column)))
    counts = [row for row in shared if row.name[0] == "count"]
    return _rows(counts + rows + [row for row in shared if row.name[0] != "count"], ["arm", "head"])


def _gain_summary(per_arm: dict[str, pd.DataFrame], target: str | None = None) -> pd.DataFrame | None:
    """Per arm, each model embedding and its gain over each model-free bar, by depth; the bars (and floor) once.

    The bars are ``smoother``, ``position`` and ``smoother+position`` as far as they were probed;
    the gain over a bar is ``<bar>+<embedding>`` minus ``<bar>``.
    """
    rows, reference = [], []
    for arm, table in per_arm.items():
        if target is not None:
            if target not in table.index.get_level_values(0):
                continue
            table = table.xs(target, level=0)
        if not reference:
            reference = [
                table.loc[name].rename(("reference", name)) for name in PROBE_REFERENCES if name in table.index
            ]
        for embedding in ("global", "local"):
            if embedding not in table.index:
                continue
            rows.append(table.loc[embedding].rename((arm, embedding)))
            for bar in (SMOOTHER, POSITION, f"{SMOOTHER}+{POSITION}"):
                if f"{bar}+{embedding}" in table.index and bar in table.index:
                    gain = table.loc[f"{bar}+{embedding}"] - table.loc[bar]
                    rows.append(gain.rename((arm, f"{embedding}: gain over {bar}")))
    return _rows(reference + rows, ["arm", "row"])


def _group_depth_summary(per_arm: dict[str, pd.DataFrame], groups: list[str] | None) -> pd.DataFrame | None:
    """Rows ``(group, arm, head)``: per gene group the count and the best reference once, every arm's heads.

    ``best`` is the highest reference in each depth bin (not always the same one), ``slide_mean``
    the floor without position. ``groups`` keeps those groups, in that order; None keeps all.
    """
    rows = {}
    for arm, table in per_arm.items():
        for group in table.index.get_level_values("group").unique():
            if groups is not None and group not in groups:
                continue
            part = table.xs(group, level="group")
            found = rows.setdefault(group, {"shared": [], "heads": []})
            for head, values in part.iterrows():
                name = str(head)
                if name not in COUNTS and not name.startswith(REFERENCE_PREFIX):
                    found["heads"].append(values.rename((group, arm, name)))
            if not found["shared"]:
                for count in (h for h in part.index if str(h) in COUNTS):
                    found["shared"].append(part.loc[count].rename((group, "count", COUNTS[str(count)])))
                references = part.loc[[h for h in part.index if str(h).startswith(REFERENCE_PREFIX)]]
                if len(references):
                    found["shared"].append(references.max().rename((group, "reference", "best")))
                if "ref_slide_mean" in part.index:
                    found["shared"].append(part.loc["ref_slide_mean"].rename((group, "reference", "slide_mean")))
    order = [g for g in (groups or rows) if g in rows]
    ordered_rows = []
    for group in order:
        counts = [r for r in rows[group]["shared"] if r.name[1] == "count"]
        references = [r for r in rows[group]["shared"] if r.name[1] != "count"]
        ordered_rows += counts + rows[group]["heads"] + references
    return _rows(ordered_rows, ["group", "arm", "head"])


def _select(per_arm: dict[str, pd.DataFrame], level: str, value, keep: list[str]) -> pd.DataFrame | None:
    """Rows ``(arm, <the remaining index>)`` where index level ``level`` equals ``value``."""
    parts = []
    for arm, table in per_arm.items():
        if value not in table.index.get_level_values(level):
            continue
        part = table.xs(value, level=level)
        part.index = pd.MultiIndex.from_tuples(
            [(arm, *((i,) if not isinstance(i, tuple) else i)) for i in part.index], names=["arm", *keep]
        )
        parts.append(part)
    if not parts:
        return None
    table = pd.concat(parts)
    return table[ordered(table.columns)].dropna(axis=1, how="all")


#: How far a programme's score must lie from the bar's to count as a win or a loss rather than a tie.
TIE_MARGIN = 0.05
#: ``bar="best"``: every programme is compared with the highest of the references on it.
BEST_REFERENCE = "best"


def _versus_bar(per_arm: dict[str, pd.DataFrame], bar: str, prefix: str, margin: float) -> pd.DataFrame | None:
    """Rows ``(arm, head)``: on how many programmes the head beats, ties or loses to ``bar``.

    The programmes are the groups whose name starts with ``prefix``; a win or a loss is a score more
    than ``margin`` above or below the bar's on that programme, a tie anything between. The means
    weigh every programme alike, which a pooled score does not: that one is carried by the programmes
    holding most of the hidden entries. ``best`` and ``worst`` name the programmes of the largest gain and loss.
    ``bar`` is a column of the head scores, or :data:`BEST_REFERENCE` for the highest reference on each programme
    (picked on the same scores, so a hard bar).
    """
    rows = []
    for arm, table in per_arm.items():
        programmes = [g for g in table.index if str(g).startswith(prefix)]
        references = [c for c in table.columns if str(c).startswith(REFERENCE_PREFIX)]
        if not programmes or (bar != BEST_REFERENCE and bar not in table.columns):
            continue
        if bar == BEST_REFERENCE:
            reference = table.loc[programmes, references].max(axis=1)
        else:
            reference = table.loc[programmes, bar]
        heads = [
            c
            for c in table.columns
            if c not in COUNTS and not str(c).startswith((REFERENCE_PREFIX, "blend_"))
        ]
        for head in heads:
            score = table.loc[programmes, head]
            valid = (score.notna() & reference.notna()).to_numpy()
            if not valid.any():
                continue
            gain = (score - reference)[valid].round(9)  # a difference of exactly the margin is a tie, not float noise
            best, worst = gain.idxmax(), gain.idxmin()
            rows.append(
                {
                    "arm": arm,
                    "head": head,
                    "programmes": int(valid.sum()),
                    "wins": int((gain > margin).sum()),
                    "ties": int((gain.abs() <= margin).sum()),
                    "losses": int((gain < -margin).sum()),
                    "mean": float(score[valid].mean()),
                    "bar mean": float(reference[valid].mean()),
                    "best": f"{str(best)[len(prefix):]} {gain[best]:+.2f}",
                    "worst": f"{str(worst)[len(prefix):]} {gain[worst]:+.2f}",
                }
            )
    if not rows:
        return None
    return pd.DataFrame(rows).set_index(["arm", "head"])


def summary_tables(
    sections,
    arms: list[str],
    groups: list[str] | None = None,
    *,
    bar: str | None = None,
    programmes: str = "interaction=",
    margin: float = TIE_MARGIN,
) -> list[tuple[str, pd.DataFrame]]:
    """The cross-arm tables of the summary, from the sections of ``arms``; missing analyses are skipped.

    ``groups`` limits the gene-group tables to those groups (None: every group). ``bar`` names the
    reference (a column of the head scores, e.g. ``ref_own+context``, or ``"best"`` for the highest
    reference on each programme) every programme -- a group named ``programmes<...>`` -- is compared
    with, for the tables of wins, ties and losses (:func:`_versus_bar`, ``margin`` apart); None leaves
    them out.
    """
    if seed_groups(arms):
        mean, sd, shown, found = average_over_seeds(sections, arms)
        note = ", ".join(f"{base} [{len(runs)} seeds]" for base, runs in found.items())
        options = {"bar": bar, "programmes": programmes, "margin": margin}
        tables = [
            (f"{title} (seeds averaged: {note})" if k == 0 else title, t)
            for k, (title, t) in enumerate(summary_tables(mean, shown, groups, **options))
        ]
        return tables + _seed_tables(sections, mean, sd, found, shown, groups)

    out = []

    def add(title, table):
        if table is not None and len(table) and table.notna().any().any():
            out.append((title, table))

    add(
        "centred cosine on hidden spots by depth (µm to the nearest visible spot), higher is better; the bar is the best reference",
        _heads_summary(_per_arm(sections, "heads_cosine_by_depth", arms)),
    )
    add(
        "raw SCE on hidden spots by depth, lower is better",
        _heads_summary(_per_arm(sections, "heads_sce_by_depth", arms)),
    )
    by_group = _per_arm(sections, "heads_by_group", arms)
    if groups is not None:
        by_group = {arm: table.loc[[g for g in groups if g in table.index]] for arm, table in by_group.items()}
    add("centred cosine per gene group on hidden spots", _heads_summary(by_group))
    if bar is not None:
        bar_name = "the best reference on each" if bar == BEST_REFERENCE else f"the bar {bar}"
        what = f"programmes ('{programmes}...') against {bar_name}: wins / ties / losses by more than {margin:g}"
        add(
            f"{what}; means weigh every programme alike",
            _versus_bar(_per_arm(sections, "heads_by_group", arms), bar, programmes, margin),
        )
        add(
            f"{what}, scored within their receivers alone",
            _versus_bar(_per_arm(sections, "heads_by_group_cells", arms), bar, programmes, margin),
        )
    add(
        "centred cosine per gene group on hidden spots by depth; best = the highest reference per bin",
        _group_depth_summary(_per_arm(sections, "heads_group_by_depth", arms), groups),
    )
    labels = _per_arm(sections, "context_labels", arms)
    targets = list(dict.fromkeys(t for table in labels.values() for t in table.index.get_level_values(0)))
    for target in targets:
        add(f"label '{target}' from hidden spots by depth, balanced accuracy", _gain_summary(labels, target))
    add(
        "source distances from hidden spots by depth, centred cosine",
        _gain_summary(_per_arm(sections, "source_distances", arms)),
    )

    reps = _per_arm(sections, "representations", arms)
    wanted = [
        "cka local~global",
        "cka global~smoother",
        "cka local~smoother",
        "cka global~pe",
        "pr global",
        "pr local",
        "pr smoother",
    ]
    parts = []
    for arm, table in reps.items():
        part = table.loc[[m for m in wanted if m in table.index]]
        part.index = pd.MultiIndex.from_tuples([(arm, m) for m in part.index], names=["arm", "measure"])
        parts.append(part)
    if parts:
        table = pd.concat(parts)
        add("representations by query group: linear CKA and participation ratio", table[ordered(table.columns)])

    attention = _per_arm(sections, "attention_by_query", arms)
    means = _select(attention, "head", "mean", ["layer", "metric"])
    for metric, what in (
        ("visible_lift", "attention on visible spots over their share of the keys (1: blind to what is hidden)"),
        ("span_rel", "attention span over uniform attention's (below 1: nearer)"),
        ("effective_rel", "effective keys over the keys a spot can attend to (1: uniform)"),
        ("cls_mass", "attention on CLS"),
    ):
        if means is not None and metric in means.index.get_level_values("metric"):
            add(f"masked pass, mean over heads: {what}", means.xs(metric, level="metric").dropna(axis=1, how="all"))
    for analysis, unit, what in (("attention_profile", "layer", "attention"), ("erf_profile", "head", "influence")):
        profiles = _per_arm(sections, analysis, arms)
        groups = [q for table in profiles.values() for q in table.index.get_level_values("query")]
        bin_ = deepest(groups)
        if bin_ is not None:
            add(
                f"{what} of hidden spots at depth {bin_} µm by distance (µm), over uniform weights",
                _select(profiles, "query", bin_, [unit]),
            )
    edges = _per_arm(sections, "attention_edges", arms)
    for metric, what in (
        ("matched_rank", "sender's rank among visible spots as far from the hidden receiver (0.5: blind)"),
        ("coverage", "share of planted pairs whose sender the hidden receiver can attend to"),
    ):
        rows = []
        for arm, table in edges.items():
            if metric in table.columns:
                row = table[metric]
                row.index = [f"{rc} l{layer}" for layer, rc in row.index]
                rows.append(row.rename(arm))
        if rows:
            add(f"attention, planted pairs: {what}", pd.DataFrame(rows))
    erf = _per_arm(sections, "erf_by_query", arms)
    metrics = list(dict.fromkeys(m for table in erf.values() for m in table.index.get_level_values("metric")))
    beyond = sorted(
        (m for m in metrics if str(m).startswith("beyond_")), key=lambda m: float(m[7:-2].replace("p", "."))
    )
    for metric, what in (
        ("influence", "summed influence of the visible spots (0: nothing reaches the prediction)"),
        ("span_rel", "influence-weighted distance over uniform's"),
        ("effective_rel", "effective influencing spots over the visible spots"),
        *[(m, f"share of the influence {m.replace('beyond_', 'beyond ')}") for m in beyond[:1]],
    ):
        add(f"receptive field of hidden spots by depth: {what}", _select(erf, "metric", metric, ["head"]))
    rows = []
    for arm, table in _per_arm(sections, "erf_edges", arms).items():
        row = table["matched_rank"]
        row.index = pd.MultiIndex.from_tuples(
            [(arm, head, rc) for head, rc in row.index], names=["arm", "head", "range_class"]
        )
        rows.append(row)
    if rows:
        add(
            "receptive field, planted pairs: sender's rank in the receiver's influence among visible spots as far (0.5: blind)",
            pd.concat(rows).unstack("range_class"),
        )
    rows = []
    for arm, table in _per_arm(sections, "response_edges", arms).items():
        overall = table.xs("any", level="program")["matched_rank"]
        overall.index = pd.MultiIndex.from_tuples(
            [(arm, head, rc) for head, rc in overall.index], names=["arm", "head", "range_class"]
        )
        rows.append(overall)
    if rows:
        add(
            "programme masking: planted sender's rank in its receiver's predicted response among cells as far (0.5: blind)",
            pd.concat(rows).unstack("range_class"),
        )
    rows = []
    for arm, table in _per_arm(sections, "response_check", arms).items():
        overall = table.xs("any", level="program")["completeness_gap"]
        rows.append(overall.rename(arm))
    if rows:
        add(
            "programme masking, attribution: share of the response's change from the baseline the contributions "
            "fail to add up to (0: complete; over 0.1: too few integration steps)",
            pd.DataFrame(rows),
        )
    rows = []
    for arm, table in _per_arm(sections, "attention_check", arms).items():
        worst = table.groupby(level=["check", "quantity"], sort=False)["max_abs_diff"].max()
        worst.index = [f"{check} {quantity}" for check, quantity in worst.index]
        rows.append(worst.rename(arm))
    if rows:
        add(
            "attention recomputed from each layer's inputs: largest absolute difference over the layers "
            "(reference = the definition, torch = torch's stock modules; ~1e-6 is rounding)",
            pd.DataFrame(rows),
        )
    parts = []
    for arm, table in _per_arm(sections, "readout_edges", arms).items():
        part = table.copy()
        part.index = pd.MultiIndex.from_tuples([(arm, r) for r in part.index], names=["arm", "readout"])
        parts.append(part)
    if parts:
        add(
            "ways of reading attention, planted pairs: sender's rank among cells as far from the receiver (0.5: blind)",
            pd.concat(parts),
        )
    rows = []
    for arm, table in _per_arm(sections, "readout_span", arms).items():
        if "span_um" in table.columns:
            rows.append(table["span_um"].rename(arm))
    if rows:
        add("ways of reading attention: score-weighted distance to the query (µm)", pd.DataFrame(rows))
    parts = []
    for arm, table in _per_arm(sections, "flow_summary", arms).items():
        part = table.copy()
        part.index = pd.MultiIndex.from_tuples([(arm, kind) for kind in part.index], names=["arm", "kind"])
        parts.append(part)
    if parts:
        add(
            "patch flow against the expression-permuted slide: reach (µm), self share, KL and centre-of-mass shift "
            "with their permutation floors, share above the null",
            pd.concat(parts),
        )
    rows = [table["seconds"].rename(arm) for arm, table in _per_arm(sections, "timing", arms).items() if "seconds" in table]
    if rows:
        add("evaluation: seconds per step", pd.DataFrame(rows))
    rows = [table["value"].rename(arm) for arm, table in _per_arm(sections, "training", arms).items()]
    if rows:
        add("training: the last epoch's throughput and peak memory, the final validation metrics", pd.DataFrame(rows))
    parts = []
    for arm, table in _per_arm(sections, "phases", arms).items():
        part = table.copy()
        part.index = pd.MultiIndex.from_tuples([(arm, phase) for phase in part.index], names=["arm", "phase"])
        parts.append(part)
    if parts:
        add(
            "staged training, each phase's end: relative weight change per component (0: frozen), last validation losses",
            pd.concat(parts),
        )
    return out


def render_summary(tables: list[tuple[str, pd.DataFrame]]) -> str:
    """The summary tables as text: a ``###`` title and an aligned table per entry."""
    parts = []
    for title, table in tables:
        with pd.option_context("display.width", 250, "display.max_columns", None, "display.max_colwidth", 40):
            parts.append(f"### {title}\n\n```text\n{table.round(3).to_string()}\n```\n")
    return "\n".join(parts)
