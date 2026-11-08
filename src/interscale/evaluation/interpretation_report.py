"""The readouts of :mod:`interscale.evaluation.interpretation` as something to read: a report per run and a summary over runs.

Everything is rebuilt from the tables a run wrote to ``<run>/tables/`` (:func:`write_tables`), so a report can be redrawn anywhere without a model:
:func:`write_run_report` for one run, :func:`write_summary` for a folder of runs. Figures show a matrix or the tissue alone. Two rules for the wording: a flow is
"what the prediction lost when these cells were changed", never signalling; and a condition difference from one run is one donor pair, so the summary reads it over folds.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.stats import mannwhitneyu, spearmanr

from interscale.evaluation.interpretation import EVERYONE, flow_matrix, lr_cluster_p, match_programmes

SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
#: Blue <-> red with a neutral midpoint: polarity. The same ramp as the permutation notebook's net-flow figures.
DIVERGING = LinearSegmentedColormap.from_list(
    "interpretation_diverging", ["#0d366b", "#256abf", "#9ec5f4", "#f0efec", "#f2a3a2", "#d03b3b", "#6e1817"]
)
UNTRAINED = "_untrained"
#: A programme's score drawn on the largest tissue pieces: at most this many programmes per figure.
MAP_PROGRAMMES = 6
TABLE_ROWS = 15


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
        }
    )


def md_table(frame: pd.DataFrame, digits: int = 3) -> str:
    """A Markdown table; floats at ``digits`` significant digits, nothing else needed to render it."""

    def cell(value) -> str:
        if isinstance(value, float | np.floating):
            return "" if not np.isfinite(value) else f"{value:.{digits}g}"
        return str(value).replace("|", "/")

    if not len(frame):
        return "_(empty)_"
    lines = ["| " + " | ".join(str(c) for c in frame.columns) + " |", "|" + "---|" * frame.shape[1]]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines)


def _read(folder: Path, name: str) -> pd.DataFrame:
    path = folder / f"{name}.csv"
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def write_tables(folder, tables: dict) -> None:
    """Write ``{name: DataFrame}`` as ``<folder>/tables/<name>.csv``; a dict or list goes to ``<name>.json``, an array to ``<name>.npy``."""
    folder = Path(folder) / "tables"
    folder.mkdir(parents=True, exist_ok=True)
    for name, value in tables.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(folder / f"{name}.csv", index=False)
        elif isinstance(value, np.ndarray):
            np.save(folder / f"{name}.npy", value)
        else:
            (folder / f"{name}.json").write_text(json.dumps(value, indent=1, default=float))


# ----------------------------------------------------------------------------------------------- figures


def _order(names, first=()) -> list[str]:
    return [n for n in first if n in names] + sorted(n for n in names if n not in first)


def matrix_figure(
    panels: list[tuple[str, pd.DataFrame]],
    path,
    *,
    title: str,
    note: str,
    unit: str = "drop in centred cosine x 10⁻³",
    scale: float = 1e3,
    xlabel="Receiver",
    ylabel="Sender",
    marks: dict[str, pd.DataFrame] | None = None,
):
    """One heatmap per panel, one symmetric colour scale. The value is written only into the strongest cells of a panel, so a table stays the place to read numbers.
    ``marks``: per panel name a boolean table of the same shape; its true cells get an outline (e.g. "repeats in every run and beyond the twin").
    """
    _style()
    panels = [(name, table) for name, table in panels if len(table)]
    if not panels:
        return None
    columns = min(3, len(panels))
    rows = int(np.ceil(len(panels) / columns))
    size = max(len(t.columns) for _, t in panels)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * (1.6 + 0.42 * size) + 1.0, rows * (1.4 + 0.36 * size) + 0.9),
        squeeze=False,
        layout="constrained",
        dpi=150,
    )
    limit = max(float(np.nanmax(np.abs(t.to_numpy(dtype=float)))) for _, t in panels) * scale
    limit = limit if np.isfinite(limit) and limit > 0 else 1.0
    image = None
    for ax, (name, table) in zip(axes.ravel(), panels, strict=False):
        values = table.to_numpy(dtype=float) * scale
        image = ax.imshow(values, cmap=DIVERGING, vmin=-limit, vmax=limit, aspect="equal")
        ax.set_xticks(np.arange(table.shape[1]), table.columns, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(np.arange(table.shape[0]), table.index, fontsize=8)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(name, loc="left", fontsize=9, fontweight="bold")
        ax.set_xticks(np.arange(table.shape[1] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(table.shape[0] + 1) - 0.5, minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="both", length=0)
        ax.spines[:].set_visible(False)
        marked = (marks or {}).get(name)
        if marked is not None and len(marked):
            flags = marked.reindex(index=table.index, columns=table.columns).fillna(False).to_numpy(dtype=bool)
            for i, j in zip(*np.nonzero(flags), strict=True):
                ax.add_patch(plt.Rectangle((j - 0.46, i - 0.46), 0.92, 0.92, fill=False, edgecolor=INK, linewidth=1.4))
        finite = np.abs(values[np.isfinite(values)])
        if len(finite):
            cut = np.quantile(finite, 0.9)
            for i, j in zip(*np.nonzero(np.nan_to_num(np.abs(values)) >= max(cut, 1e-12)), strict=True):
                shade = abs(values[i, j]) / limit
                ax.text(
                    j,
                    i,
                    f"{values[i, j]:.3g}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="#ffffff" if shade > 0.6 else INK,
                )
    for ax in axes.ravel()[len(panels) :]:
        ax.set_visible(False)
    fig.colorbar(image, ax=axes, shrink=0.6, label=unit)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.supxlabel(note, x=0.01, ha="left", fontsize=7.5, color=INK_2)
    fig.savefig(path)
    plt.close(fig)
    return path


def programme_type_figure(by_type: pd.DataFrame, cards: pd.DataFrame, path):
    """Programmes x cell types: mean score in within-section SD; each row named by its class and leading genes."""
    _style()
    if not len(by_type):
        return None
    table = by_type.pivot(index="programme", columns="cell_type", values="z")
    names = cards.set_index("programme")
    labels = [
        f"P{p} {names.loc[p, 'class']}: " + " ".join(str(names.loc[p, "top_genes"]).split()[:3]) for p in table.index
    ]
    limit = float(np.nanmax(np.abs(table.to_numpy()))) or 1.0
    fig, ax = plt.subplots(
        figsize=(3.6 + 0.5 * table.shape[1], max(3.0, 1.2 + 0.32 * table.shape[0])), layout="constrained", dpi=150
    )
    image = ax.imshow(table.to_numpy(), cmap=DIVERGING, vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(table.shape[1]), table.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(np.arange(table.shape[0]), labels, fontsize=8)
    ax.set_xticks(np.arange(table.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(table.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)
    ax.spines[:].set_visible(False)
    ax.set_title("Programmes by cell type", loc="left", fontsize=11, fontweight="bold")
    fig.colorbar(image, ax=ax, shrink=0.7, label="mean score (SD)")
    fig.savefig(path)
    plt.close(fig)
    return path


def programme_map_figure(pos, scores, cards: pd.DataFrame, path, *, title: str):
    """The first programmes on one piece of tissue, one tile each, nothing but cells. ``pos`` ``[N, 2]`` and ``scores`` ``[N, k]`` of that piece."""
    _style()
    k = min(MAP_PROGRAMMES, scores.shape[1])
    columns = min(3, k)
    rows = int(np.ceil(k / columns))
    span = np.ptp(pos, axis=0)
    ratio = float(span[1] / max(span[0], 1e-9))
    width = 4.0
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * width, rows * width * ratio + 0.5),
        squeeze=False,
        layout="constrained",
        dpi=150,
    )
    names = cards.set_index("programme")
    for c, ax in enumerate(axes.ravel()):
        if c >= k:
            ax.set_visible(False)
            continue
        value = scores[:, c]
        limit = float(np.quantile(np.abs(value), 0.99)) or 1.0
        order = np.argsort(np.abs(value))
        ax.scatter(
            pos[order, 0],
            pos[order, 1],
            c=value[order],
            s=2.0,
            linewidths=0,
            cmap=DIVERGING,
            norm=TwoSlopeNorm(0.0, -limit, limit),
        )
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(
            f"P{c} {names.loc[c, 'class']}: " + " ".join(str(names.loc[c, "top_genes"]).split()[:4]),
            loc="left",
            fontsize=8,
        )
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.savefig(path)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------------------------- one run


def _condition_panels(flows: pd.DataFrame, *, kind: str, region: str) -> list[tuple[str, pd.DataFrame]]:
    panels, per_condition = [], {}
    conditions = sorted(flows["condition"].dropna().unique()) if "condition" in flows else []
    for condition in conditions:
        table = flow_matrix(flows[flows["condition"] == condition], kind=kind, region=region)
        if len(table):
            per_condition[condition] = table
            panels.append((str(condition), table))
    if not conditions:
        table = flow_matrix(flows, kind=kind, region=region)
        if len(table):
            panels.append(("all sections", table))
    if len(per_condition) == 2:
        (a, ta), (b, tb) = sorted(per_condition.items())
        panels.append((f"{b} − {a}", tb.sub(ta)))
    return panels


def top_flows(flows: pd.DataFrame, targets: pd.DataFrame, *, n: int = TABLE_ROWS) -> pd.DataFrame:
    """The largest flows over kinds and regions, each with the targets that moved most, a sign for the direction the prediction moved."""
    rows = []
    for region in sorted(flows["region"].unique()) if len(flows) else []:
        for kind in ("identity", "state"):
            matrix = flow_matrix(flows, kind=kind, region=region)
            if not len(matrix):
                continue
            for (sender, receiver), value in matrix.stack().items():
                if value > 0:
                    rows.append(
                        {
                            "sender": sender,
                            "receiver": receiver,
                            "kind": kind,
                            "region": region,
                            "delta_cos": float(value),
                        }
                    )
    table = pd.DataFrame(rows, columns=["sender", "receiver", "kind", "region", "delta_cos"])
    if not len(table):
        return table
    table = table.sort_values("delta_cos", ascending=False, kind="stable").head(int(n)).reset_index(drop=True)
    genes = []
    for row in table.itertuples():
        sub = targets
        if len(targets):
            sub = targets[
                (targets["region"] == row.region)
                & (targets["kind"] == row.kind)
                & (targets["sender"] == row.sender)
                & (targets["receiver"] == row.receiver)
            ]
            # One row per gene: the same target can lead in several sections.
            sub = (
                sub.groupby("gene", as_index=False)[["delta_cos", "shift"]]
                .mean()
                .sort_values("delta_cos", ascending=False, kind="stable")
                .head(5)
            )
        genes.append(
            " ".join(f"{g}{'+' if s > 0 else '-'}" for g, s in zip(sub["gene"], sub["shift"], strict=True))
            if len(sub)
            else ""
        )
    return table.assign(top_targets=genes)


def _context_verdict(use: pd.DataFrame) -> str:
    part = use[use["region"] == "context"] if len(use) else use
    if not len(part):
        return "No context-use table: no receiver was scored."
    state = part.set_index("receiver")["state"]
    arrangement = part.set_index("receiver")["arrangement"]
    return (
        f"Swapping every other cell's profile **within its type** lowers the prediction's centred cosine within the receiver type by {state.mean():.4f} on average over receiver types (largest: {state.idxmax()}, {state.max():.4f}); "
        f"swapping **among all cells** {arrangement.mean():.4f} (largest: {arrangement.idxmax()}, {arrangement.max():.4f}). Near 0 means the prediction does not rest on those cells, "
        "whatever a flow plot shows. Compare with the untrained twin in `summary.md`."
    )


def lr_row(tables) -> dict:
    """The known-pair summary of a run, with the p that allows for shared genes (:func:`interpretation.lr_cluster_p`). Tables written before 2026-11-08 have
    only the per-pair p: it is recomputed from ``lr_pairs`` and kept as ``p_independent``. ``{}`` when the run has no known-pair test.
    """
    lr = _read(Path(tables), "lr_summary")
    if not len(lr):
        return {}
    row = lr.iloc[0].to_dict()
    if not np.isfinite(float(row.get("cluster_sd", np.nan))):
        pairs = _read(Path(tables), "lr_pairs")
        if len(pairs):
            row["p_independent"] = row.get("p", np.nan)
            row |= lr_cluster_p(pairs, float(row["null_mean"]), float(row["null_sd"]))
    return row


def _arm(name: str) -> str:
    """A run's name without its seed suffix (``_s42``) or ``_untrained``."""
    return re.sub(r"_s\d+$", "", name[: -len(UNTRAINED)] if name.endswith(UNTRAINED) else name)


def twin_of(run_dir) -> Path | None:
    """The untrained twin next to a trained run: its own (``<run>_untrained``), else the twin of another seed of the same arm (the same architecture at
    another initialisation). None for a twin or when there is none.
    """
    run_dir = Path(run_dir)
    if run_dir.name.endswith(UNTRAINED):
        return None
    own = run_dir.parent / f"{run_dir.name}{UNTRAINED}"
    if (own / "tables").exists():
        return own
    for other in sorted(run_dir.parent.glob(f"*{UNTRAINED}")):
        if (other / "tables").exists() and _arm(other.name) == _arm(run_dir.name):
            return other
    return None


def planted_against_twin(run_dir) -> dict:
    """``first``: the share of planted targets whose best planted sender ranks first; ``twin_first`` the same in the twin; ``beyond_twin``: first here and not
    first in the twin. A planted sender can rank first at random weights too (synth_spot 2026-11-08: the twin 5 of 15, its ranks 1 or last: a large effect of
    random sign), so the bar is the twin's, not 1 / senders.
    """
    recovery = _read(Path(run_dir) / "tables", "edge_recovery")
    if not len(recovery):
        return {}
    found = {"first": float((recovery["best_rank"] == 1).mean()), "targets": len(recovery), "chance": float(recovery["chance_top1"].mean())}
    twin = twin_of(run_dir)
    other = _read(twin / "tables", "edge_recovery") if twin is not None else pd.DataFrame()
    if len(other):
        ranks = recovery.set_index("target")["best_rank"]
        theirs = other.set_index("target")["best_rank"].reindex(ranks.index)
        found |= {"twin": twin.name, "twin_first": float((theirs == 1).mean()), "beyond_twin": float(((ranks == 1) & (theirs != 1)).mean())}
    return found


def _sections_note(flows: pd.DataFrame) -> str:
    """What a condition difference of this run rests on: how many test sections each condition has."""
    if not len(flows) or "condition" not in flows or "section" not in flows:
        return "- Everything is read on this run's test sections."
    counts = flows.drop_duplicates("section").groupby("condition")["section"].count()
    text = ", ".join(f"{n} {c}" for c, n in counts.items())
    if len(counts) == 2 and counts.min() == 1:
        return f"- Everything is read on the test sections of one fold ({text}). Differences between them are one donor pair, confounded with the section: read them across folds in `summary.md`."
    return f"- Everything is read on {text} test sections. A condition difference is read over sections in `summary.md`; runs that share their test sections share their donors."


# ----------------------------------------------------------------------------------------------- which genes matter, and are they spatially variable

#: The share of genes a spatial statistic must lead to count as spatially variable (top quarter), and how many of the most influential genes are read.
SVG_QUANTILE, TOP_GENES = 0.75, 20


def _expression_bins(mean: pd.Series, n_bins: int = 10) -> pd.Series:
    return pd.qcut(mean.rank(method="first"), q=min(int(n_bins), max(1, int(mean.notna().sum()))), labels=False, duplicates="drop")


def _spearman_within(a: pd.Series, b: pd.Series, bins: pd.Series) -> float:
    """Spearman of ``a`` and ``b`` within each expression bin, averaged weighted by the bin's genes: what is left when expression is held fixed."""
    found, weights = [], []
    for _, idx in bins.dropna().groupby(bins.dropna()).groups.items():
        both = pd.concat([a.reindex(idx), b.reindex(idx)], axis=1).dropna()
        if len(both) > 3 and both.iloc[:, 0].nunique() > 1 and both.iloc[:, 1].nunique() > 1:
            found.append(spearmanr(both.iloc[:, 0], both.iloc[:, 1]).statistic)
            weights.append(len(both))
    return float(np.average(found, weights=weights)) if found else float("nan")


def influence_table(table: pd.DataFrame, twin: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per gene of ``gene_influence``: ``learned_near`` / ``learned_far`` (the influence minus the untrained twin's, the influence alone without a twin), the
    spatial percentiles and classes. ``svg``: in the top quarter of genes by local or by global Moran's I within type; ``scale``: ``global`` where the global
    percentile is at least the local one, else ``local``, for spatially variable genes; ``flat`` where both percentiles are below one half; else ``mid``.
    """
    out = table.copy()
    for region in ("near", "far"):
        column = f"influence_{region}"
        if column not in out:
            out[f"learned_{region}"] = np.nan
            continue
        base = twin[column].reindex(out.index) if twin is not None and column in twin else 0.0
        out[f"learned_{region}"] = out[column] - base
    local, glob = out["moran_local"].rank(pct=True), out["moran_global"].rank(pct=True)
    out["svg"] = (local >= SVG_QUANTILE) | (glob >= SVG_QUANTILE)
    out["scale"] = np.where(out["svg"], np.where(glob >= local, "global", "local"), np.where((local < 0.5) & (glob < 0.5), "flat", "mid"))
    out["expression_bin"] = _expression_bins(out["mean"])
    return out


def svg_among_top(table: pd.DataFrame, column: str, *, top: int = TOP_GENES, n_perm: int = 2000, seed: int = 0) -> dict:
    """Are the ``top`` genes by ``column`` spatially variable more often than genes of the same expression? ``observed`` share, ``expected`` (each top gene's
    expression bin's share), ``p``: draws of as many genes from the same bins, the share at least the observed (a gene's expression alone makes it both
    influential and patterned, which this takes away).
    """
    ranked = table[[column, "svg", "expression_bin"]].dropna(subset=[column, "expression_bin"])
    if len(ranked) < 2 * int(top):
        return {}
    chosen = ranked.sort_values(column, ascending=False, kind="stable").head(int(top))
    rate = ranked.groupby("expression_bin")["svg"].mean()
    observed = float(chosen["svg"].mean())
    rng = np.random.default_rng(seed)
    pools = {b: ranked.index[ranked["expression_bin"] == b].to_numpy() for b in chosen["expression_bin"].unique()}
    counts = chosen["expression_bin"].value_counts()
    svg = ranked["svg"]
    null = np.array(
        [np.mean(np.concatenate([svg.loc[rng.choice(pools[b], size=n, replace=False)].to_numpy() for b, n in counts.items()])) for _ in range(int(n_perm))]
    )
    return {
        "top": int(top),
        "observed": observed,
        "expected": float(chosen["expression_bin"].map(rate).mean()),
        "base_rate": float(svg.mean()),
        "p": float((1 + np.sum(null >= observed)) / (1 + len(null))),
        "genes": " ".join(f"{g}{'*' if v else ''}" for g, v in zip(chosen.index, chosen["svg"], strict=True)),
    }


def influence_read(table: pd.DataFrame, twin: pd.DataFrame | None = None, *, seed: int = 0) -> dict:
    """The read of one run's ``gene_influence`` table against its twin's: ``genes`` (:func:`influence_table`), ``correlations`` (influence against each spatial
    statistic, raw and within expression bins), ``top`` (:func:`svg_among_top` for the near and the far influence), ``separation`` (among spatially variable
    genes, how much more a gene leans far than near, by percentile: global genes against local ones).
    """
    genes = influence_table(table, twin)
    rows = []
    for region in ("near", "far"):
        for stat in ("moran_local", "moran_global", "moran_local_raw"):
            a, b = genes[f"learned_{region}"], genes[stat]
            both = pd.concat([a, b], axis=1).dropna()
            rows.append(
                {
                    "influence": f"learned_{region}",
                    "spatial": stat,
                    "spearman": float(spearmanr(both.iloc[:, 0], both.iloc[:, 1]).statistic) if len(both) > 3 else np.nan,
                    "within_expression": _spearman_within(a, b, genes["expression_bin"]),
                    "genes": len(both),
                }
            )
    top = [{"influence": f"learned_{r}"} | svg_among_top(genes, f"learned_{r}", seed=seed) for r in ("near", "far")]
    separation = {}
    lean = genes["learned_far"].rank(pct=True) - genes["learned_near"].rank(pct=True)
    local, glob = lean[genes["scale"] == "local"].dropna(), lean[genes["scale"] == "global"].dropna()
    if len(local) > 2 and len(glob) > 2:
        separation = {
            "global_genes": len(glob),
            "local_genes": len(local),
            "lean_far_global": float(glob.median()),
            "lean_far_local": float(local.median()),
            "p": float(mannwhitneyu(glob, local, alternative="greater").pvalue),
        }
    return {"genes": genes.assign(lean_far=lean), "correlations": pd.DataFrame(rows), "top": pd.DataFrame([t for t in top if len(t) > 1]), "separation": separation}


def influence_figure(genes: pd.DataFrame, path, *, title: str):
    """Learned influence against Moran's I within type: near against local, far against global; spatially variable genes marked."""
    _style()
    pairs = [(stat, f"learned_{region}") for stat, region in (("moran_local", "near"), ("moran_global", "far")) if genes[f"learned_{region}"].notna().any()]
    if not pairs:
        return None
    fig, axes = plt.subplots(1, len(pairs), figsize=(5.2 * len(pairs), 4.4), squeeze=False)
    for ax, (stat, column) in zip(axes[0], pairs, strict=True):
        part = genes[[stat, column, "svg"]].dropna()
        ax.scatter(part.loc[~part["svg"], stat], part.loc[~part["svg"], column], s=9, color="#b8b6b0", lw=0, label="other genes")
        ax.scatter(part.loc[part["svg"], stat], part.loc[part["svg"], column], s=11, color="#d03b3b", lw=0, label="spatially variable (top quarter)")
        for gene in part[column].nlargest(8).index:
            ax.annotate(gene, (part.loc[gene, stat], part.loc[gene, column]), fontsize=7, color=INK_2, xytext=(3, 2), textcoords="offset points")
        ax.axhline(0.0, color=INK_2, lw=0.6)
        ax.set_xlabel(f"{stat.replace('_', ' ')} within type")
        ax.set_ylabel(f"{column.replace('_', ' ')} (minus the twin)")
    axes[0][0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle(title, fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def write_run_report(run_dir) -> Path:
    """``<run>/report.md`` and ``<run>/figures/`` from ``<run>/tables/``. Sections whose tables are missing say so."""
    run_dir = Path(run_dir)
    t = run_dir / "tables"
    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
    out = [f"# {run_dir.name}: from the model to biology", ""]
    flows, targets = _read(t, "flows"), _read(t, "flow_targets")
    use = _read(t, "context_use")

    out += ["## 1. Does the prediction use other cells at all?", "", _context_verdict(use), "", md_table(use), ""]

    out += ["## 2. Who affects whom", ""]
    out += [
        "Other cells are changed and the receivers' own rows kept; the number is how much worse the receivers' hidden genes are predicted. `identity`: the sender type's cells take another type's "
        "profile, minus the same change on the cells next to them (co-location taken off). `state`: the sender type's cells swap profiles among themselves, so types stay where they are. "
        "`far`: only cells beyond the near radius change. The sign in `top_targets` is the direction the prediction moved.",
        "",
    ]
    if len(flows):
        for region, label in (("context", "all other cells"), ("far", "only cells beyond the near radius")):
            panels = [(f"identity · {n}", m) for n, m in _condition_panels(flows, kind="identity", region=region)]
            panels += [(f"state · {n}", m) for n, m in _condition_panels(flows, kind="state", region=region)]
            path = matrix_figure(
                panels,
                figures / f"flows_{region}.png",
                title=f"{run_dir.name}: what the prediction loses, {label}",
                note="Row: the cells changed. Column: the cells scored. Red: the receivers are predicted worse. One colour scale for all panels.",
            )
            if path:
                out += [f"![flows {region}](figures/{Path(path).name})", ""]
        out += [md_table(top_flows(flows, targets)), ""]
    else:
        out += ["_No flow table._", ""]

    out += ["## 3. Through which genes", ""]
    gene_top, lr_pairs = _read(t, "gene_top"), _read(t, "lr_pairs")
    row = lr_row(t)
    if row:
        out += [
            f"Known ligand-receptor pairs against other targets of the same ligand in the same receiver type (matched on detection in that type): {int(row['n_pairs'])} pair x type reads, "
            f"mean rank {row['auc']:.3f}, null {row['null_mean']:.3f} ± {row['null_sd']:.3f}. The pairs share genes (one receptor serves many ligands and types), so the observed mean "
            f"varies by ± {row.get('cluster_sd', np.nan):.3f} over ligand and receptor genes: z = {row.get('z', np.nan):.2f}, p = {row['p']:.3g} "
            f"(as if every pair were independent: p = {row.get('p_independent', np.nan):.3g}, too small).",
            "",
        ]
    if len(gene_top):
        pairs = (
            {(a, b) for a, b in zip(lr_pairs.get("ligand", []), lr_pairs.get("receptor", []), strict=True)}
            if len(lr_pairs)
            else set()
        )
        shown = gene_top.head(TABLE_ROWS).assign(
            known_pair=lambda f: [(a, b) in pairs for a, b in zip(f["sender"], f["target"], strict=True)]
        )
        out += [
            "The sender gene is permuted within its cell type in the context of the receivers. This table always fills: compare its size with the twin's in `summary.md`.",
            "",
            md_table(shown),
            "",
        ]
        if len(lr_pairs):
            out += ["Known pairs with the largest effect:", "", md_table(lr_pairs.head(TABLE_ROWS)), ""]
    else:
        out += ["_No gene-flow table._", ""]
    planted = planted_against_twin(run_dir)
    if planted:
        line = f"Best planted sender first for {planted['first']:.0%} of {planted['targets']} targets."
        if "twin_first" in planted:
            line += (
                f" The untrained twin ({planted['twin']}): {planted['twin_first']:.0%}; first here but not in the twin: {planted['beyond_twin']:.0%}. Read it against the twin, not against "
                f"drawing a sender at random ({planted['chance']:.1%}): a planted sender can move the prediction at random weights too."
            )
        else:
            line += f" Drawing a sender at random: {planted['chance']:.1%}; no untrained twin to compare with, and a twin can rank planted senders first too."
        out += ["Planted truth, the check of this readout:", "", line, ""]

    out += ["## 4. Attention, read biologically", ""]
    pairs_table, by_type, attention_genes = (
        _read(t, "attention_pairs"),
        _read(t, "attention_by_type"),
        _read(t, "attention_genes"),
    )
    if len(pairs_table):
        panels = []
        for condition in sorted(pairs_table["condition"].dropna().unique()) if "condition" in pairs_table else []:
            part = pairs_table[pairs_table["condition"] == condition]
            panels.append((str(condition), part.pivot(index="source", columns="receiver", values="log2_ratio")))
        path = matrix_figure(
            panels,
            figures / "attention_pairs.png",
            title=f"{run_dir.name}: top attention sources against local label shuffles",
            note="log2(observed / expected), expected from labels shuffled within 100 µm squares: more than co-location explains. Read against the twin.",
            unit="log2 ratio",
            scale=1.0,
            ylabel="Source",
        )
        if path:
            out += [f"![attention pairs](figures/{Path(path).name})", ""]
        if len(flows) and "condition" in pairs_table:
            intervention = flow_matrix(flows, kind="identity", region="context").stack().rename("intervention")
            read = pairs_table.groupby(["source", "receiver"])["log2_ratio"].mean().rename("attention")
            read.index.names = ["sender", "receiver"]
            joined = pd.concat([intervention, read], axis=1).dropna()
            if len(joined) > 3:
                rho = spearmanr(joined["intervention"], joined["attention"]).statistic
                out += [
                    f"Does attention agree with what the model uses? Spearman of the two type-by-type tables: {rho:.2f} over {len(joined)} pairs.",
                    "",
                ]
    if len(by_type):
        summary = (
            by_type.groupby("cell_type")[["mean_net", "share_top_decile", "share_bottom_decile"]]
            .mean()
            .sort_values("mean_net", ascending=False)
            .reset_index()
        )
        out += [
            "Sources (read more than they read) and sinks by type, smoothed net flow, mean over sections. A share of 0.1 is chance:",
            "",
            md_table(summary),
            "",
        ]
    if len(attention_genes):
        high = attention_genes.sort_values("r", ascending=False, kind="stable")
        out += [
            "Within section and type, genes high in sources: "
            + ", ".join(f"{g} ({r:.2f})" for g, r in zip(high.head(12)["gene"], high.head(12)["r"], strict=True))
            + ".",
            "",
            "Genes high in sinks: "
            + ", ".join(
                f"{g} ({r:.2f})" for g, r in zip(high.tail(12)["gene"][::-1], high.tail(12)["r"][::-1], strict=True)
            )
            + ".",
            "",
        ]
    enrich = _read(t, "attention_enrichment")
    if len(enrich):
        out += [
            md_table(
                enrich[enrich["q"] < 0.05].head(TABLE_ROWS)[
                    ["side", "library", "set", "overlap", "set_size", "q", "genes"]
                ]
            ),
            "",
        ]
    composition = _read(t, "attention_composition")
    if len(composition):
        out += [
            "Net flow against the neighbourhood's composition, within section and type:",
            "",
            md_table(composition.sort_values("r_with_neighbour_fraction", ascending=False)),
            "",
        ]

    out += ["## 5. What the embedding's programmes are", ""]
    cards, type_table = _read(t, "programme_cards"), _read(t, "programme_by_type")
    if len(cards):
        out += [
            "The prediction's main directions, rotated to sparse programmes. `class`, by the thresholds in `interpretation.CLASS_THRESHOLDS`: `handling` (stress, ribosomal, spillover genes dominate), `niche` (varies in space within a type and "
            "follows the neighbourhood), `spatial state` (varies in space within a type), `identity` (mostly the cell type), `weak`. `moran_within_*`: spatial autocorrelation of what the type leaves, at 8 and 128 neighbours. "
            "`condition_effect`: one donor pair, standardised, in the type with the largest difference.",
            "",
        ]
        columns = [
            "programme",
            "share",
            "class",
            "top_type",
            "eta2",
            "moran_within_near",
            "moran_within_far",
            "neighbour_r2",
            "handling_share",
            "condition_type",
            "condition_effect",
            "top_genes",
            "pathway",
        ]
        out += [md_table(cards[[c for c in columns if c in cards.columns]]), ""]
        path = programme_type_figure(type_table, cards, figures / "programmes_by_type.png")
        if path:
            out += [f"![programmes by type](figures/{Path(path).name})", ""]
        maps = t / "programme_map.npz"
        if maps.exists():
            with np.load(maps, allow_pickle=False) as saved:
                path = programme_map_figure(
                    saved["pos"],
                    saved["scores"],
                    cards,
                    figures / "programmes_map.png",
                    title=f"{run_dir.name}: programme scores on {saved['name']}",
                )
            out += [f"![programmes on tissue](figures/{Path(path).name})", ""]
    else:
        out += ["_No programme table: the decoder is not linear, or the readout failed._", ""]

    out += ["## 6. Known biology", ""]
    programme_enrich = _read(t, "programme_enrichment")
    controls = _read(t, "control_genes")
    if len(programme_enrich) or len(controls):
        out += [
            "The proposed control sets (`biology_resource.json`; to be checked) in the programmes and the attention. Identity sets should be found by any model that fits; the T1D sets are the question; the handling sets warn.",
            "",
        ]
        if len(programme_enrich):
            found = (
                programme_enrich[programme_enrich["library"] == "controls"]
                .sort_values("q", kind="stable")
                .groupby("set", as_index=False)
                .first()
            )
            out += [md_table(found[["set", "programme", "side", "overlap", "set_size", "q", "genes"]]), ""]
        if len(controls):
            out += [
                "Where the control genes stand in the attention's within-type ranking (0: strongest in sinks, 1: strongest in sources):",
                "",
                md_table(controls),
                "",
            ]

    influence = _read(t, "gene_influence")
    if len(influence):
        influence = influence.set_index("gene")
        twin = twin_of(run_dir)
        theirs = _read(twin / "tables", "gene_influence").set_index("gene") if twin is not None and len(_read(twin / "tables", "gene_influence")) else None
        found = influence_read(influence, theirs)
        out += [
            "## 7. Which genes matter, and are they the spatially variable ones",
            "",
            "Each visible gene permuted within its cell type, in every cell but the receivers (`near`) or only beyond the near radius (`far`); its influence is the mean drop "
            "in centred cosine over the receivers' hidden genes, "
            + (f"minus the untrained twin's ({twin.name})." if theirs is not None else "**without a twin to subtract**: read with care.")
            + " Spatially variable: the top quarter of genes by Moran's I within type, locally (the 8 nearest cells) or across the section. "
            "`within_expression`: the Spearman inside expression deciles, since abundant genes are both influential and patterned.",
            "",
            md_table(found["correlations"]),
            "",
        ]
        if len(found["top"]):
            out += [
                f"The {TOP_GENES} most influential genes (`*`: spatially variable), against genes of the same expression:",
                "",
                md_table(found["top"]),
                "",
            ]
        if found["separation"]:
            sep = found["separation"]
            out += [
                f"Local and global separated? Among spatially variable genes, how much a gene leans to far influence (percentile far minus near): global genes {sep['lean_far_global']:+.2f} "
                f"(n = {sep['global_genes']}), local genes {sep['lean_far_local']:+.2f} (n = {sep['local_genes']}); one-sided p = {sep['p']:.3g}.",
                "",
            ]
        path = influence_figure(found["genes"], figures / "influence_vs_moran.png", title=f"{run_dir.name}: influence against spatial pattern")
        if path:
            out += [f"![influence against Moran's I](figures/{Path(path).name})", ""]
    out += [
        "## How far this goes",
        "",
        "- A flow is what this model's prediction loses, not signalling. A real interaction can be missed and a correlation can look like one; agreement with priors is support, not proof.",
        _sections_note(flows),
        "- Identity flows partly reflect segmentation spillover. Check `handling_share` and the targets before reading a pair biologically.",
    ]
    path = run_dir / "report.md"
    path.write_text("\n".join(out) + "\n")
    return path


# ----------------------------------------------------------------------------------------------- over runs


def _runs(results) -> list[Path]:
    return sorted(p.parent for p in Path(results).glob("runs/*/tables"))


def _base(name: str) -> str:
    return name[: -len(UNTRAINED)] if name.endswith(UNTRAINED) else name


def flows_over_runs(frames: dict[str, pd.DataFrame], *, kind: str, region: str = "context") -> pd.DataFrame:
    """Per sender and receiver: the flow in each run and how many agree in sign. Trained runs only; twins are read on their own."""
    columns = {}
    for name, flows in frames.items():
        if not name.endswith(UNTRAINED) and len(flows):
            matrix = flow_matrix(flows, kind=kind, region=region)
            if len(matrix):
                columns[name] = matrix.stack()
    if not columns:
        return pd.DataFrame()
    table = pd.DataFrame(columns)
    table.index.names = ["sender", "receiver"]
    values = table.to_numpy(dtype=float)
    table["mean"] = np.nanmean(values, axis=1)
    finite = np.isfinite(values).sum(axis=1)
    table["sd"] = [float(np.nanstd(row, ddof=1)) if n > 1 else np.nan for row, n in zip(values, finite, strict=True)]
    table["same_sign"] = [int(max((row > 0).sum(), (row < 0).sum())) for row in values]
    table["runs"] = finite
    return table.reset_index().sort_values("mean", ascending=False, kind="stable").reset_index(drop=True)


def condition_over_runs(
    frames: dict[str, pd.DataFrame], *, kind: str = "state", region: str = "context"
) -> pd.DataFrame:
    """T1D minus ND (the later condition name minus the earlier) of each flow per run: the folds are the donor pairs. ``same_sign`` of ``runs``."""
    columns = {}
    for name, flows in frames.items():
        if name.endswith(UNTRAINED) or not len(flows) or "condition" not in flows:
            continue
        conditions = sorted(flows["condition"].dropna().unique())
        if len(conditions) != 2:
            continue
        a = flow_matrix(flows[flows["condition"] == conditions[0]], kind=kind, region=region)
        b = flow_matrix(flows[flows["condition"] == conditions[1]], kind=kind, region=region)
        if len(a) and len(b):
            columns[name] = b.sub(a).stack()
    if not columns:
        return pd.DataFrame()
    table = pd.DataFrame(columns)
    table.index.names = ["sender", "receiver"]
    values = table.to_numpy(dtype=float)
    table["mean"] = np.nanmean(values, axis=1)
    table["same_sign"] = [int(max((row > 0).sum(), (row < 0).sum())) for row in values]
    table["runs"] = np.isfinite(values).sum(axis=1)
    return table.reset_index().sort_values("mean", key=np.abs, ascending=False, kind="stable").reset_index(drop=True)


def sections_per_condition(flows: pd.DataFrame) -> pd.Series:
    """How many test sections each condition has in a run's flow table (empty without ``section`` and ``condition``)."""
    if not len(flows) or not {"section", "condition"} <= set(flows.columns):
        return pd.Series(dtype=int)
    return flows.dropna(subset=["condition"]).drop_duplicates("section").groupby("condition")["section"].count()


def condition_by_section(flows: pd.DataFrame, *, kind: str = "state", region: str = "context") -> pd.DataFrame:
    """One run, sections as replicates: per sender and receiver the flow in each section, ``diff`` (the later condition name's mean minus the earlier's),
    ``separated`` (every section of one condition beyond every section of the other) and ``p`` (exact two-sided Mann-Whitney over sections; 3 against 3
    sections cannot go below 0.1). Empty unless both conditions have at least two sections: one section per condition is one donor pair, read over runs
    by :func:`condition_over_runs` instead.
    """
    counts = sections_per_condition(flows)
    if len(counts) != 2 or counts.min() < 2:
        return pd.DataFrame()
    first, second = sorted(counts.index)
    columns = {}
    for (condition, section), part in flows.dropna(subset=["condition"]).groupby(["condition", "section"], observed=True):
        matrix = flow_matrix(part, kind=kind, region=region)
        if len(matrix):
            columns[(str(condition), str(section))] = matrix.stack()
    if not columns:
        return pd.DataFrame()
    table = pd.DataFrame(columns)
    rows = []
    for (sender, receiver), values in table.iterrows():
        a = values[first].to_numpy(dtype=float) if first in values.index.get_level_values(0) else np.zeros(0)
        b = values[second].to_numpy(dtype=float) if second in values.index.get_level_values(0) else np.zeros(0)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 2 or len(b) < 2:
            continue
        rows.append(
            {
                "sender": sender,
                "receiver": receiver,
                first: float(a.mean()),
                second: float(b.mean()),
                "diff": float(b.mean() - a.mean()),
                "separated": bool(b.min() > a.max() or b.max() < a.min()),
                "p": float(mannwhitneyu(b, a, alternative="two-sided", method="exact").pvalue),
                "n_sections": f"{len(a)}/{len(b)}",
            }
        )
    return pd.DataFrame(rows).sort_values("diff", key=np.abs, ascending=False, kind="stable").reset_index(drop=True) if rows else pd.DataFrame()


def gene_flows_over_runs(run_dirs: list[Path]) -> pd.DataFrame:
    """(sender gene, target, receiver type) in the gene-flow top list of trained runs: the drop per run, in how many runs it is listed, and the twins' drop where
    the twin lists it. Built from each run's top 1000 (``gene_top``), so a missing value means not in that run's top, not zero.
    """
    trained = [r for r in run_dirs if not r.name.endswith(UNTRAINED)]
    keys = ["sender", "target", "receiver"]
    columns = {}
    for run in run_dirs:
        top = _read(run / "tables", "gene_top")
        if len(top):
            columns[run.name] = top.drop_duplicates(keys).set_index(keys)["delta_cos"]
    names = [r.name for r in trained if r.name in columns]
    if len(names) < 2:
        return pd.DataFrame()
    table = pd.DataFrame({n: columns[n] for n in names})
    table["listed"] = table[names].notna().sum(axis=1)
    table["mean"] = table[names].mean(axis=1)
    for name in sorted(n for n in columns if n.endswith(UNTRAINED)):
        table[name] = columns[name].reindex(table.index)
    return table.reset_index().sort_values(["listed", "mean"], ascending=False, kind="stable").reset_index(drop=True)


def programmes_over_runs(run_dirs: list[Path]) -> pd.DataFrame:
    """Programmes matched to the first trained run's by their loadings: is a programme found again, and does its condition effect repeat?"""
    runs = [
        r for r in run_dirs if not r.name.endswith(UNTRAINED) and (r / "tables" / "programme_loadings.npy").exists()
    ]
    if len(runs) < 2:
        return pd.DataFrame()
    reference = np.load(runs[0] / "tables" / "programme_loadings.npy")
    cards = _read(runs[0] / "tables", "programme_cards").set_index("programme")
    rows = [
        {
            "programme": int(p),
            "class": cards.loc[p, "class"],
            "top_genes": " ".join(str(cards.loc[p, "top_genes"]).split()[:5]),
            runs[0].name: cards.loc[p].get("condition_effect", np.nan),
        }
        for p in cards.index
    ]
    table = pd.DataFrame(rows).set_index("programme")
    for run in runs[1:]:
        other = np.load(run / "tables" / "programme_loadings.npy")
        if other.shape[0] != reference.shape[0]:
            continue
        matched = match_programmes(reference, other).set_index("a")
        theirs = _read(run / "tables", "programme_cards").set_index("programme")
        table[f"cosine_{run.name}"] = matched["cosine"]
        effect = (
            matched["b"].map(theirs["condition_effect"]) * matched["sign"] if "condition_effect" in theirs else np.nan
        )
        table[run.name] = effect
    return table.reset_index()


def _class_counts(cards: pd.DataFrame) -> str:
    if not len(cards):
        return ""
    return ", ".join(f"{n} {name}" for name, n in cards["class"].value_counts().items())


def influence_over_runs(run_dirs: list[Path]) -> list[str]:
    """Summary lines: per trained run the influence read (against its twin), then how stable the learned near influence's ranking is over runs."""
    from interscale.evaluation.ranking import rank_stability

    rows, learned = [], {}
    for run in run_dirs:
        table = _read(run / "tables", "gene_influence")
        if not len(table) or run.name.endswith(UNTRAINED):
            continue
        twin = twin_of(run)
        theirs = _read(twin / "tables", "gene_influence") if twin is not None else pd.DataFrame()
        found = influence_read(table.set_index("gene"), theirs.set_index("gene") if len(theirs) else None)
        learned[run.name] = found["genes"]["learned_near"]
        corr = found["correlations"].set_index(["influence", "spatial"])
        top = found["top"].set_index("influence") if len(found["top"]) else pd.DataFrame()
        sep = found["separation"]
        rows.append(
            {
                "run": run.name,
                "twin": twin.name if twin is not None else "",
                "near_vs_local_I": corr["within_expression"].get(("learned_near", "moran_local"), np.nan),
                "far_vs_global_I": corr["within_expression"].get(("learned_far", "moran_global"), np.nan),
                "svg_top_near": top["observed"].get("learned_near", np.nan) if len(top) else np.nan,
                "expected": top["expected"].get("learned_near", np.nan) if len(top) else np.nan,
                "p_near": top["p"].get("learned_near", np.nan) if len(top) else np.nan,
                "svg_top_far": top["observed"].get("learned_far", np.nan) if len(top) else np.nan,
                "p_far": top["p"].get("learned_far", np.nan) if len(top) else np.nan,
                "lean_global_minus_local": sep["lean_far_global"] - sep["lean_far_local"] if sep else np.nan,
                "p_separation": sep.get("p", np.nan) if sep else np.nan,
            }
        )
    if not rows:
        return []
    out = [
        "## Which genes matter, and are they the spatially variable ones",
        "",
        "Per run (each against its twin): the Spearman of learned influence with Moran's I within type, inside expression deciles; the share of the 20 most influential "
        "genes that are spatially variable against genes of the same expression (`expected`, `p`); and whether global genes lean more to far influence than local "
        "ones do. One measure of relevance among the others in this summary.",
        "",
        md_table(pd.DataFrame(rows)),
        "",
    ]
    scores = pd.DataFrame(learned)
    if scores.shape[1] >= 2:
        pairs, genes = rank_stability(scores, top_k=TOP_GENES)
        always = genes[genes["top_k_share"] == 1.0].sort_values("mean_rank").index.tolist()
        out += [
            f"Stability of the learned near influence over runs: rank-biased overlap {pairs['rbo'].mean():.2f}, Spearman {pairs['spearman'].mean():.2f}, top-{TOP_GENES} Jaccard "
            f"{pairs['jaccard'].mean():.2f} (means over {len(pairs)} pairs of runs). In the top {TOP_GENES} of every run: {', '.join(always) if always else 'none'}.",
            "",
            md_table(pairs),
            "",
        ]
    return out


def write_summary(results) -> Path:
    """``<results>/summary.md``: what repeats over runs, against the untrained twins. Also redraws every run's report."""
    results = Path(results)
    run_dirs = _runs(results)
    for run in run_dirs:
        write_run_report(run)
    frames = {r.name: _read(r / "tables", "flows") for r in run_dirs}
    out = [
        "# From the model to biology: over runs",
        "",
        f"Runs: {', '.join(r.name for r in run_dirs) or 'none'}. Each has its `runs/<run>/report.md`.",
        "",
    ]

    rows = []
    for run in run_dirs:
        use = _read(run / "tables", "context_use")
        part = use[use["region"] == "context"] if len(use) else use
        lr = lr_row(run / "tables")
        planted = planted_against_twin(run)
        rows.append(
            {
                "run": run.name,
                "context_state": part["state"].mean() if len(part) else np.nan,
                "context_arrangement": part["arrangement"].mean() if len(part) else np.nan,
                "far_state": use[use["region"] == "far"]["state"].mean() if len(use) else np.nan,
                "lr_pairs": int(lr["n_pairs"]) if lr else 0,
                "lr_auc": lr.get("auc", np.nan),
                "lr_null": lr.get("null_mean", np.nan),
                "lr_p": lr.get("p", np.nan),
                "planted_first": planted.get("first", np.nan),
                "planted_twin": planted.get("twin_first", np.nan),
                "planted_beyond_twin": planted.get("beyond_twin", np.nan),
                "top_gene_flow": _read(run / "tables", "gene_top")["delta_cos"].max()
                if len(_read(run / "tables", "gene_top"))
                else np.nan,
                "programmes": _class_counts(_read(run / "tables", "programme_cards")),
            }
        )
    out += [
        "## Does each model use other cells, and do known pairs stand out?",
        "",
        "Context columns: the drop in centred cosine within the receiver type, mean over receiver types (positive: the prediction needs those cells). **Every table below fills whatever the model is, so read each number against the `_untrained` twin's row**: a twin's prediction moves too, because its weights read their "
        "inputs at random, and its programmes get pathways and classes as well. `top_gene_flow`: the largest gene-level drop (a target's correlation within its type, at most 2). `lr_p`: known pairs against matched targets, allowing for pairs that share a gene (before 2026-11-08 every pair counted as independent, and p came out too small). "
        "`planted_first`, on planted data: how often a planted sender ranks first, the check of the readout; `planted_twin` the same for the untrained twin, which can rank them first too, and "
        "`planted_beyond_twin` first here but not in the twin: the bar is the twin, not chance.",
        "",
        md_table(pd.DataFrame(rows)),
        "",
    ]

    figures = results / "figures"
    figures.mkdir(exist_ok=True)
    panels, marks = [], {}
    for kind in ("state", "identity"):
        table = flows_over_runs(frames, kind=kind)
        if not len(table):
            continue
        runs = [c for c in table.columns if c not in {"sender", "receiver", "mean", "sd", "same_sign", "runs"}]
        twins = [flow_matrix(f, kind=kind) for n, f in frames.items() if n.endswith(UNTRAINED) and len(f)]
        bar = max((float(np.nanmax(np.abs(m.to_numpy()))) for m in twins if len(m)), default=0.0)
        indexed = table.set_index(["sender", "receiver"])
        name = f"{kind}: mean over {len(runs)} runs"
        panels.append((name, indexed["mean"].unstack("receiver")))
        marks[name] = ((indexed["same_sign"] == indexed["runs"]) & (indexed["runs"] >= 2) & (indexed["mean"].abs() > bar)).unstack("receiver")
    path = matrix_figure(
        panels,
        figures / "flows_over_runs.png",
        title="What the prediction loses, mean over runs",
        note="Outlined: one sign in every run and beyond the largest flow of any untrained twin. Row: the cells changed; column: the cells scored.",
        marks=marks,
    )
    if path:
        out += ["![flows over runs](figures/flows_over_runs.png)", ""]
    for kind, label in (("state", "state (types stay in place)"), ("identity", "identity, co-location taken off")):
        table = flows_over_runs(frames, kind=kind)
        if len(table):
            repeated = table[(table["same_sign"] == table["runs"]) & (table["runs"] >= 2) & (table["mean"] > 0)]
            out += [
                f"## Flows that repeat over runs: {label}",
                "",
                f"{len(repeated)} of {len(table)} pairs have one sign in every run and a positive mean. The largest:",
                "",
                md_table(repeated.head(TABLE_ROWS)),
                "",
            ]
            twins = {n: f for n, f in frames.items() if n.endswith(UNTRAINED) and len(f)}
            for name, flows in twins.items():
                matrix = flow_matrix(flows, kind=kind)
                if len(matrix):
                    out += [
                        f"The twin {name}: largest {np.nanmax(matrix.to_numpy()):.4g}, 95th percentile {np.nanquantile(matrix.to_numpy(), 0.95):.4g}. A flow below that is no more than an untrained model shows.",
                        "",
                    ]
    trained = {n: f for n, f in frames.items() if not n.endswith(UNTRAINED) and len(f)}
    replicated = {n: f for n, f in trained.items() if len(sections_per_condition(f)) == 2 and sections_per_condition(f).min() >= 2}
    paired = {n: f for n, f in trained.items() if n not in replicated}
    condition_panels, condition_marks = [], {}
    for kind in ("state", "identity"):
        table = condition_over_runs(paired, kind=kind)
        if len(table):
            indexed = table.set_index(["sender", "receiver"])
            name = f"{kind}: later minus earlier condition, mean over {int(indexed['runs'].max())} runs (one donor pair each; runs may share it)"
            condition_panels.append((name, indexed["mean"].unstack("receiver")))
            condition_marks[name] = ((indexed["same_sign"] == indexed["runs"]) & (indexed["runs"] >= 2)).unstack("receiver")
            consistent = table[(table["same_sign"] == table["runs"]) & (table["runs"] >= 2)]
            n_runs = int(table["runs"].max())
            out += [
                f"## Condition: later minus earlier condition name, {kind} flows, one donor pair per run",
                "",
                f"{len(consistent)} of {len(table)} pairs keep their sign over all runs. With {n_runs} independent donor pairs that alone happens for {2 * 0.5**n_runs:.0%} by chance "
                "(runs that share a test slide are not independent).",
                "",
                md_table(consistent.head(TABLE_ROWS)),
                "",
            ]
        per_run = {n: condition_by_section(f, kind=kind) for n, f in replicated.items()}
        per_run = {n: t for n, t in per_run.items() if len(t)}
        if per_run:
            keys = ["sender", "receiver"]
            diffs = pd.DataFrame({n: t.set_index(keys)["diff"] for n, t in per_run.items()})
            separated = pd.DataFrame({n: t.set_index(keys)["separated"] for n, t in per_run.items()}).fillna(False).astype(bool)
            diffs["separated_in"] = separated.sum(axis=1)
            diffs["same_sign"] = [int(max((r > 0).sum(), (r < 0).sum())) for r in diffs[list(per_run)].to_numpy(dtype=float)]
            diffs["mean"] = diffs[list(per_run)].mean(axis=1)
            diffs = diffs.reset_index().sort_values(["separated_in", "mean"], ascending=[False, False], key=lambda c: c.abs() if c.name == "mean" else c, kind="stable")
            one = next(iter(per_run.values()))
            conditions = [c for c in one.columns if c not in {"sender", "receiver", "diff", "separated", "p", "n_sections"}]
            smallest = float(one["p"].min()) if len(one) else np.nan
            indexed = diffs.set_index(["sender", "receiver"])
            name = f"{kind}: {conditions[-1]} minus {conditions[0]}, mean over {len(per_run)} runs"
            condition_panels.append((name, indexed["mean"].unstack("receiver")))
            condition_marks[name] = (indexed["separated_in"] == len(per_run)).unstack("receiver")
            counts = ", ".join(f"{n}: {int(t['separated'].sum())}" for n, t in per_run.items())
            out += [
                f"## Condition: {conditions[-1]} minus {conditions[0]}, {kind} flows, sections as replicates",
                "",
                f"Each run reads {one['n_sections'].iloc[0]} test sections ({conditions[0]}/{conditions[-1]}), each section one donor. `separated`: every section of one condition lies beyond every section of the other, "
                f"the strongest a rank test can say with so few sections (exact two-sided p = {smallest:.2g} at best); by chance that happens for about {smallest:.0%} of the {len(one)} pairs in one run. "
                f"Pairs separated per run: {counts}. **The runs share their test sections, so they share their donors**: agreement over runs says the model reads the same difference, "
                "not that it holds in other donors.",
                "",
                md_table(diffs.head(TABLE_ROWS)),
                "",
            ]
    path = matrix_figure(
        condition_panels,
        figures / "condition_over_runs.png",
        title="Condition differences of the flows",
        note="Outlined: with one donor pair per run, one sign in every run; with sections as replicates, every section of one condition beyond the other's in every run.",
        marks=condition_marks,
    )
    if path:
        out += ["![condition over runs](figures/condition_over_runs.png)", ""]
    genes = gene_flows_over_runs(run_dirs)
    if len(genes):
        n_trained = int(sum(not r.name.endswith(UNTRAINED) for r in run_dirs if len(_read(r / "tables", "gene_top"))))
        everywhere = genes[genes["listed"] == n_trained]
        out += [
            "## Gene flows that repeat over runs",
            "",
            f"(sender gene, target, receiver type) triples in the top 1000 of every trained run: {len(everywhere)} of {n_trained} runs; the most repeated below (`listed`: in how many runs' top). "
            "A twin's column is filled where the twin lists the triple too: a triple that a random model also moves is a property of the input (a high-variance sender, a spillover target), "
            "not of training.",
            "",
            md_table(genes.head(TABLE_ROWS)),
            "",
        ]
    out += influence_over_runs(run_dirs)
    programmes = programmes_over_runs(run_dirs)
    if len(programmes):
        out += [
            "## Programmes found again",
            "",
            "Each programme of the first run matched to the others' by loadings (`cosine_*`: 1 is the same programme); a run's own column is the condition effect in that run, signed to the first run's direction.",
            "",
            md_table(programmes),
            "",
        ]
    path = results / "summary.md"
    path.write_text("\n".join(out) + "\n")
    return path


__all__ = ["EVERYONE", "md_table", "write_run_report", "write_summary", "write_tables"]
