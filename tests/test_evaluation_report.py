"""The results file of a sweep (`evaluation.report`).

Pinned here, on tables with the evaluation's schemas: each analysis compacts to one wide table
(depth groups in reading order, model heads before references, per-head attention only for the key
metrics); sections round-trip through the file with their index, interval labels included; a later
section replaces an earlier one and missing ones do not matter; the rewrite puts the summary first
and keeps every section, and appending after it still works; the summary computes the gain over the
smoother and lists the references once.
"""

import numpy as np
import pandas as pd
import pytest

from interscale.evaluation.report import (
    append_sections,
    compact_tables,
    deepest,
    ordered,
    read_sections,
    summary_tables,
    write_report,
)

BINS = ["(0.0, 150.0]", "(150.0, 290.0]", "(500.0, inf]"]


def _raw(shift=0.0, attention=True):
    """The evaluation's tables for one arm, with planted values."""
    heads = {"n_cells": 40, "global_sce": 0.5 + shift, "global_cosine_centred": 0.1 + shift}
    heads |= {"ref_slide_mean_sce": 0.4, "ref_slide_mean_cosine_centred": 0.2, "global - local_sce": 0.0}
    tables = {
        "heads_cells": pd.DataFrame([heads]),
        "heads_by_depth": pd.DataFrame(
            [{"depth_um": b, **heads, "global_cosine_centred": 0.1 * k + shift} for k, b in enumerate(BINS)]
        ),
        "heads_per_gene": pd.DataFrame({"gene": ["a", "b"], "global_cosine": [0.1, 0.2]}),
        "heads_by_group": pd.DataFrame(
            {
                "group": ["A", "noise"],
                "n_genes": [3, 5],
                "global_cosine": [0.3, 0.0],
                "ref_slide_mean_cosine": [0.2, 0.0],
            }
        ),
        "training": pd.DataFrame({"metric": ["val_loss", "efficiency/train_steps_per_s"], "value": [0.07, 12.0]}),
    }
    rows = []
    for depth in ["all", *BINS[::-1]]:  # out of order on purpose
        for feature_set, value in (
            ("majority", 0.5),
            ("smoother", 0.6),
            ("global", 0.55 + shift),
            ("smoother+global", 0.7 + shift),
        ):
            for metric in ("balanced_accuracy", "f1_macro"):
                rows.append(
                    {
                        "target": "region",
                        "feature_set": feature_set,
                        "metric": metric,
                        "value": value,
                        "depth_um": depth,
                        "n": 9,
                    }
                )
    tables["context_labels"] = pd.DataFrame(rows)
    tables["source_distances"] = pd.DataFrame(
        [
            {"feature_set": f, "metric": "cosine_centred", "value": v, "depth_um": d, "n": 9}
            for d in ("all", BINS[0])
            for f, v in (("smoother", 0.3), ("smoother+global", 0.4))
        ]
    )
    tables["representations"] = pd.DataFrame(
        [
            {"query": q, "metric": "cka", "a": "global", "b": "smoother", "value": 0.9, "n_cells": 20}
            for q in ("visible", "hidden", BINS[0])
        ]
        + [{"query": "hidden", "metric": "participation_ratio", "a": "global", "b": "", "value": 1.5, "n_cells": 20}]
    )
    tables["erf_by_query"] = pd.DataFrame(
        [
            {"head": "global", "query": q, "metric": m, "value": 0.5, "n_queries": 4}
            for q in BINS
            for m in ("influence", "span_rel", "beyond_300um")
        ]
    )
    tables["erf_profile"] = pd.DataFrame(
        [{"head": "global", "query": q, "distance_um": "(0.0, 300.0]", "lift": 1.2, "n_queries": 4} for q in BINS]
    )
    tables["erf_edges"] = pd.DataFrame(
        {
            "head": ["global"] * 2,
            "range_class": ["short", "any"],
            "metric": "matched_rank",
            "value": [0.6, 0.55],
            "n_pairs": [5, 8],
        }
    )
    if attention:
        rows = []
        for layer in (0, 1):
            for head in ("h0", "h1", "mean"):
                for metric in ("visible_lift", "span_rel", "cls_mass", "beyond_300um"):
                    for query in ("visible", *BINS):
                        rows.append(
                            {
                                "layer": layer,
                                "head": head,
                                "query": query,
                                "metric": metric,
                                "value": 1.0 + layer,
                                "n_queries": 7,
                            }
                        )
            rows.append(
                {"layer": layer, "head": "mean", "query": "all", "metric": "token_cosine", "value": 0.9, "n_queries": 2}
            )
        tables["attention_by_query"] = pd.DataFrame(rows)
        tables["attention_profile"] = pd.DataFrame(
            [
                {"layer": l, "query": q, "distance_um": d, "lift": 1.0, "n_queries": 3}
                for l in (0, 1)
                for q in BINS
                for d in ("(300.0, 500.0]", "(0.0, 300.0]")
            ]
        )
        tables["attention_edges"] = pd.DataFrame(
            [
                {"layer": l, "range_class": "any", "metric": m, "value": v, "n_pairs": 6}
                for l in (0, 1)
                for m, v in (("coverage", 1.0), ("matched_rank", 0.5))
            ]
        )
    return tables


def test_each_analysis_compacts_to_one_wide_table_in_reading_order():
    compact = compact_tables(_raw() | {"extra": pd.DataFrame({"x": [1, 2]})})
    assert "heads_per_gene" not in compact and "heads_cells" not in compact and "extra" in compact
    cosine = compact["heads_cosine_by_depth"]
    assert list(cosine.index) == ["all", *BINS] and list(cosine.columns) == ["n_cells", "global", "ref_slide_mean"]
    assert cosine.loc["all", "global"] == pytest.approx(0.1)
    assert list(compact["heads_sce_by_depth"].columns) == ["n_cells", "global", "ref_slide_mean"]  # no "global - local"
    labels = compact["context_labels"]
    assert list(labels.columns) == ["all", *BINS]  # sorted by depth, f1 left out
    assert labels.loc[("region", "smoother+global"), "all"] == pytest.approx(0.7)
    attention = compact["attention_by_query"]
    assert set(attention.index.get_level_values("head")) == {"h0", "h1", "mean"}
    per_head = attention.xs("h0", level="head").index.get_level_values("metric")
    assert set(per_head) == {"visible_lift", "span_rel"}  # cls_mass and radii only for the mean
    assert list(attention.columns) == ["all", "visible", *BINS]
    assert list(compact["attention_profile"].columns) == ["(0.0, 300.0]", "(300.0, 500.0]"]
    assert list(compact["attention_edges"].columns) == ["coverage", "matched_rank", "n_pairs"]
    assert compact["representations"].loc["pr global", "hidden"] == pytest.approx(1.5)
    assert list(compact["erf_edges"].columns) == ["matched_rank", "n_pairs"]


def test_sections_round_trip_with_their_index(tmp_path):
    path = tmp_path / "results.md"
    compact = compact_tables(_raw())
    append_sections(path, "lin_rope", compact, title="global only")
    found = read_sections(path)
    assert list(found) == [("lin_rope", name) for name in compact]
    for name, table in compact.items():
        back = found[("lin_rope", name)]
        assert list(back.index.names) == [n if n is not None else "index" for n in table.index.names] or name == "extra"
        np.testing.assert_allclose(back.to_numpy(dtype=float), table.to_numpy(dtype=float), rtol=1e-4, equal_nan=True)
        assert list(back.columns) == [str(c) for c in table.columns]
    text = path.read_text()
    assert text.startswith("# global only\n") and text.count("# Sections") == 1
    assert "\n## lin_rope | context_labels\n" in text and '"(0.0, 150.0]"' in text  # quoted: the label holds a comma

    append_sections(path, "plain", {"extra": pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})})
    plain = read_sections(path)[("plain", "extra")]
    assert list(plain.columns) == ["x", "y"] and plain["y"].tolist() == ["a", "b"]


def test_a_later_section_wins_and_missing_ones_do_not_matter(tmp_path):
    path = tmp_path / "results.md"
    append_sections(path, "a", compact_tables(_raw(shift=0.0)))
    append_sections(path, "b", compact_tables({"heads_cells": _raw()["heads_cells"]}))
    append_sections(path, "a", {"heads_cosine_by_depth": compact_tables(_raw(shift=0.5))["heads_cosine_by_depth"]})
    found = read_sections(path)
    assert found[("a", "heads_cosine_by_depth")].loc["all", "global"] == pytest.approx(0.6)
    assert ("b", "context_labels") not in found
    titles = [title for title, _ in summary_tables(found, ["a", "b"])]
    assert any(t.startswith("label 'region'") for t in titles) and any(
        t.startswith("centred cosine on hidden") for t in titles
    )
    assert not any("masked pass" in t for t, _ in summary_tables(found, ["b"]))  # b has no attention: no such table


def test_the_rewrite_puts_the_summary_first_keeps_every_section_and_takes_appends(tmp_path):
    path = tmp_path / "results.md"
    append_sections(path, "lin_rope_untrained", compact_tables(_raw(shift=-0.1)))
    append_sections(path, "lin_rope", compact_tables(_raw(shift=0.1)))
    append_sections(path, "local_gcn", compact_tables(_raw(attention=False)))
    before = read_sections(path)
    summary = write_report(path, title="global only", arms=["lin_rope", "lin_rope_untrained", "local_gcn"])
    text = path.read_text()
    assert (
        text.index("# Summary")
        < text.index("### centred cosine")
        < text.index("# Sections")
        < text.index("## lin_rope |")
    )
    assert text.index("## lin_rope |") < text.index("## lin_rope_untrained |") < text.index("## local_gcn |")
    assert summary in text and "from 3 arms: lin_rope, lin_rope_untrained, local_gcn" in text
    after = read_sections(path)
    assert set(after) == set(before)
    for key in before:
        np.testing.assert_allclose(after[key].to_numpy(dtype=float), before[key].to_numpy(dtype=float), equal_nan=True)

    append_sections(path, "joint_rope", {"training": compact_tables(_raw())["training"]})
    assert ("joint_rope", "training") in read_sections(path)
    write_report(path, title="global only", arms=["lin_rope"])
    assert "from 4 arms: lin_rope, lin_rope_untrained, local_gcn, joint_rope" in path.read_text()


def test_the_summary_reads_the_gain_over_the_smoother_and_lists_references_once(tmp_path):
    path = tmp_path / "results.md"
    append_sections(path, "a", compact_tables(_raw(shift=0.0)))
    append_sections(path, "b", compact_tables(_raw(shift=0.2)))
    tables = dict(summary_tables(read_sections(path), ["a", "b"]))
    heads = next(t for title, t in tables.items() if title.startswith("centred cosine on hidden"))
    assert list(heads.index) == [
        ("count", "hidden spots"),
        ("a", "global"),
        ("b", "global"),
        ("reference", "slide_mean"),
    ]
    assert heads.loc[("count", "hidden spots"), "all"] == 40
    labels = next(t for title, t in tables.items() if title.startswith("label 'region'"))
    assert labels.loc[("reference", "smoother"), "all"] == pytest.approx(0.6)
    assert labels.loc[("reference", "majority"), "all"] == pytest.approx(0.5)
    assert labels.loc[("b", "global: gain over smoother"), "all"] == pytest.approx(0.3)
    attention = next(t for title, t in tables.items() if "attention on visible spots" in title)
    assert list(attention.index) == [("a", 0), ("a", 1), ("b", 0), ("b", 1)]
    assert attention.loc[("a", 1), BINS[0]] == pytest.approx(2.0)
    profile = next(
        t for title, t in tables.items() if title.startswith("attention of hidden spots at depth (500.0, inf]")
    )
    assert list(profile.columns) == ["(0.0, 300.0]", "(300.0, 500.0]"]


def test_staged_phases_pass_through_and_get_a_summary_table(tmp_path):
    path = tmp_path / "results.md"
    phases = pd.DataFrame({"drift_local": [0.3, 0.0], "drift_global": [0.0, 0.2], "val_global_loss": [0.9, 0.5]})
    append_sections(path, "l2g", compact_tables({"phases": phases.rename_axis("phase")}))
    append_sections(path, "joint", compact_tables(_raw()))  # no phases: no rows
    found = read_sections(path)
    assert found[("l2g", "phases")].loc[1, "drift_global"] == pytest.approx(0.2)
    table = next(t for title, t in summary_tables(found, ["l2g", "joint"]) if title.startswith("staged training"))
    assert list(table.index) == [("l2g", 0), ("l2g", 1)] and table.loc[("l2g", 1), "drift_local"] == 0.0


def test_depth_labels_are_ordered_by_their_lower_edge():
    assert ordered(["(500.0, inf]", "x", "visible", "(0.0, 150.0]", "all"]) == [
        "all",
        "visible",
        "(0.0, 150.0]",
        "(500.0, inf]",
        "x",
    ]
    assert deepest(["visible", "(0.0, 150.0]", "(290.0, 500.0]"]) == "(290.0, 500.0]"
    assert deepest(["visible"]) is None
