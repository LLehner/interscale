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
    read_parts,
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


def _group_depth(shift=0.0):
    """``score_heads``' per-group, per-depth table for one arm: groups A and noise, a model head and two references."""
    rows = []
    for group, base in (("A", 0.4), ("noise", 0.0)):
        for k, depth in enumerate(["all", *BINS]):
            rows.append(
                {
                    "group": group,
                    "depth_um": depth,
                    "n_genes": 3,
                    "n_hidden": 30 - k,
                    "global_cosine": base + shift,
                    "blend_global_cosine": base + shift + 0.05,
                    "ref_slide_mean_cosine": base - 0.2,
                    "ref_kernel_400um_cosine": base - 0.1 * k,
                }
            )
    return pd.DataFrame(rows)


def test_group_scores_by_depth_compact_to_rows_per_group_and_head(tmp_path):
    compact = compact_tables({"heads_by_group_depth": _group_depth()})["heads_group_by_depth"]
    assert list(compact.columns) == ["all", *BINS]
    heads = ["n_hidden", "global", "blend_global", "ref_slide_mean", "ref_kernel_400um"]
    assert list(compact.index) == [(g, h) for g in ("A", "noise") for h in heads]
    assert compact.loc[("A", "n_hidden"), BINS[1]] == 28
    path = tmp_path / "results.md"
    append_sections(path, "a", {"heads_group_by_depth": compact})
    back = read_sections(path)[("a", "heads_group_by_depth")]
    np.testing.assert_allclose(back.to_numpy(dtype=float), compact.to_numpy(dtype=float))
    assert list(back.index.names) == ["group", "head"]


def test_the_summary_shows_each_group_by_depth_with_the_best_reference_once(tmp_path):
    path = tmp_path / "results.md"
    append_sections(path, "a", compact_tables(_raw() | {"heads_by_group_depth": _group_depth(0.0)}))
    append_sections(path, "b", compact_tables(_raw() | {"heads_by_group_depth": _group_depth(0.3)}))
    tables = dict(summary_tables(read_sections(path), ["a", "b"]))
    table = next(
        t for title, t in tables.items() if title.startswith("centred cosine per gene group on hidden spots by depth")
    )
    assert list(table.index) == [
        (g, arm, head)
        for g in ("A", "noise")
        for arm, head in (
            ("count", "hidden entries"),
            ("a", "global"),
            ("a", "blend_global"),
            ("b", "global"),
            ("b", "blend_global"),
            ("reference", "best"),
            ("reference", "slide_mean"),
        )
    ]
    best = table.loc[("A", "reference", "best")]
    assert best["all"] == pytest.approx(0.4) and best[BINS[2]] == pytest.approx(0.2)  # the kernel, then the mean
    assert table.loc[("noise", "b", "global"), "all"] == pytest.approx(0.3)

    kept = dict(summary_tables(read_sections(path), ["a", "b"], groups=["noise"]))
    table = next(
        t for title, t in kept.items() if title.startswith("centred cosine per gene group on hidden spots by depth")
    )
    assert set(table.index.get_level_values("group")) == {"noise"}
    pooled = next(t for title, t in kept.items() if title == "centred cosine per gene group on hidden spots")
    assert list(pooled.columns) == ["noise"]


def test_the_summary_reads_the_gain_over_smoother_and_position_together(tmp_path):
    raw = _raw()
    extra = []
    for depth in ("all", BINS[0]):
        for feature_set, value in (("position", 0.65), ("smoother+position", 0.75), ("smoother+position+global", 0.8)):
            extra.append(
                {
                    "target": "region",
                    "feature_set": feature_set,
                    "metric": "balanced_accuracy",
                    "value": value,
                    "depth_um": depth,
                    "n": 9,
                }
            )
    raw["context_labels"] = pd.concat([raw["context_labels"], pd.DataFrame(extra)], ignore_index=True)
    path = tmp_path / "results.md"
    append_sections(path, "a", compact_tables(raw))
    labels = next(t for title, t in summary_tables(read_sections(path), ["a"]) if title.startswith("label 'region'"))
    references = [row for arm, row in labels.index if arm == "reference"]
    assert references == ["smoother", "position", "smoother+position", "majority"]
    assert labels.loc[("a", "global: gain over smoother+position"), "all"] == pytest.approx(0.05)
    assert labels.loc[("a", "global: gain over smoother"), "all"] == pytest.approx(0.1)
    assert ("a", "global: gain over position") not in labels.index  # position+global was not probed


def test_the_patch_flow_compacts_and_gets_summary_tables(tmp_path):
    summary = pd.DataFrame(
        [
            {"kind": kind, "metric": metric, "value": value}
            for kind, value in (("influence", 0.5), ("rollout", 0.2))
            for metric in ("shift_um", "kl_null", "kl_null_floor", "reach_um")
        ]
    )
    labels = pd.DataFrame(
        [
            {"kind": "rollout", "query_label": a, "key_label": b, "observed": 0.5, "null_mean": 0.25, "null_std": 0.05}
            for a in (0, 1)
            for b in (0, 1)
        ]
    )
    compact = compact_tables({"flow_summary": summary, "flow_labels": labels})
    assert list(compact["flow_summary"].columns) == ["reach_um", "kl_null", "kl_null_floor", "shift_um"]
    assert compact["flow_summary"].loc["rollout", "kl_null"] == pytest.approx(0.2)
    assert compact["flow_labels"].loc[("rollout", "log2_ratio", 1), 0] == pytest.approx(1.0)
    assert compact["flow_labels"].loc[("rollout", "z", 0), 1] == pytest.approx(5.0)
    path = tmp_path / "results.md"
    append_sections(path, "base", compact)
    append_sections(path, "topk", compact)
    tables = dict(summary_tables(read_sections(path), ["base", "topk"]))
    flow = next(t for title, t in tables.items() if title.startswith("patch flow against"))
    assert list(flow.index) == [(arm, kind) for arm in ("base", "topk") for kind in ("influence", "rollout")]


def test_the_response_influence_compacts_and_gets_a_summary_table(tmp_path):
    edges = pd.DataFrame(
        [
            {"head": "global", "range_class": rc, "program": prog, "metric": "matched_rank", "value": v, "n_pairs": 4}
            for rc, prog, v in (
                ("short", "S1>R1", 0.9),
                ("short", "any", 0.9),
                ("sparse", "M12>MC16", 0.6),
                ("sparse", "any", 0.6),
            )
        ]
    )
    profile = pd.DataFrame(
        [
            {"head": "global", "range_class": "short", "distance_um": d, "lift": 2.0, "n_queries": 3}
            for d in ("(0.0, 300.0]", "(300.0, 500.0]")
        ]
    )
    compact = compact_tables({"response_edges": edges, "response_profile": profile})
    assert compact["response_edges"].loc[("global", "sparse", "M12>MC16"), "matched_rank"] == pytest.approx(0.6)
    assert list(compact["response_profile"].columns) == ["(0.0, 300.0]", "(300.0, 500.0]"]
    path = tmp_path / "results.md"
    append_sections(path, "lin_rope", compact)
    table = next(
        t for title, t in summary_tables(read_sections(path), ["lin_rope"]) if title.startswith("programme masking")
    )
    assert list(table.columns) == ["short", "sparse"] and table.loc[("lin_rope", "global"), "short"] == pytest.approx(
        0.9
    )


# --------------------------------------------------------------------------- seeds


def _edges(value):
    return pd.DataFrame(
        [
            {"head": "global", "range_class": rc, "program": "any", "metric": "matched_rank", "value": value + d, "n_pairs": 4}
            for rc, d in (("short", 0.0), ("sparse", 0.1))
        ]
    )


def _seeded(tmp_path, runs):
    path = tmp_path / "results.md"
    for arm, shift in runs:
        tables = _raw(shift=shift, attention=False) | {"response_edges": _edges(0.5 + shift)}
        append_sections(path, arm, compact_tables(tables))
    return path


def test_arms_at_several_seeds_are_one_row_of_means_with_the_spread_beside_it(tmp_path):
    from interscale.evaluation.report import seed_groups

    runs = [("a_s1", 0.0), ("a_s2", 0.1), ("a_s3", 0.2), ("b_s1", 0.5), ("a_s1_untrained", -0.3)]
    path = _seeded(tmp_path, runs)
    arms = [arm for arm, _ in runs]
    assert seed_groups(arms) == {"a": ["a_s1", "a_s2", "a_s3"]}  # b ran once; a twin is not a seed

    tables = dict(summary_tables(read_sections(path), arms))
    heads = next(t for title, t in tables.items() if title.startswith("centred cosine on hidden spots by depth (µm"))
    rows = [r for r in heads.index if r[0] not in ("count", "reference")]
    assert rows == [("a [3 seeds]", "global"), ("b_s1", "global"), ("a_s1_untrained", "global")]
    # global_cosine_centred is 0.1 * depth_index + shift: the mean of the shifts 0, 0.1, 0.2 is 0.1.
    assert heads.loc[("a [3 seeds]", "global"), BINS[1]] == pytest.approx(0.1 + 0.1)
    assert heads.loc[("b_s1", "global"), BINS[1]] == pytest.approx(0.1 + 0.5)

    spread = next(t for title, t in tables.items() if title.startswith("centred cosine on hidden spots by depth: standard"))
    assert list(spread.index) == [("a [3 seeds]", "global")]
    assert spread.loc[("a [3 seeds]", "global"), BINS[1]] == pytest.approx(0.1)  # ddof=1 over 0, 0.1, 0.2
    runs_table = next(t for title, t in tables.items() if title.endswith("every run"))
    assert [r[0] for r in runs_table.index] == ["a_s1", "a_s2", "a_s3"]
    assert runs_table.loc[("a_s3", "global"), "all"] == pytest.approx(0.1 + 0.2)  # heads_cells: 0.1 + shift

    ranks = next(t for title, t in tables.items() if title.startswith("programme masking, sender's rank: standard"))
    assert ranks.loc[("a [3 seeds]", "global"), "short"] == pytest.approx(0.1)


def test_the_rewrite_keeps_every_run_in_its_sections_and_the_note_names_the_seed_groups(tmp_path):
    path = _seeded(tmp_path, [("a_s1", 0.0), ("a_s2", 0.2)])
    summary = write_report(path, arms=["a_s1", "a_s2"])
    text = path.read_text()
    assert "## a_s1 | heads_cosine_by_depth" in text and "## a_s2 | heads_cosine_by_depth" in text
    assert "a [2 seeds]" in summary and "seeds averaged: a [2 seeds]" in summary


def test_without_two_seeds_nothing_changes(tmp_path):
    path = _seeded(tmp_path, [("a_s1", 0.0), ("b_s1", 0.2)])
    tables = summary_tables(read_sections(path), ["a_s1", "b_s1"])
    assert not any("seeds" in title for title, _ in tables)
    heads = next(t for title, t in tables if title.startswith("centred cosine on hidden spots by depth (µm"))
    assert ("a_s1", "global") in heads.index and ("b_s1", "global") in heads.index


def test_tables_that_do_not_line_up_are_averaged_where_they_overlap():
    from interscale.evaluation.report import _mean_sd

    one = pd.DataFrame({"v": [1.0, 3.0]}, index=pd.Index(["x", "y"], name="k"))
    two = pd.DataFrame({"v": [3.0, 5.0]}, index=pd.Index(["y", "z"], name="k"))
    mean, sd = _mean_sd([one, two])
    assert mean.loc["x", "v"] == 1.0 and mean.loc["y", "v"] == 3.0 and mean.loc["z", "v"] == 5.0  # y: (3 + 3) / 2
    assert np.isnan(sd.loc["x", "v"]) and sd.loc["y", "v"] == 0.0  # one run has no spread


# --------------------------------------------------------------------------- parts and timing


def test_per_arm_files_merge_into_the_report_and_the_newest_section_wins(tmp_path):
    import os

    parts = tmp_path / "sections"
    parts.mkdir()
    append_sections(parts / "a.md", "a", compact_tables(_raw(shift=0.0, attention=False)))
    append_sections(parts / "b.md", "b", compact_tables(_raw(shift=0.2, attention=False)))
    append_sections(parts / "a_older.md", "a", compact_tables(_raw(shift=0.7, attention=False)))  # an earlier run of a
    path = tmp_path / "results.md"
    append_sections(path, "a", compact_tables(_raw(shift=0.9, attention=False)))  # an even earlier one
    os.utime(parts / "a_older.md", (1.0e9 - 10, 1.0e9 - 10))
    os.utime(parts / "a.md", (1.0e9, 1.0e9))
    os.utime(parts / "b.md", (1.0e9 + 10, 1.0e9 + 10))
    summary = write_report(path, arms=["a", "b"], parts=parts)
    found = read_sections(path)
    assert {arm for arm, _ in found} == {"a", "b"}
    # The newest file wins: the part a.md (0.1), not the older part (0.8) or the results file's own section (1.0).
    assert found[("a", "heads_cosine_by_depth")].loc["all", "global"] == pytest.approx(0.1)
    assert "global" in summary
    assert read_parts(tmp_path / "nothing", {("x", "y"): 1}) == {("x", "y"): 1}  # a missing folder changes nothing


def test_the_evaluation_time_is_a_table_of_seconds_per_step(tmp_path):
    path = tmp_path / "results.md"
    for arm, scale in (("a", 1.0), ("b", 2.0)):
        append_sections(
            path, arm, compact_tables({"timing": pd.DataFrame({"step": ["heads", "flow"], "seconds": [10.0 * scale, 60.0 * scale]}).set_index("step")})
        )
    table = next(t for title, t in summary_tables(read_sections(path), ["a", "b"]) if title.startswith("evaluation: seconds"))
    assert table.loc["b", "flow"] == pytest.approx(120.0) and list(table.index) == ["a", "b"]
