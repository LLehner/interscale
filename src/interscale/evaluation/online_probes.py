"""Downstream probes run *during* training, as a Lightning callback.

The reconstruction loss answers "how well is expression copied back". It does not answer the
question this architecture exists to ask -- **which component learned what**. A dual-decoder run
already reports ``val_local_loss`` and ``val_global_loss`` separately, but those are two numbers
about the same task; they cannot say that the transformer holds a long-range interaction program
the GCN cannot see, because reconstructing ``int_long`` well is worth a tiny fraction of a loss
averaged over every gene.

So the same target is read out of the local embedding and out of the global one, with an
otherwise identical readout, and every number logged here is one half of a local-vs-global pair.
That pairing is the whole design: a probe score on its own is close to meaningless (it depends on
the embedding width, the readout, the class balance), while the *gap* between the two columns is
the attribution the dataset was built to test.

Protocol
--------
The readout is **fit on the train split and scored on the val split** -- the standard linear
probing protocol, not a cross-validation inside val. That costs one extra no-grad pass over each
split per probe epoch, which is why :data:`~interscale.config.probe_config` exposes
``probe.every_n_epochs``.

Three properties keep the measurement from disturbing what it measures:

* **No RNG is consumed.** The probe builds its own ``shuffle=False`` loaders over the datamodule's
  ``Data`` lists rather than iterating ``train_dataloader()``, whose shuffle would draw from the
  same torch generator the training batch order comes from -- so turning the probe on would
  reorder training batches and change the model. Dropout is off (``eval()``), masks are read from
  the ``Data`` objects rather than redrawn, and the subsample uses its own seeded numpy generator.
* **The encoder is not updated.** Everything runs under ``torch.no_grad()``, and the module's
  training flag is restored afterwards.
* **The corruption is the run's own.** ``_common_step`` masks its input exactly as train and val
  do, so the probe measures the representation the model actually operates on. ``batch.x`` is
  untouched by that -- ``tl.masking.apply_mask`` corrupts a clone -- which is what makes it a
  legitimate source of uncorrupted regression targets.

Both halves of a pair see identical rows in identical order, because the local embedding is
gathered by ``ViewOutput.padded_node_idx`` -- the same index that brings the global tokens into
cell order. Nothing here may assume the two are already aligned; see ``_step_output.py``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error, precision_score, r2_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader

from interscale.tl.geome_utils import label_codes

#: The representations a probe can read a target out of, and the ``ViewOutput`` field each comes
#: from. A model lacking a component contributes no columns for it rather than erroring, so one
#: probe block works unchanged for LocalModel, GlobalModel and CombinedModel.
EMBEDDINGS = ("local", "global")

#: Metrics reported per (target, embedding), by task. Precision and recall are macro-averaged:
#: the synthetic cell types are not equally frequent (``stroma`` is 20-60% of a niche), and a
#: micro average on an imbalanced problem mostly reports the majority class.
CLASSIFICATION_METRICS = ("precision", "recall")
#: ``mse`` is the headline; ``r2`` is logged beside it because MSE is in the units of the gene's
#: own variance, so a noise gene and an interaction gene cannot be compared on it. R2 is the one
#: that answers "is there any signal here at all" -- ~0 for a noise control, >0 for a program the
#: encoder found.
REGRESSION_METRICS = ("mse", "r2")


#: Prefix for every probe scalar. The slash is load-bearing: wandb groups panels into sections
#: by the text before the first "/", so this is what keeps the probe numbers out of the default
#: section where InterScale's own train_/val_ metrics live. Monitor strings match on the full
#: name, so "probe/celltype_precision_global" is still usable as optim.monitor.
PROBE_PREFIX = "probe"


def probe_metric_name(target: str, metric: str, embedding: str) -> str:
    """The scalar name a probe result is logged under.

    All probes are scored on the validation split, so the name carries no ``val_`` -- the
    prefix already says these are probe numbers and nothing else produces them.
    """
    return f"{PROBE_PREFIX}/{target}_{metric}_{embedding}"


@dataclass
class ProbeBatchFeatures:
    """Embeddings and targets collected over one split, all in the same cell order.

    Attributes
    ----------
    embeddings
        ``{"local": [N, E], "global": [N, E]}``, missing a key when the model has no such
        component.
    categorical
        ``{target_name: [N]}`` integer class codes.
    continuous
        ``{gene_name: [N]}`` uncorrupted expression.
    cell_masked
        ``[N]`` boolean, True where the cell was a reconstruction target this epoch.
    gene_masked
        ``{gene_name: [N]}`` boolean, True where *that gene of that cell* was blanked. Equal to
        :attr:`cell_masked` under ``mask_strategy="node"``, where a masked cell has every gene
        blanked; genuinely per-gene under ``"gene"``.
    """

    embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    categorical: dict[str, np.ndarray] = field(default_factory=dict)
    continuous: dict[str, np.ndarray] = field(default_factory=dict)
    cell_masked: np.ndarray | None = None
    gene_masked: dict[str, np.ndarray] = field(default_factory=dict)

    def n_cells(self) -> int:
        """Number of collected cells, or 0 when nothing was collected."""
        for block in self.embeddings.values():
            return int(block.shape[0])
        return 0

    def select(self, rows: np.ndarray) -> ProbeBatchFeatures:
        """Index every block by the same row selection.

        One helper for both the mask restriction and the subsample, because the failure mode
        they share -- indexing the embeddings but not the targets -- does not raise. It
        produces a probe that scores at chance, which reads like a real (negative) result.
        """
        return ProbeBatchFeatures(
            embeddings={k: v[rows] for k, v in self.embeddings.items()},
            categorical={k: v[rows] for k, v in self.categorical.items()},
            continuous={k: v[rows] for k, v in self.continuous.items()},
            cell_masked=None if self.cell_masked is None else self.cell_masked[rows],
            gene_masked={k: v[rows] for k, v in self.gene_masked.items()},
        )

    def subsample(self, max_cells: int, rng: np.random.Generator) -> ProbeBatchFeatures:
        """A seeded row subsample, or ``self`` when already small enough."""
        n = self.n_cells()
        if max_cells <= 0 or n <= max_cells:
            return self
        return self.select(rng.choice(n, size=max_cells, replace=False))


def _concat(chunks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate a list of per-batch dicts into one dict of full-split arrays."""
    if not chunks:
        return {}
    return {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in chunks[0]}


@torch.no_grad()
def collect_features(
    module,
    data_list,
    *,
    prediction_task: str,
    prediction_level: str,
    embeddings: tuple[str, ...],
    gene_index: dict[str, int],
    categorical_targets: tuple[str, ...],
    obs_targets: tuple[str, ...],
    mask_strategy: str,
    batch_size: int,
    device,
    corrupt: bool = True,
) -> ProbeBatchFeatures:
    """Run ``module`` over ``data_list`` and gather embeddings and probe targets.

    Parameters
    ----------
    module
        The :class:`~interscale.module.base.BaseModule` being trained.
    data_list
        The split's PyG ``Data`` objects, e.g. ``datamodule.train_data``. Iterated through a
        private ``shuffle=False`` loader -- see the module docstring for why the datamodule's
        own train loader must not be used.
    prediction_task, prediction_level
        Passed straight to ``module._common_step``.
    embeddings
        Which of :data:`EMBEDDINGS` to gather.
    gene_index
        ``{gene_name: column of batch.x}`` for the continuous targets.
    categorical_targets
        Names of optional annotations attached to the ``Data`` objects, e.g. ``("celltype",)``.
    obs_targets
        Names of the numeric obs columns carried in ``batch.probe_targets``, in the column order
        the entrypoint stacked them.
    mask_strategy
        ``cfg.dataset.mask_strategy``. Passed explicitly rather than sniffed from whether a
        ``gene_mask`` attribute happens to be present, for the reason ``tl/masking.py``
        documents: a stale attribute on a reused ``Data`` object would otherwise silently
        decide which entries the probe considers held out.
    batch_size
        Graphs per forward pass. Does not affect the values: one graph is one transformer
        sequence and attention never crosses a sequence, so a cell's embedding is the same at
        any batch size.
    device
        Device to move each batch to.
    corrupt
        True runs the model on the batch as training does, masked input and all -- required for
        gene targets, whose value would otherwise be sitting in the encoder's input. False
        blanks nothing, giving every cell an uncorrupted embedding, which is what a label probe
        wants: the label is not an input, so there is no leak and masking only destroys signal.

    Returns
    -------
    ProbeBatchFeatures
        Everything collected, in a single consistent cell order per split.
    """
    was_training = module.training
    module.eval()

    emb_chunks: list[dict[str, np.ndarray]] = []
    cat_chunks: list[dict[str, np.ndarray]] = []
    con_chunks: list[dict[str, np.ndarray]] = []
    mask_chunks: list[dict[str, np.ndarray]] = []
    gene_mask_chunks: list[dict[str, np.ndarray]] = []

    loader = DataLoader(dataset=data_list, shuffle=False, batch_size=batch_size, num_workers=0)

    try:
        for batch in loader:
            batch = batch.to(device)
            # Captured before the forward pass purely for readability: apply_mask corrupts a
            # clone, so this reference stays the uncorrupted expression either way.
            x_true = batch.x

            if not corrupt:
                # An all-False mask makes apply_mask a no-op (it assigns MASK_VALUE to an empty
                # selection), so the encoder sees the real expression. Assigned as a NEW tensor
                # rather than filled in place: the loader collates a fresh Batch per iteration
                # but its tensors can share storage with the source Data objects, and writing
                # through would destroy the training masks this split is using.
                batch.mask = torch.zeros_like(batch.mask)
                if getattr(batch, "gene_mask", None) is not None:
                    batch.gene_mask = torch.zeros_like(batch.gene_mask)

            out = module._common_step(batch, prediction_task, prediction_level)
            view = out.view

            # `padded_node_idx` is the bridge from batch node order into token order. A module
            # with no global component leaves it None, and then the local embedding is already
            # the row set we want, in order.
            idx = view.padded_node_idx
            if idx is None:
                if view.local_embedding is None:
                    raise RuntimeError("Probe found neither a padded_node_idx nor a local embedding to gather.")
                idx = torch.arange(view.local_embedding.shape[0], device=device)

            batch_emb: dict[str, np.ndarray] = {}
            if "local" in embeddings and view.local_embedding is not None:
                batch_emb["local"] = view.local_embedding[idx].detach().cpu().numpy()
            if "global" in embeddings and view.global_embedding is not None:
                batch_emb["global"] = view.tokens().detach().cpu().numpy()

            if not batch_emb:
                raise RuntimeError(
                    f"probe.embeddings={list(embeddings)} but this module produced none of them. "
                    "A LocalModel has no 'global' embedding and a GlobalModel no 'local' one."
                )

            batch_cat = {name: label_codes(batch, name)[idx].detach().cpu().numpy() for name in categorical_targets}
            batch_con = {gene: x_true[idx, col].detach().cpu().numpy() for gene, col in gene_index.items()}
            if obs_targets:
                targets = getattr(batch, "probe_targets", None)
                if targets is None:
                    raise AttributeError(
                        f"probe.regression_obs asks for {list(obs_targets)} but the batch carries no "
                        "'probe_targets'; set dataset.probe_obsm_key and stack those obs columns "
                        "into that obsm key before building the graphs."
                    )
                # Column order is probe.regression_obs, fixed by whoever built the obsm.
                for col, name in enumerate(obs_targets):
                    batch_con[name] = targets[idx, col].detach().cpu().numpy()

            # Which rows the model was actually asked about. `batch.mask` is the row-wise OR of
            # `gene_mask` under gene masking, so it means "this cell is a supervision target"
            # under both strategies -- see geome_dataloader._assign_random_mask.
            batch_cell_masked = batch.mask[idx].bool().detach().cpu().numpy()
            gene_mask = getattr(batch, "gene_mask", None) if mask_strategy == "gene" else None
            if gene_mask is None:
                batch_gene_masked = {gene: batch_cell_masked for gene in gene_index}
            else:
                batch_gene_masked = {
                    gene: gene_mask[idx, col].bool().detach().cpu().numpy() for gene, col in gene_index.items()
                }

            emb_chunks.append(batch_emb)
            cat_chunks.append(batch_cat)
            con_chunks.append(batch_con)
            mask_chunks.append({"cell": batch_cell_masked})
            gene_mask_chunks.append(batch_gene_masked)
    finally:
        module.train(was_training)

    return ProbeBatchFeatures(
        embeddings=_concat(emb_chunks),
        categorical=_concat(cat_chunks),
        continuous=_concat(con_chunks),
        cell_masked=_concat(mask_chunks).get("cell"),
        gene_masked=_concat(gene_mask_chunks),
    )


def score_classification(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int,
    max_iter: int,
) -> dict[str, float]:
    """Fit a standardised logistic readout on train, score macro precision/recall on val.

    ``class_weight="balanced"`` because a probe is asked whether the label is *present* in the
    representation, and an unweighted fit on an imbalanced problem answers the easier question
    of whether the majority class is predictable.

    Returns an empty dict when the training split carries fewer than two classes, which is not
    an error -- it is what a degenerate subsample looks like -- but has no precision to report.
    """
    if len(np.unique(y_train)) < 2:
        warnings.warn(
            f"probe: training split carries a single class ({np.unique(y_train)}); skipping.",
            RuntimeWarning,
            stacklevel=2,
        )
        return {}

    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=max_iter, class_weight="balanced", random_state=seed)),
        ]
    )
    with warnings.catch_warnings():
        # A probe that has not converged in max_iter is still a valid readout of what is
        # linearly present; the warning fires once per target per epoch and drowns the log.
        warnings.simplefilter("ignore")
        pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_val)

    # zero_division=0: a class the readout never predicts has undefined precision, and scoring
    # it 0 is the honest reading -- it contributes nothing recoverable to the macro average.
    return {
        "precision": float(precision_score(y_val, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_val, y_pred, average="macro", zero_division=0)),
    }


def score_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    alpha: float,
) -> dict[str, float]:
    """Fit a standardised ridge readout on train, score MSE and R2 on val.

    Both are reported because they answer different questions: MSE is in the gene's own units
    and so is the quantity the user asked for, while R2 normalises by the target's variance and
    is therefore the only one of the two that can be compared between a noise control and an
    interaction gene.
    """
    if float(np.var(y_train)) == 0.0:
        warnings.warn("probe: regression target is constant on train; skipping.", RuntimeWarning, stacklevel=2)
        return {}

    pipe = Pipeline([("scale", StandardScaler()), ("reg", Ridge(alpha=alpha))])
    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_val)

    scores = {"mse": float(mean_squared_error(y_val, y_pred))}
    # R2 is undefined against a constant val target; MSE still is, so report what exists.
    if float(np.var(y_val)) > 0.0:
        scores["r2"] = float(r2_score(y_val, y_pred))
    return scores


class OnlineProbeCallback(Callback):
    """Fit local-vs-global downstream probes at the end of selected validation epochs.

    Parameters
    ----------
    cfg
        The run's config; reads the ``probe`` block plus ``dataset.batch_size``.
    var_names
        ``adata.var_names``, used to turn ``probe.regression_genes`` into column indices of
        ``batch.x``.

    Notes
    -----
    Results are logged twice. The flat scalars from :func:`probe_metric_name` go through
    ``pl_module.log`` so they land in ``trainer.callback_metrics`` and are monitorable. When
    wandb is in use, each target additionally gets a **custom chart carrying local and global as
    two lines on one plot** -- the comparison is the point, and reading it off two separate
    panels with independently scaled axes is how a small but consistent gap gets missed.
    """

    def __init__(self, cfg, var_names):
        super().__init__()
        self._cfg = cfg
        self._probe_cfg = cfg.probe
        self._embeddings = tuple(cfg.probe.embeddings)
        self._categorical = tuple(cfg.probe.classification_targets)
        # Column order in batch.probe_targets is exactly this order -- whoever builds the obsm
        # must stack the columns the same way, which `build_probe_obsm` guarantees.
        self._obs_targets = tuple(cfg.probe.regression_obs)

        var_names = list(var_names)
        missing = [g for g in cfg.probe.regression_genes if g not in var_names]
        if missing:
            raise KeyError(
                f"probe.regression_genes names genes that are not in adata.var_names: {missing}. "
                f"The probe reads them as columns of batch.x, so a typo would silently probe the "
                f"wrong gene if this were not checked."
            )
        self._gene_index = {g: var_names.index(g) for g in cfg.probe.regression_genes}
        self._n_var = len(var_names)

        # Per-metric history, {chart_key: {series_name: [(epoch, value), ...]}}. Kept so each
        # wandb chart can be re-rendered with the full curve every time it is refreshed.
        self._history: dict[str, dict[str, list[tuple[int, float]]]] = {}
        # Probe rounds completed, which paces the chart refresh. Counted in rounds rather than
        # read off trainer.current_epoch because the post-fit trainer.validate() reports the
        # epoch training ended on, so an epoch-based cadence could skip the final render.
        self._rounds = 0

    # ---------------------------------------------------------------- Lightning hooks

    def on_validation_epoch_end(self, trainer, pl_module):
        """Run the probes, unless this is a sanity check or an off-cadence epoch."""
        if trainer.sanity_checking:
            return
        if trainer.current_epoch % int(self._probe_cfg.every_n_epochs) != 0:
            return

        datamodule = trainer.datamodule
        if datamodule is None or not getattr(datamodule, "setup_called", False):
            return

        results = self.run(pl_module.module, datamodule, device=pl_module.device)

        for name, value in results.items():
            # batch_size is only used by Lightning to weight an epoch average; these are already
            # epoch-level scalars, so on_epoch with a single value is what is wanted.
            pl_module.log(name, value, on_step=False, on_epoch=True, batch_size=int(self._cfg.dataset.batch_size))

        self._record(trainer, results)
        # trainer.validate() after fit restores the best checkpoint and revalidates, so this is
        # the run's final and most representative probe -- always chart it.
        self._publish_charts(final=trainer.state.fn != "fit")

    # ---------------------------------------------------------------- the probe itself

    def run(self, module, datamodule, *, device) -> dict[str, float]:
        """Collect both splits, fit every (target, embedding) readout, and return the scalars.

        Runs up to two passes per split, because the two kinds of target need opposite inputs:
        a CLEAN pass for labels (never model inputs, so masking only deletes signal) and a
        CORRUPTED pass for genes (model inputs, so an unmasked probe measures invertibility).
        A pass is skipped entirely when no target needs it.

        Separated from the hook so it can be called directly in a test, with no trainer.
        """
        # Checked before the passes, not after: a mismatch means every gene name resolves to the
        # wrong column, and there is no point paying for the encoder pass to find that out.
        if self._n_var != module.n_input:
            raise ValueError(
                f"probe was built from {self._n_var} var_names but batch.x has {module.n_input} "
                "columns, so a gene name would resolve to the wrong column."
            )

        collect_kwargs = {
            "prediction_task": self._cfg.dataset.prediction_task,
            "prediction_level": self._cfg.dataset.prediction_level,
            "embeddings": self._embeddings,
            "gene_index": self._gene_index,
            "categorical_targets": self._categorical,
            "obs_targets": self._obs_targets,
            "mask_strategy": self._cfg.dataset.mask_strategy,
            "batch_size": int(self._cfg.dataset.batch_size),
            "device": device,
        }

        results: dict[str, float] = {}
        wants_labels = bool(self._categorical or self._obs_targets)
        wants_genes = bool(self._gene_index)

        if wants_labels:
            clean_train = collect_features(module, datamodule.train_data, corrupt=False, **collect_kwargs)
            clean_val = collect_features(module, datamodule.val_data, corrupt=False, **collect_kwargs)
            results.update(self._score_labels(clean_train, clean_val))

        if wants_genes:
            train = collect_features(module, datamodule.train_data, corrupt=True, **collect_kwargs)
            val = collect_features(module, datamodule.val_data, corrupt=True, **collect_kwargs)
            results.update(self._score_genes(train, val))

        return results

    def _score_labels(self, train: ProbeBatchFeatures, val: ProbeBatchFeatures) -> dict[str, float]:
        """Categorical and numeric-obs targets, on every cell of the clean pass."""
        results: dict[str, float] = {}
        x_train, x_val = self._capped(train), self._capped(val)

        for target in self._categorical:
            for embedding in x_train.embeddings:
                scores = score_classification(
                    x_train.embeddings[embedding],
                    x_train.categorical[target],
                    x_val.embeddings[embedding],
                    x_val.categorical[target],
                    seed=int(self._probe_cfg.seed),
                    max_iter=int(self._probe_cfg.max_iter),
                )
                for metric, value in scores.items():
                    results[probe_metric_name(target, metric, embedding)] = value

        for target in self._obs_targets:
            for embedding in x_train.embeddings:
                scores = score_regression(
                    x_train.embeddings[embedding],
                    x_train.continuous[target],
                    x_val.embeddings[embedding],
                    x_val.continuous[target],
                    alpha=float(self._probe_cfg.ridge_alpha),
                )
                for metric, value in scores.items():
                    results[probe_metric_name(target, metric, embedding)] = value

        return results

    def _score_genes(self, train: ProbeBatchFeatures, val: ProbeBatchFeatures) -> dict[str, float]:
        """Gene targets, restricted to the held-out entries of the corrupted pass."""
        restrict = bool(self._probe_cfg.masked_cells_only)
        results: dict[str, float] = {}

        for gene in self._gene_index:
            # Selected per gene, not once: under mask_strategy "gene" the held-out entries are
            # drawn independently per (cell, gene), so each target has its own row set.
            g_train = train.select(train.gene_masked[gene]) if restrict else train
            g_val = val.select(val.gene_masked[gene]) if restrict else val
            if g_train.n_cells() == 0 or g_val.n_cells() == 0:
                warnings.warn(f"probe: no held-out cells for gene '{gene}'; skipping.", RuntimeWarning, stacklevel=2)
                continue
            g_train, g_val = self._capped(g_train), self._capped(g_val)
            for embedding in g_train.embeddings:
                scores = score_regression(
                    g_train.embeddings[embedding],
                    g_train.continuous[gene],
                    g_val.embeddings[embedding],
                    g_val.continuous[gene],
                    alpha=float(self._probe_cfg.ridge_alpha),
                )
                for metric, value in scores.items():
                    results[probe_metric_name(f"gene_{gene}", metric, embedding)] = value

        return results

    def _capped(self, features: ProbeBatchFeatures) -> ProbeBatchFeatures:
        """Subsample to ``probe.max_cells`` with a generator seeded fresh on every call.

        Fresh, not shared: a single generator threaded through every target would make each
        draw depend on how many targets were scored before it, so adding a gene to the config
        would silently change every other target's curve.
        """
        rng = np.random.default_rng(int(self._probe_cfg.seed))
        return features.subsample(int(self._probe_cfg.max_cells), rng)

    @staticmethod
    def _masked(features: ProbeBatchFeatures) -> ProbeBatchFeatures:
        """The rows that were reconstruction targets, or every row if nothing was masked.

        The fallback matters for graph-level runs, where `GraphAnnDataModule` deliberately
        leaves val and test uncorrupted -- there the all-False mask means "no cell was held
        out", not "no cell qualifies", and dropping every row would silently log nothing.
        """
        if features.cell_masked is None or not features.cell_masked.any():
            return features
        return features.select(features.cell_masked)

    # ---------------------------------------------------------------- wandb charts

    def chart_series(self, trainer, results: dict[str, float]) -> dict[str, dict[str, float]]:
        """Group this epoch's numbers into ``{chart_key: {series: value}}``.

        The reconstruction losses are pulled from ``trainer.callback_metrics`` rather than
        recomputed: under a dual decoder the training plan already logs ``val_local_loss`` and
        ``val_global_loss``, and those are the same local-vs-global pair as every probe below,
        so they belong on a chart of the same shape.
        """
        series: dict[str, dict[str, float]] = {}

        recon = {}
        for embedding in EMBEDDINGS:
            value = trainer.callback_metrics.get(f"val_{embedding}_loss") if trainer is not None else None
            if value is not None:
                recon[embedding] = float(value)
        if not recon and trainer is not None and trainer.callback_metrics.get("val_loss") is not None:
            # Single-decoder runs report one number for the whole model; charting it keeps the
            # loss panel present rather than silently missing.
            recon["combined"] = float(trainer.callback_metrics["val_loss"])
        if recon:
            series["recon_loss"] = recon

        for name, value in results.items():
            # probe/<target>_<metric>_<embedding> -> chart "<target>_<metric>", series <embedding>
            body = name[len(PROBE_PREFIX) + 1 :]
            target_metric, _, embedding = body.rpartition("_")
            series.setdefault(target_metric, {})[embedding] = value

        return series

    def _record(self, trainer, results: dict[str, float]) -> None:
        self._rounds += 1
        epoch = trainer.current_epoch
        for chart_key, points in self.chart_series(trainer, results).items():
            chart = self._history.setdefault(chart_key, {})
            for series_name, value in points.items():
                chart.setdefault(series_name, []).append((epoch, value))

    def _publish_charts(self, *, final: bool = False) -> None:
        """Re-render one wandb line chart per target, with every embedding as its own line.

        Throttled by ``probe.chart_every_n_epochs``: every refresh re-uploads the whole curve
        as a fresh wandb Table, so refreshing on every round costs O(epochs^2) bytes and one
        file per chart per round. ``final=True`` overrides the cadence so the end-of-run chart
        is always complete.
        """
        if not self._probe_cfg.wandb_charts or not self._cfg.wandb.use:
            return
        cadence = max(1, int(self._probe_cfg.chart_every_n_epochs))
        if not final and self._rounds % cadence != 0:
            return

        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return

        payload = {}
        for chart_key, chart in self._history.items():
            names = sorted(chart)
            # line_series needs one x per series; each series carries its own epoch list
            # because a series can start late (a probe that was skipped on an early epoch).
            xs = [[epoch for epoch, _ in chart[name]] for name in names]
            ys = [[value for _, value in chart[name]] for name in names]
            payload[f"probe_chart/{chart_key}"] = wandb.plot.line_series(
                xs=xs,
                ys=ys,
                keys=names,
                title=chart_key.replace("_", " "),
                xname="epoch",
            )
        if payload:
            wandb.log(payload, commit=False)


def build_probe_obsm(adata, cfg) -> bool:
    """Stack ``probe.regression_obs`` into ``adata.obsm[cfg.dataset.probe_obsm_key]``.

    geome cannot attach a numeric ``obs`` column -- it returns a pandas Series and dies inside
    ``torch.cat`` -- so continuous covariates have to reach the graphs as an obsm matrix. This
    lives beside :func:`collect_features`, which reads the matrix back by position, so the two
    cannot disagree about column order.

    Call it BEFORE ``prepare_geome_dataset``; afterwards the graphs are already built.

    Parameters
    ----------
    adata
        Mutated in place: the obsm key is added.
    cfg
        Read for ``probe.regression_obs`` and ``dataset.probe_obsm_key``.

    Returns
    -------
    bool
        True if a matrix was written, False if there was nothing to write.

    Raises
    ------
    KeyError
        If a named column is not in ``adata.obs``.
    ValueError
        If ``dataset.probe_obsm_key`` is unset, or a column is not numeric -- a categorical
        column silently coerced to codes would be probed as though its labels were a ruler.
    """
    names = list(getattr(cfg.probe, "regression_obs", []) or [])
    if not names:
        return False

    key = cfg.dataset.probe_obsm_key
    if not key:
        raise ValueError(
            f"probe.regression_obs names {names} but dataset.probe_obsm_key is unset, so there "
            "is no obsm key to stack them into and the graphs would carry no probe_targets."
        )

    missing = [c for c in names if c not in adata.obs.columns]
    if missing:
        raise KeyError(f"probe.regression_obs names columns that are not in adata.obs: {missing}")

    import pandas as pd

    non_numeric = [c for c in names if not pd.api.types.is_numeric_dtype(adata.obs[c])]
    if non_numeric:
        raise ValueError(
            f"probe.regression_obs must name NUMERIC obs columns; {non_numeric} are not. A "
            "categorical column belongs in probe.classification_targets instead -- regressing on "
            "its codes would treat the category order as a distance."
        )

    adata.obsm[key] = np.asarray(adata.obs[names].to_numpy(), dtype=np.float32)
    print(f"probe targets -> adata.obsm['{key}']: {names}")
    return True


def build_probe_callback(cfg, adata):
    """Construct the probe callback, or ``None`` when ``probe.use`` is off.

    The ``None`` return is what keeps this feature inert: ``_training.py`` filters it out of the
    callback list, so a run with probes disabled builds exactly the trainer it always did.
    """
    if not getattr(cfg, "probe", None) or not cfg.probe.use:
        return None
    return OnlineProbeCallback(cfg, adata.var_names)
