import math
import os

import lightning as L
import lightning.pytorch as pl
import torch
import wandb
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.trainer import seed_everything

from interscale.evaluation.online_probes import build_probe_callback
from interscale.tl.depth import regime_loss_on
from interscale.tl.utils import get_model_filename_prefix
from interscale.train._schedule import component_modules, freezing, phase_epochs, phase_warmup, schedule_phases
from interscale.train._trainingplans import TrainingPlan
from interscale.train._utils import (
    MetricsHistory,
    NodeMaskResampleCallback,
    StepOffsetWandbLogger,
    build_efficiency_monitor,
    redraws_masks,
    train_gene_means,
)

# from interscale.model.base._trainer import TrainRunner


# adjusted from scvi-tools
# https://github.com/scverse/scvi-tools/blob/main/src/scvi/model/base/_training_mixin.py#L31
# accessed on 24 April 2025
class NodeMaskingTrainingPlan:
    """General purpose class for training plans that mask nodes and predict masked nodes."""

    # _data_splitter_cls = DataSplitter
    _training_plan_cls = TrainingPlan

    def _resolve_monitor(self) -> tuple[str, str]:
        """Metric driving EarlyStopping / ModelCheckpoint, and the direction to optimise it.

        Returns
        -------
        tuple[str, str]
            The metric name and the mode (``"min"`` or ``"max"``).
        """
        monitor = self._cfg.optim.monitor
        if monitor == "auto":
            # val_loss is a poor early-stopping criterion under class imbalance: a collapsed
            # constant predictor is a genuine minimum of the weighted loss.
            monitor = "val_f1_macro" if "classification" in self.prediction_task else "val_loss"
        # Lower is better for a loss and for an error -- mse, and the scaled cosine error, which
        # would otherwise be maximised when monitored. Everything else logged (r2, correlations,
        # cosine similarity, f1, accuracy) is higher-is-better.
        return monitor, ("min" if monitor.endswith(("loss", "error", "mse")) else "max")

    # @devices_dsp.dedent -TODO: Why is this here in scvi-tools?
    def _refuse_unsupported_options(self) -> None:
        """Raise for a config option that this model would silently ignore.

        The config validator does not know the model type: ``optim.regime_loss`` splits the two heads of a
        dual-decoder CombinedModel, ``model.decoder.additive`` adds that pair's decoders into one prediction, and
        ``model.decoder.own_cell_skip`` adds a cell's own token to a GlobalModel's prediction, and
        ``model.global_component.parameters.context_only`` changes the transformer. Any other model has no such
        part, and would train the plain objective.
        """
        if regime_loss_on(self._cfg) and getattr(self.module, "regime", None) is None:
            raise ValueError(
                "optim.regime_loss gives the local and the global head their own hidden cells, but this model has "
                "no such pair of heads: use a CombinedModel with model.decoder.dual_decoder: true."
            )
        params = self._cfg.model.global_component.get("parameters", None)
        wants_context = params is not None and params.get("context_only", "none") != "none"
        transformer = getattr(self.module, "global_module", self.module)
        if wants_context and getattr(transformer, "context_only", "none") == "none":
            raise ValueError(
                "model.global_component.parameters.context_only reads the cells' tokens with a query stream of the "
                "transformer; this model has no transformer (use a GlobalModel or a CombinedModel)."
            )
        if self._cfg.model.decoder.get("additive", False) and not getattr(self.module, "additive", False):
            raise ValueError(
                "model.decoder.additive adds the two decoders of a dual-decoder CombinedModel into one prediction; this "
                "model has no such pair: use a CombinedModel with model.decoder.dual_decoder: true."
            )
        if self._cfg.model.decoder.get("own_cell_skip", False) and not getattr(self.module, "own_cell_skip", False):
            raise ValueError(
                "model.decoder.own_cell_skip adds a cell's own token to the prediction of a GlobalModel; this "
                "model has none to add."
            )

    def train(
        self,
        max_epochs: int,
        shuffle_set_split: bool = True,
        load_sparse_tensor: bool = False,
        early_stopping: bool = True,
        patience: int | None = None,
        datasplitter_kwargs: dict | None = None,
        plan_kwargs: dict | None = None,
        datamodule: L.LightningDataModule | None = None,
        wandb_use: bool | None = None,
        **trainer_kwargs,
    ):
        """Train the model.

        With ``optim.schedule`` other than ``joint`` the run is split into phases
        (``interscale.train._schedule``; ``.claude/global_plan.md``, Stage C): ``max_epochs`` is
        divided by the phases' shares, and each phase trains with its own frozen components, loss
        weights, learning rate, warm-up and cosine, on the weights the previous phase left. All
        phases log to one W&B run, the step continuing across them. Only the last phase restores
        its best epoch, validates, tests and saves. ``self.history_`` is the last phase's history,
        ``self.phase_histories_`` every phase's, and ``self.phase_summaries_`` each phase's end: how
        far each component's weights moved and the last epoch's validation losses (also in the W&B
        summary as ``schedule/phase<k>_<metric>``).

        Parameters
        ----------
        max_epochs
            The maximum number of epochs to train the model. The actual number of
            epochs may be less if early stopping is enabled. Split over the phases of a schedule.
        shuffle_set_split
            Whether to shuffle indices before splitting. If ``False``, the val,
            train, and test set are split in the sequential order of the data.
        load_sparse_tensor
            Whether to load data as sparse tensors.
        early_stopping
            Perform early stopping. Additional arguments can be passed in through
            ``**trainer_kwargs``. In a schedule it applies within each phase.
        patience
            Patience for early stopping: number of epochs to wait for improvement
            before stopping.
        datasplitter_kwargs
            Additional keyword arguments passed into the data splitter. Not used
            if ``datamodule`` is passed in.
        plan_kwargs
            Additional keyword arguments passed into the training plan.
        datamodule
            A :class:`~lightning.pytorch.core.LightningDataModule` instance to use
            for training.
        wandb_use
            Whether to log to Weights & Biases. Defaults to the project config.
        **trainer_kwargs
            Additional keyword arguments passed into the
            :class:`~lightning.pytorch.trainer.trainer.Trainer` (of every phase).
        """
        self._refuse_unsupported_options()
        phases = schedule_phases(self._cfg)
        run = {
            "early_stopping": early_stopping,
            "patience": patience,
            "plan_kwargs": plan_kwargs or {},
            "datamodule": datamodule,
            "wandb_use": wandb_use,
            "trainer_kwargs": trainer_kwargs,
        }
        if len(phases) == 1:
            # One phase (`joint`, the default): the run as it always was, with `optim.frozen` and
            # `optim.loss_weights` applied.
            self._train_phase(
                phases[0],
                0,
                1,
                max_epochs=max_epochs,
                lr=self._cfg.optim.lr * phases[0].lr_scale,
                lr_warmup=self._cfg.optim.lr_warmup,
                lr_max_epochs=self._cfg.optim.n_epochs,
                min_epochs=int(self._cfg.optim.min_epochs),
                **run,
            )
            return

        epochs = phase_epochs(phases, int(max_epochs))
        print(f"Schedule {self._cfg.optim.schedule!r}: {len(phases)} phases, epochs {epochs}")
        for i, (phase, n) in enumerate(zip(phases, epochs, strict=True)):
            print(
                f"  phase {i}: {n} epochs, frozen {list(phase.frozen) or 'none'}, "
                f"loss weights {phase.loss_weights or 'default'}, lr x{phase.lr_scale:g}"
            )
        self.phase_histories_ = []
        self.phase_summaries_ = []
        for i, (phase, n) in enumerate(zip(phases, epochs, strict=True)):
            self._train_phase(
                phase,
                i,
                len(phases),
                max_epochs=n,
                lr=self._cfg.optim.lr * phase.lr_scale,
                lr_warmup=phase_warmup(self._cfg.optim.lr_warmup, n, int(max_epochs)),
                lr_max_epochs=n,
                min_epochs=min(int(self._cfg.optim.min_epochs), n),
                **run,
            )

    def _fit_input_pca(self, datamodule) -> None:
        """``pca_fit: train``: fit a GlobalModel's PCA tokens on the training slides, before any step.

        The masks of a split are drawn when the datamodule is set up, which Lightning does inside
        ``fit``; a graph without one gets its first draw here (the same draw ``setup`` makes).
        """
        params = self._cfg.model.global_component.get("parameters", None)
        if params is None or params.get("type_gex_embedding", None) != "PCA" or params.get("pca_fit", "first_batch") != "train":
            return
        if not hasattr(self.module, "fit_pca_on_training"):
            return
        graphs = list(datamodule.train_data)
        for graph in graphs:
            if getattr(graph, "mask", None) is None:
                datamodule._assign_random_mask(graph)
        rows = self.module.fit_pca_on_training(graphs)
        if rows:
            print(f"PCA tokens fitted on the masked inputs of {len(graphs)} training graphs ({rows} cells)")

    def _train_phase(
        self,
        phase,
        index: int,
        n_phases: int,
        *,
        max_epochs: int,
        lr: float,
        lr_warmup: int,
        lr_max_epochs: int,
        min_epochs: int,
        early_stopping: bool,
        patience: int | None,
        plan_kwargs: dict,
        datamodule: L.LightningDataModule,
        wandb_use: bool | None,
        trainer_kwargs: dict,
    ):
        """One phase of :meth:`train`; the whole run when there is one phase.

        The first phase seeds and opens the W&B run; later ones continue its random streams and its
        logger, whose step they offset past the earlier phases'. The last phase alone checkpoints,
        restores the best epoch, validates, tests and saves. The phase's frozen components are
        untrainable for the fit and trainable again afterwards.
        """
        first, last = index == 0, index == n_phases - 1
        staged = n_phases > 1
        if staged:
            print(f"\n=== phase {index} of {n_phases} ({self._cfg.optim.schedule}): {max_epochs} epochs ===")

        if first:
            seed_everything(self._cfg.optim.seed, workers=True)

            self.wandb_use = wandb_use if wandb_use is not None else self._cfg.wandb.use
            self.batch_size = self._cfg.dataset.batch_size
            self.train_size = self._cfg.dataset.train_size
            self.validation_size = self._cfg.dataset.val_size
            self._fit_input_pca(datamodule)

        # TODO: change steps per epoch to be based on datamodule
        # steps_per_epoch = math.ceil(len(train_ds) / cfg.dataset.batch_size)
        steps_per_epoch = math.ceil(len(datamodule.train_data) / self.batch_size)
        print("Steps per epoch", steps_per_epoch)
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        self.history_ = MetricsHistory()
        if staged:
            self.phase_histories_.append(self.history_)
        mask_resample_callback = NodeMaskResampleCallback() if redraws_masks(self._cfg) else None
        # None unless cfg.probe.use, so a run without probes builds the trainer it always did.
        # Reads adata for var_names only -- the probe resolves gene names to batch.x columns.
        probe_callback = build_probe_callback(self._cfg, self._adata)
        if probe_callback is not None:
            print(
                f"Online probes every {self._cfg.probe.every_n_epochs} epoch(s) on "
                f"{list(self._cfg.probe.embeddings)}: "
                f"classification={list(self._cfg.probe.classification_targets)}, "
                f"genes={list(self._cfg.probe.regression_genes)}, "
                f"gene_vector={bool(self._cfg.probe.get('gene_vector', False))}"
            )
        # None unless cfg.optim.log_efficiency: steps/s, cells/s and peak GPU memory per epoch.
        efficiency_callback = build_efficiency_monitor(self._cfg)
        checkpoint_callback = None
        loss_callback = None
        performance_callback = None

        # defines optimizers, training step, val step, logged metrics
        # Built and attached at model construction (BaseModel._attach_aux_losses), so that a
        # reloaded checkpoint's heads are already in place rather than being replaced here.
        aux_losses = self.module.aux_losses
        if aux_losses:
            print(f"Auxiliary losses: {dict(aux_losses.weights)} (views required: {aux_losses.requires_views})")
        # optim.center_targets: the loss compares predictions and targets minus each gene's mean
        # over the training split (never val or test).
        target_center = None
        if self._cfg.optim.get("center_targets", False):
            target_center = train_gene_means(datamodule.train_data)
            print(f"Loss on gene-centred values: means of {len(target_center)} genes over the training graphs")

        training_plan = self._training_plan_cls(
            self.module,
            self.prediction_task,
            self.prediction_level,
            self._cfg.optim.loss,
            self._cfg.optim.cross_corr,
            self.batch_size,
            self.class_weights,
            self.class_labels if self.prediction_task == "classification" else None,
            **plan_kwargs,
            aux_losses=aux_losses,
            lr_scheduler=self._cfg.optim.lr_scheduler,
            weight_decay=self._cfg.optim.wd,
            lr=lr,
            lr_warmup=lr_warmup,
            lr_max_epochs=lr_max_epochs,
            patience_in_steps=steps_per_epoch,
            sce_gamma=self._cfg.optim.sce_gamma,
            target_center=target_center,
            loss_weights=phase.loss_weights,
            frozen=phase.frozen,
        )

        monitor, mode = self._resolve_monitor()

        if early_stopping:
            # TODO: why does the self.history_ stop working when using loss_callback?
            # loss_callback = EarlyStopping(
            #         monitor="val_loss",
            #         min_delta=0.001,
            #         patience=(patience//2)*steps_per_epoch,  # Often shorter patience for loss
            #         verbose=False,
            #         mode="min"
            #     )
            if not ("classification" in self.prediction_task or "regression" in self.prediction_task):
                raise Exception("Training must be classification or regression based.")
            performance_callback = EarlyStopping(
                monitor=monitor,
                min_delta=float(self._cfg.optim.min_delta),
                patience=int(patience if patience is not None else self._cfg.optim.patience),
                verbose=True,
                mode=mode,
            )

        # Only the last phase checkpoints: an earlier phase hands its final weights on in memory.
        if last and self._cfg.model.save is not None:
            run_name = get_model_filename_prefix(self._cfg, self.local_component, self.global_component)
            if "classification" in self._cfg.dataset.prediction_task:
                checkpoint_callback = ModelCheckpoint(
                    dirpath=self._cfg.model.save,
                    filename=run_name,
                    monitor=monitor,
                    mode=mode,
                    save_top_k=1,
                )  # save the best model according to `monitor`
            elif "regression" in self._cfg.dataset.prediction_task:
                # Every arm of the former if/elif chain built the same callback with a hardcoded
                # monitor="val_loss", so `optim.monitor` was silently ignored for regression while
                # classification honoured it. That is not cosmetic: train() restores the best
                # checkpoint before trainer.validate(), so every metric reported for the run --
                # and therefore whatever a sweep ranks on -- came from the lowest-val_loss epoch,
                # no matter which metric the run was supposed to be selecting for.
                #
                # `auto` still resolves to val_loss/min for regression, so existing configs are
                # unaffected; only a config that explicitly sets optim.monitor changes behaviour.
                checkpoint_callback = ModelCheckpoint(
                    dirpath=self._cfg.model.save,
                    filename=run_name,
                    monitor=monitor,
                    mode=mode,
                    save_top_k=1,
                )

        # Create list of callbacks and filter out None values
        callbacks = [
            callback
            for callback in [
                lr_monitor,
                performance_callback,
                loss_callback,
                self.history_,
                mask_resample_callback,
                checkpoint_callback,
                probe_callback,
                efficiency_callback,
            ]
            if callback is not None
        ]

        # Set up WandB logger if requested: once, in the first phase; later phases reuse it.
        logger = None
        if self.wandb_use:
            if first:
                print("Wandb initialize...")
                run_name = get_model_filename_prefix(self._cfg, self.local_component, self.global_component)
                if self._cfg.wandb.project_name is None:
                    raise ValueError("wandb_project must be specified when use_wandb is True")
                wandb.init(
                    project=self._cfg.wandb.project_name, config=self._cfg, name=run_name, job_type="model_training"
                )
                self._wandb_logger = StepOffsetWandbLogger(name=run_name, log_model=True)
            logger = self._wandb_logger

        # Where each component's weights start, for how far the phase moves them.
        start = (
            {name: _parameter_vector(part) for name, part in component_modules(self.module).items()} if staged else None
        )
        with freezing(self.module, phase.frozen):
            if logger is not None:
                total_params = sum(p.numel() for p in self.module.parameters())
                trainable_params = sum(p.numel() for p in self.module.parameters() if p.requires_grad)
                if first:
                    wandb.log({"total_parameters": total_params, "trainable_parameters": trainable_params})
                    print(f"Total parameters: {total_params:,}")
                    print(f"Trainable parameters: {trainable_params:,}")
                if staged:
                    wandb.log({"schedule/phase": index, "schedule/trainable_parameters": trainable_params})
            if staged:
                trainable = sum(p.numel() for p in self.module.parameters() if p.requires_grad)
                print(f"phase {index}: frozen {list(phase.frozen) or 'none'}, {trainable:,} trainable parameters")

            if not last:
                # Lightning adds a default ModelCheckpoint (lightning_logs/) unless told not to; an
                # earlier phase hands its weights on in memory and must not write any.
                trainer_kwargs = {**trainer_kwargs, "enable_checkpointing": False}
            trainer = pl.Trainer(
                min_epochs=min_epochs,
                max_epochs=int(max_epochs),
                # enable_progress_bar=True,
                callbacks=callbacks,
                log_every_n_steps=1,
                logger=logger,
                deterministic=True,  # ensure reproducibility
                accelerator=self._cfg.optim.accelerator,  # Default to CPU (can be overridden via trainer_kwargs)
                check_val_every_n_epoch=int(self._cfg.optim.get("check_val_every_n_epoch", 1)),
                **trainer_kwargs,
            )

            trainer.fit(training_plan, datamodule)
            if logger is not None:
                # The next phase's trainer counts from 0 again; its points go after this phase's.
                logger.step_offset += trainer.global_step
            if staged:
                self._record_phase(index, start)
            if not last:
                return

            # EarlyStopping does not restore best weights and ModelCheckpoint only writes them to
            # disk. Without this, the validate()/test() calls below -- and the save() further down --
            # all report the LAST epoch, which for an early-stopped run is `patience` epochs past the
            # best one.
            best_ckpt = getattr(checkpoint_callback, "best_model_path", "") if checkpoint_callback else ""
            if best_ckpt and os.path.exists(best_ckpt):
                best_score = checkpoint_callback.best_model_score
                print(f"Restoring best checkpoint ({monitor}={float(best_score):.4f}) from {best_ckpt}")
                state = torch.load(best_ckpt, map_location="cpu", weights_only=False)["state_dict"]
                missing, unexpected = training_plan.load_state_dict(state, strict=False)
                if missing or unexpected:
                    print(f"restore: missing={missing}, unexpected={unexpected}")
            else:
                print("WARNING: no best checkpoint found; evaluating FINAL-epoch weights.")

            trainer.validate(training_plan, datamodule)
            if self._cfg.dataset.evaluate_test and self.train_size + self.validation_size < 1:
                trainer.test(training_plan, datamodule)

        # Print early stopping information if it was used
        if early_stopping and loss_callback is not None:
            if loss_callback.stopped_epoch > 0:
                print(f"\nEarly stopping triggered at epoch {loss_callback.stopped_epoch}")
                print(f"Best {loss_callback.monitor}: {loss_callback.best_score:.4f}")
        if early_stopping and performance_callback is not None:
            if performance_callback.stopped_epoch > 0:
                print(f"\nEarly stopping triggered at epoch {performance_callback.stopped_epoch}")
                print(f"Best {performance_callback.monitor}: {performance_callback.best_score:.4f}")

        if self._cfg.model.save is not None:
            print("Model checkpoint will be saved in: ", self._cfg.model.save + run_name + "model.ckpt")
            trainer.save_checkpoint(self._cfg.model.save + run_name + "model.ckpt")
            self.save(self._cfg.model.save)

        # Close WandB logger if it was used
        if self.wandb_use and logger is not None:
            logger.finalize("success")

        self.is_trained_ = True

    def _record_phase(self, index: int, start: dict) -> None:
        """A phase's end, in ``self.phase_summaries_`` and the W&B summary (``schedule/phase<k>_<metric>``).

        ``drift_<component>`` is ``||w_end - w_start|| / ||w_start||`` over that component's weights:
        0 for a component the phase kept frozen. Beside it, the validation losses of the phase's last
        epoch (:data:`PHASE_METRICS`, those that were logged).
        """
        summary = {}
        for name, part in component_modules(self.module).items():
            before, after = start[name], _parameter_vector(part)
            norm = float(before.norm())
            summary[f"drift_{name}"] = float((after - before).norm()) / norm if norm > 0 else float("nan")
        logged = [epoch for epoch in self.history_.history if "val_loss" in epoch]
        if logged:
            summary.update({key: float(logged[-1][key]) for key in PHASE_METRICS if key in logged[-1]})
        self.phase_summaries_.append({"phase": index, **summary})
        if self.wandb_use and wandb.run is not None:
            for key, value in summary.items():
                wandb.run.summary[f"schedule/phase{index}_{key}"] = value
        print(f"phase {index} done: " + ", ".join(f"{key} {value:.4g}" for key, value in summary.items()))


#: The validation metrics a staged run records at the end of every phase (``_record_phase``).
PHASE_METRICS = ("val_loss", "val_local_loss", "val_global_loss", "val_masked_scaled_cosine_error")


def _parameter_vector(module) -> torch.Tensor:
    """Every parameter of ``module``, detached and flattened into one float64 vector on the CPU."""
    parts = [p.detach().reshape(-1).double().cpu() for p in module.parameters()]
    return torch.cat(parts) if parts else torch.zeros(0, dtype=torch.float64)


# adjusted from scvi-tools
# https://github.com/scverse/scvi-tools/blob/main/src/scvi/dataloaders/_data_splitting.py#L182
# accessed on 24 April 2025
# class DataSplitter(pl.LightningDataModule):
#     """Creates data loaders ``train_set``, ``validation_set``, ``test_set``.

#     If ``train_size + validation_set < 1`` then ``test_set`` is non-empty.

#     Parameters
#     ----------
#     adata_manager
#         :class:`~scvi.data.AnnDataManager` object that has been created via ``setup_anndata``.
#     train_size
#         float, or None (default is None, which is practicaly 0.9 and potentially adding small last
#         batch to validation cells)
#     validation_size
#         float, or None (default is None)
#     shuffle_set_split
#         Whether to shuffle indices before splitting. If `False`, the val, train, and test set are
#         split in the sequential order of the data according to `validation_size` and `train_size`
#         percentages.
#     external_indexing
#         A list of data split indices in the order of training, validation, and test sets.
#         Validation and test set are not required and can be left empty.
#     **kwargs
#         Keyword args for data loader. If adata has labeled data, data loader
#         class is :class:`~scvi.dataloaders.SemiSupervisedDataLoader`,
#         else data loader class is :class:`~scvi.dataloaders.AnnDataLoader`.

#     Examples
#     --------
#     >>> adata = scvi.data.synthetic_iid()
#     >>> scvi.model.SCVI.setup_anndata(adata)
#     >>> adata_manager = scvi.model.SCVI(adata).adata_manager
#     >>> splitter = DataSplitter(adata)
#     >>> splitter.setup()
#     >>> train_dl = splitter.train_dataloader()
#     """

#     data_loader_cls = AnnDataLoader #TODO: change to InterScale.dataloader.GeomeDataLoader

#     def __init__(
#         self,
#         adata_manager: AnnDataManager,
#         train_size: float | None = None,
#         validation_size: float | None = None,
#         shuffle_set_split: bool = True,
#         external_indexing: list[np.array, np.array, np.array] | None = None,
#         **kwargs,
#     ):
#         super().__init__()
#         self.adata_manager = adata_manager
#         self.train_size_is_none = not bool(train_size)
#         self.train_size = 0.9 if self.train_size_is_none else float(train_size)
#         self.validation_size = validation_size
#         self.shuffle_set_split = shuffle_set_split
#         self.drop_last = kwargs.pop("drop_last", False)
#         self.data_loader_kwargs = kwargs
#         self.external_indexing = external_indexing

#         if self.external_indexing is not None:
#             self.n_train, self.n_val = validate_data_split_with_external_indexing(
#                 self.adata_manager.adata.n_obs,
#                 self.external_indexing,
#                 self.data_loader_kwargs.get("batch_size", settings.batch_size),
#                 self.drop_last,
#             )
#         else:
#             self.n_train, self.n_val = validate_data_split(
#                 self.adata_manager.adata.n_obs,
#                 self.train_size,
#                 self.validation_size,
#                 self.data_loader_kwargs.get("batch_size", settings.batch_size),
#                 self.drop_last,
#                 self.train_size_is_none,
#             )

#     def setup(self, stage: str | None = None):
#         """Split indices in train/test/val sets."""
#         if self.external_indexing is not None:
#             # The structure and its order are guaranteed at this stage
#             # (can include missing indexes for some group)
#             self.train_idx = self.external_indexing[0]
#             self.val_idx = self.external_indexing[1]
#             self.test_idx = self.external_indexing[2]
#         else:
#             # just like it used to be w/o external indexing
#             n_train = self.n_train
#             n_val = self.n_val
#             indices = np.arange(self.adata_manager.adata.n_obs)

#             if self.shuffle_set_split:
#                 random_state = np.random.RandomState(seed=settings.seed)
#                 indices = random_state.permutation(indices)

#             self.val_idx = indices[:n_val]
#             self.train_idx = indices[n_val : (n_val + n_train)]
#             self.test_idx = indices[(n_val + n_train) :]

#     def train_dataloader(self):
#         """Create train data loader."""
#         return self.data_loader_cls(
#             self.adata_manager,
#             indices=self.train_idx,
#             shuffle=True,
#             drop_last=self.drop_last,
#             load_sparse_tensor=self.load_sparse_tensor,
#             pin_memory=self.pin_memory,
#             **self.data_loader_kwargs,
#         )

#     def val_dataloader(self):
#         """Create validation data loader."""
#         if len(self.val_idx) > 0:
#             return self.data_loader_cls(
#                 self.adata_manager,
#                 indices=self.val_idx,
#                 shuffle=False,
#                 drop_last=False,
#                 load_sparse_tensor=self.load_sparse_tensor,
#                 pin_memory=self.pin_memory,
#                 **self.data_loader_kwargs,
#             )
#         else:
#             pass

#     def test_dataloader(self):
#         """Create test data loader."""
#         if len(self.test_idx) > 0:
#             return self.data_loader_cls(
#                 self.adata_manager,
#                 indices=self.test_idx,
#                 shuffle=False,
#                 drop_last=False,
#                 load_sparse_tensor=self.load_sparse_tensor,
#                 pin_memory=self.pin_memory,
#                 **self.data_loader_kwargs,
#             )
#         else:
#             pass
