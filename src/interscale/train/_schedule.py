"""Staged training of the local and global components (``.claude/global_plan.md``, Stage C).

A joint run trains every component at once on the mean of the dual decoder's two losses. A
*schedule* splits the epoch budget into phases, each of which can freeze a component and weight
the two heads' losses differently -- e.g. train the local component first, then the global one
on top of the frozen local embedding. Phases run one after the other in one process, so the
weights carry over in memory.

A phase is :class:`Phase`: its share of the epoch budget, the components it freezes, the loss
weights of the two heads (None: the default mean), and a factor on the learning rate. Schedules
come from ``optim.schedule``: a preset name (:data:`SCHEDULES`), or ``custom`` with the phases
listed in ``optim.schedule_phases``. ``joint`` -- one phase, everything trained, ``optim.frozen``
and ``optim.loss_weights`` applied -- is the run as it always was.

Freezing a component sets ``requires_grad = False`` on all its parameters for the phase (so the
optimizer, which takes trainable parameters only, never sees them) and keeps it in eval mode, so
its dropout is off: a frozen GCN hands the transformer the same embedding at every step.
"""

from __future__ import annotations

import hashlib
import math
from contextlib import contextmanager
from dataclasses import dataclass

from torch import nn

#: The components a phase can freeze or weight.
COMPONENTS = ("local", "global")

#: The dual decoder's loss weights that reproduce the loss of every run before Stage C: the mean
#: of the two heads' losses (one masked loss over the stacked predictions).
DEFAULT_LOSS_WEIGHTS = {"local": 0.5, "global": 0.5}

#: ``optim.schedule`` values, and the tag each puts into checkpoint names (``joint`` adds none).
SCHEDULES = {
    "joint": "",
    "local_then_global": "l2g",
    "local_then_global_ft": "l2gft",
    "alternating": "alt",
    "alternating_coupled": "altc",
    "custom": "custom",
}

#: The keys a custom phase may set.
PHASE_KEYS = ("share", "frozen", "loss_weights", "lr_scale")

#: The learning-rate factor of ``local_then_global_ft``'s joint fine-tuning phase.
FINETUNE_LR_SCALE = 0.1


@dataclass(frozen=True)
class Phase:
    """One phase of a schedule.

    Attributes
    ----------
    share
        Its share of the epoch budget; the shares of a schedule sum to 1.
    frozen
        Components (:data:`COMPONENTS`) not trained in this phase.
    loss_weights
        ``{"local": w, "global": w}`` on the two heads' losses, or None for the default mean
        (:data:`DEFAULT_LOSS_WEIGHTS`, computed exactly as before Stage C).
    lr_scale
        Factor on ``optim.lr`` for this phase.
    """

    share: float
    frozen: tuple[str, ...] = ()
    loss_weights: dict | None = None
    lr_scale: float = 1.0


def resolve_loss_weights(weights) -> dict | None:
    """``{"local": w, "global": w}`` as floats, or None when they are the default (or not given)."""
    if weights is None:
        return None
    resolved = {name: float(weights[name]) for name in COMPONENTS}
    return None if resolved == DEFAULT_LOSS_WEIGHTS else resolved


def _only_local(weight_local: float = 1.0) -> dict:
    return {"local": weight_local, "global": 0.0}


def _only_global() -> dict:
    return {"local": 0.0, "global": 1.0}


def schedule_phases(cfg) -> list[Phase]:
    """The phases of ``cfg.optim.schedule``.

    * ``joint``: one phase with ``optim.frozen`` and ``optim.loss_weights``.
    * ``local_then_global``: the local component alone (global frozen, only the local loss), then
      the global one alone (local frozen, only the global loss); half the budget each.
    * ``local_then_global_ft``: the same on 40% + 40%, then 20% of joint fine-tuning at
      :data:`FINETUNE_LR_SCALE` x the learning rate.
    * ``alternating``: ``optim.schedule_cycles`` cycles of (local alone, global alone), equal
      shares. It ends on a global phase, so the global head is trained last on top of the final
      local embedding. The local phases use only the local loss: the GCN never learns from the
      global one.
    * ``alternating_coupled``: the same, but the local phases keep the default mean of both losses,
      so the GCN can adapt its embedding to what the (frozen) global component needs.
    * ``custom``: ``optim.schedule_phases``, a list of dicts with :data:`PHASE_KEYS`.
    """
    optim = cfg.optim
    name = optim.get("schedule", "joint")
    if name == "joint":
        return [
            Phase(
                share=1.0,
                frozen=tuple(optim.get("frozen", [])),
                loss_weights=resolve_loss_weights(optim.get("loss_weights", None)),
            )
        ]
    if name == "local_then_global":
        return [Phase(0.5, ("global",), _only_local()), Phase(0.5, ("local",), _only_global())]
    if name == "local_then_global_ft":
        return [
            Phase(0.4, ("global",), _only_local()),
            Phase(0.4, ("local",), _only_global()),
            Phase(0.2, (), None, FINETUNE_LR_SCALE),
        ]
    if name in ("alternating", "alternating_coupled"):
        cycles = int(optim.get("schedule_cycles", 4))
        share = 1.0 / (2 * cycles)
        local_weights = _only_local() if name == "alternating" else None
        phases = []
        for _ in range(cycles):
            phases += [Phase(share, ("global",), local_weights), Phase(share, ("local",), _only_global())]
        return phases
    if name == "custom":
        return [
            Phase(
                share=float(phase["share"]),
                frozen=tuple(phase.get("frozen", ())),
                loss_weights=resolve_loss_weights(phase.get("loss_weights", None)),
                lr_scale=float(phase.get("lr_scale", 1.0)),
            )
            for phase in optim.get("schedule_phases", [])
        ]
    raise ValueError(f"optim.schedule must be one of {sorted(SCHEDULES)}, got {name!r}.")


def phase_epochs(phases: list[Phase], max_epochs: int) -> list[int]:
    """Epochs per phase: ``max_epochs`` split by the shares, every phase at least one epoch.

    Largest-remainder rounding, so the epochs sum to ``max_epochs`` exactly -- a staged run gets
    the same budget as a joint one. With fewer epochs than phases (a short smoke run) every phase
    gets one epoch, and the total exceeds ``max_epochs``.
    """
    n = len(phases)
    if max_epochs < n:
        return [1] * n
    raw = [phase.share * max_epochs for phase in phases]
    epochs = [math.floor(r) for r in raw]
    by_remainder = sorted(range(n), key=lambda i: raw[i] - epochs[i], reverse=True)
    for i in by_remainder[: max_epochs - sum(epochs)]:
        epochs[i] += 1
    for i in range(n):
        while epochs[i] < 1:
            donor = max(range(n), key=lambda k: epochs[k])
            epochs[donor] -= 1
            epochs[i] += 1
    return epochs


def phase_warmup(lr_warmup: int, epochs: int, max_epochs: int) -> int:
    """A phase's warm-up: ``optim.lr_warmup`` scaled to the phase's share of the run, at least 1.

    At least 1 because ``CosineWarmupScheduler`` divides by it.
    """
    return max(1, round(lr_warmup * epochs / max(1, max_epochs)))


def component_modules(module) -> dict[str, nn.Module]:
    """``{component: submodule}`` of a model's module.

    A combined module has ``local`` (``local_module``) and ``global`` (``global_module``); the
    module of a GlobalModel is all ``global``, that of a LocalModel all ``local``.
    """
    if hasattr(module, "local_module") and hasattr(module, "global_module"):
        return {"local": module.local_module, "global": module.global_module}
    if getattr(module, "registered_global_component", False):
        return {"global": module}
    return {"local": module}


def frozen_modules(module, names) -> list[nn.Module]:
    """The submodules of ``module`` that the components ``names`` stand for, checked."""
    components = component_modules(module)
    missing = sorted(set(names) - set(components))
    if missing:
        raise ValueError(f"cannot freeze {missing}: this model has only {sorted(components)}.")
    if names and set(components) <= set(names):
        raise ValueError(f"freezing {sorted(names)} leaves nothing to train in this model.")
    return [components[name] for name in names]


@contextmanager
def freezing(module, names):
    """Freeze the components ``names`` of ``module`` for the duration of the block.

    Sets ``requires_grad = False`` on every parameter of those components and restores each
    parameter's own previous flag on exit -- a parameter that was never trainable stays so.
    Keeping the frozen parts in eval mode is the training plan's job, since Lightning puts the
    whole model back into training mode at every epoch.
    """
    saved = [(p, p.requires_grad) for sub in frozen_modules(module, tuple(names)) for p in sub.parameters()]
    for p, _ in saved:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, flag in saved:
            p.requires_grad_(flag)


def schedule_tag(cfg) -> str:
    """The training-tag part for a schedule, or ``""`` for ``joint``.

    ``sch-l2g``, ``sch-l2gft``, ``sch-alt<cycles>``, ``sch-altc<cycles>``, or
    ``sch-custom-<hash>`` (a hash of the phases, so two custom schedules do not share a name).
    """
    optim = cfg.optim
    name = optim.get("schedule", "joint")
    if name == "joint":
        return ""
    tag = SCHEDULES[name]
    if name in ("alternating", "alternating_coupled"):
        tag += str(int(optim.get("schedule_cycles", 4)))
    if name == "custom":
        digest = hashlib.sha1(repr(schedule_phases(cfg)).encode()).hexdigest()[:6]
        tag += f"-{digest}"
    return f"sch-{tag}"


def validate_schedule(cfg) -> None:
    """Reject schedules, loss weights and freeze lists that cannot run or are ambiguous.

    Raises
    ------
    ValueError
        With the reason.
    """
    optim = cfg.optim
    name = optim.get("schedule", "joint")
    if name not in SCHEDULES:
        raise ValueError(f"optim.schedule must be one of {sorted(SCHEDULES)}, got {name!r}.")
    _check_weights(optim.get("loss_weights", None), "optim.loss_weights")
    _check_frozen(optim.get("frozen", []), "optim.frozen")

    if name != "joint":
        top_weights = resolve_loss_weights(optim.get("loss_weights", None))
        if top_weights is not None or list(optim.get("frozen", [])):
            raise ValueError(
                f"optim.schedule {name!r} sets freezing and loss weights per phase; leave optim.frozen empty "
                "and optim.loss_weights at its default (local 0.5, global 0.5), or use schedule 'custom'."
            )
        if not cfg.model.decoder.get("dual_decoder", False):
            raise ValueError(
                f"optim.schedule {name!r} trains the local and the global component in separate phases, each on "
                "its own head's loss; it needs model.decoder.dual_decoder: true."
            )
    if name in ("alternating", "alternating_coupled") and int(optim.get("schedule_cycles", 4)) < 1:
        raise ValueError(f"optim.schedule_cycles must be >= 1, got {optim.get('schedule_cycles')}.")
    if name == "custom":
        phases = list(optim.get("schedule_phases", []))
        if not phases:
            raise ValueError("optim.schedule 'custom' needs its phases in optim.schedule_phases.")
        for i, phase in enumerate(phases):
            where = f"optim.schedule_phases[{i}]"
            unknown = sorted(set(phase) - set(PHASE_KEYS))
            if unknown:
                raise ValueError(f"{where} has unknown keys {unknown}; a phase sets {list(PHASE_KEYS)}.")
            if "share" not in phase or not float(phase["share"]) > 0:
                raise ValueError(f"{where} needs a share > 0 of the epoch budget.")
            if not float(phase.get("lr_scale", 1.0)) > 0:
                raise ValueError(f"{where}: lr_scale must be > 0.")
            _check_weights(phase.get("loss_weights", None), f"{where}.loss_weights")
            _check_frozen(phase.get("frozen", []), f"{where}.frozen")
        total = sum(float(phase["share"]) for phase in phases)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"the shares of optim.schedule_phases sum to {total:g}, not 1.")

    weighted = any(phase.loss_weights is not None for phase in schedule_phases(cfg))
    if weighted and optim.get("loss", None) in (None, "none", "None", "SCE_EntropyATT_Loss"):
        raise ValueError(
            f"loss weights other than the default need each head's own loss, which optim.loss "
            f"{optim.get('loss')!r} does not give."
        )


def _check_weights(weights, where: str) -> None:
    if weights is None:
        return
    if set(weights) != set(COMPONENTS):
        raise ValueError(f"{where} must set exactly {list(COMPONENTS)}, got {sorted(weights)}.")
    values = [float(weights[name]) for name in COMPONENTS]
    if min(values) < 0 or sum(values) <= 0:
        raise ValueError(f"{where} must be >= 0 and not both 0, got {dict(weights)}.")


def _check_frozen(names, where: str) -> None:
    names = list(names)
    unknown = sorted(set(names) - set(COMPONENTS))
    if unknown:
        raise ValueError(f"{where} names unknown components {unknown}; use {list(COMPONENTS)}.")
    if set(COMPONENTS) <= set(names):
        raise ValueError(f"{where} freezes every component, leaving nothing to train.")
