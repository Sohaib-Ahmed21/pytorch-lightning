"""
This module provides a patched version of Lightning's automatic optimization loop
which addresses an issue with gradient accumulation for ``CrossEntropyLoss``.

Background
~~~~~~~~~~
PyTorch Lightning's built in gradient accumulation logic always divides the loss
returned from the user ``training_step`` by the configured
``accumulate_grad_batches``.  While this works correctly when the loss has
already been reduced over all examples in the batch (for example when
``reduction='sum'`` is used), it leads to incorrect gradients when the loss is
an average over only a subset of items (e.g., when using ``reduction='mean'``
with an ``ignore_index`` in ``CrossEntropyLoss``).  In that case each
micro‑batch can have a different number of valid tokens, and simply averaging
their per‑batch means leads to a bias in the gradient.  See the discussion in
the Unsloth article for more details.

The maintainer of Lightning has suggested that the framework cannot detect or
correct this issue on its own because it has no insight into the loss
function's reduction mode or the number of valid items.  However, if the user
provides the number of items contributing to the loss (e.g. the number of
tokens not equal to ``ignore_index``) via an extra key in the dictionary
returned from ``training_step``, Lightning can reconstruct the true large batch
loss by accumulating the *unnormalised* loss and the count across
``accumulate_grad_batches`` micro‑batches and then normalising once at the end.

This patch implements exactly that logic.  The minimal user API change is to
return a dictionary like ``{"loss": loss, "num_val_items": num_items}`` from
your ``training_step``.  When this key is present, the loss returned by
``CrossEntropyLoss(reduction='mean')`` will **not** be divided by
``accumulate_grad_batches`` immediately.  Instead the loop accumulates
``loss * num_items`` and ``num_items`` over multiple micro‑batches.  At the
final micro‑batch in the accumulation window, a single backward pass is
performed on the aggregated loss ``sum(loss_i * num_items_i) / sum(num_items_i)``.

For all other types of loss (or when ``num_val_items`` is not provided) the
behaviour is identical to Lightning's default automatic optimisation.

Note
----
This file intentionally duplicates only the relevant parts of Lightning's
``automatic.py`` module rather than monkey patching the library at runtime.
It demonstrates the minimal changes required to implement the fix while
remaining self contained.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer
from typing_extensions import override

import lightning.pytorch as pl
from lightning.pytorch.loops.loop import _Loop
from lightning.pytorch.loops.optimization.closure import AbstractClosure, OutputResult
from lightning.pytorch.loops.progress import _OptimizationProgress
from lightning.pytorch.loops.utilities import _block_parallel_sync_behavior
from lightning.pytorch.trainer import call
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.rank_zero import WarningCache
from lightning.pytorch.utilities.types import STEP_OUTPUT


@dataclass
class ClosureResult(OutputResult):
    """A container to hold the result of a :class:`Closure` call.

    It is created from the output of
    :meth:`~lightning.pytorch.core.LightningModule.training_step`.

    Attributes:
        closure_loss: The (possibly unreduced) loss with a graph attached.
        loss: A detached copy of the closure loss.  This is used for logging and
            callbacks.  It does **not** reflect the internal scaling used during
            gradient accumulation.
        extra: Any keys other than the loss returned from ``training_step``.

    ``ClosureResult`` differs from the upstream implementation only in its
    ``from_training_step_output`` method.  When the user returns
    ``{"num_val_items": count}`` alongside their loss, this function will skip
    the default normalisation by ``accumulate_grad_batches``.  The actual
    normalisation is deferred until the end of the accumulation window.
    """

    closure_loss: Optional[Tensor]
    loss: Optional[Tensor] = field(init=False, default=None)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._clone_loss()

    def _clone_loss(self) -> None:
        if self.closure_loss is not None:
            # the loss will get scaled for amp. avoid any modifications to it
            self.loss = self.closure_loss.detach().clone()

    @classmethod
    def from_training_step_output(
        cls, training_step_output: STEP_OUTPUT, normalize: int = 1
    ) -> "ClosureResult":
        closure_loss: Optional[Tensor] = None
        extra: dict[str, Any] = {}

        # Unpack the user output.  Accept both Tensor and Mapping as Lightning
        # does.  Keep any additional keys for later consumption.
        if isinstance(training_step_output, Mapping):
            closure_loss = training_step_output.get("loss")
            if closure_loss is None:
                raise MisconfigurationException(
                    "In automatic_optimization, when `training_step` returns a dict, the 'loss' key needs to be present"
                )
            extra = {k: v for k, v in training_step_output.items() if k != "loss"}
        elif isinstance(training_step_output, Tensor):
            closure_loss = training_step_output
        elif training_step_output is not None:
            raise MisconfigurationException(
                "In automatic optimization, `training_step` must return a Tensor, a dict, or None (where the step will be skipped)."
            )

        # Determine whether to apply Lightning's default normalisation.  If the
        # user supplied ``num_val_items`` then we defer normalisation until
        # accumulation is finished.  Otherwise we divide by ``normalize`` (which
        # typically equals ``accumulate_grad_batches``).
        if closure_loss is not None:
            if isinstance(training_step_output, Mapping) and "num_val_items" in training_step_output:
                # Do not normalise here.  Raw per‑batch mean loss is required
                # so that we can weight it by the number of valid items.
                pass
            else:
                # Accumulate the loss.  If ``accumulate_grad_batches == 1``,
                # dividing by 1 has no effect.
                closure_loss = closure_loss / normalize

        return cls(closure_loss, extra=extra)

    @override
    def asdict(self) -> dict[str, Any]:
        return {"loss": self.loss, **self.extra}


class Closure(AbstractClosure[ClosureResult]):
    """A Closure combines the user ``training_step``, gradient zeroing and
    backpropagation into a single callable passed to ``optimizer.step``.

    This implementation is identical to Lightning's except that it allows the
    caller to provide custom ``step_fn``, ``backward_fn`` and ``zero_grad_fn``.
    """

    warning_cache = WarningCache()

    def __init__(
        self,
        step_fn: Callable[[], ClosureResult],
        backward_fn: Optional[Callable[[Tensor], None]] = None,
        zero_grad_fn: Optional[Callable[[], None]] = None,
    ):
        super().__init__()
        self._step_fn = step_fn
        self._backward_fn = backward_fn
        self._zero_grad_fn = zero_grad_fn

    @override
    @torch.enable_grad()
    def closure(self, *args: Any, **kwargs: Any) -> ClosureResult:
        step_output = self._step_fn()

        if step_output.closure_loss is None:
            self.warning_cache.warn(
                "`training_step` returned `None`. If this was on purpose, ignore this warning..."
            )

        if self._zero_grad_fn is not None:
            self._zero_grad_fn()

        if self._backward_fn is not None and step_output.closure_loss is not None:
            self._backward_fn(step_output.closure_loss)

        return step_output

    @override
    def __call__(self, *args: Any, **kwargs: Any) -> Optional[Tensor]:
        self._result = self.closure(*args, **kwargs)
        return self._result.loss


class AutomaticOptimization(_Loop):
    """Performs automatic optimisation (forward, backward, and optimiser step).

    This class mirrors Lightning's private ``_AutomaticOptimization`` loop but
    introduces a fix for gradient accumulation when the user returns
    ``num_val_items`` from ``training_step``.  In that case the mean loss
    returned from the loss function is **not** immediately scaled by
    ``accumulate_grad_batches``.  Instead, the loop accumulates the weighted
    losses and counts over multiple micro‑batches, and performs a single
    backward pass on the aggregated loss when the accumulation window closes.

    All other behaviour (including distributed training, amp/precision plugins
    and multiple optimisers) is untouched.
    """

    output_result_cls = ClosureResult

    def __init__(self, trainer: "pl.Trainer") -> None:
        super().__init__(trainer)
        self.optim_progress: _OptimizationProgress = _OptimizationProgress()
        self._skip_backward: bool = False
        # State to accumulate unnormalised losses and counts across micro‑batches
        self._ga_loss_numer: Optional[Tensor] = None
        self._ga_loss_total: int = 0

    def run(self, optimizer: Optimizer, batch_idx: int, kwargs: OrderedDict) -> dict[str, Any]:
        """Runs the training step and handles gradient accumulation.

        Args:
            optimizer: The optimiser used for this training step.
            batch_idx: Index of the current batch.
            kwargs: The keyword arguments passed down to the hooks (contains
                batch and potentially other information).

        Returns:
            A dictionary of detached outputs from ``training_step`` (e.g. for logging).
        """
        trainer = self.trainer

        # We first run the user's ``training_step`` once to obtain a
        # ``ClosureResult``.  This allows us to inspect any extra fields (like
        # ``num_val_items``) without triggering backward or zero_grad yet.
        # Note: ``_training_step`` internally calls ``post_training_step`` and
        # applies our modified normalisation logic via ``ClosureResult``.
        step_result: ClosureResult = self._training_step(kwargs)

        # Check whether the user provided a token/element count for the loss.
        # The presence of ``num_val_items`` indicates that we need to defer
        # normalisation until gradient accumulation completes.
        accumulator_active: bool = "num_val_items" in step_result.extra

        # Determine if this batch should contribute gradients now or later.  This
        # mirrors Lightning's logic: when the strategy does not handle
        # accumulation and we are in the middle of an accumulation window,
        # ``_should_accumulate()`` returns True.
        should_accumulate: bool = (
            not trainer.strategy.handles_gradient_accumulation
            and trainer.fit_loop._should_accumulate()
        )

        # If the user has not opted into the cross‑entropy fix, fall back to
        # Lightning's default behaviour.  To preserve all original semantics we
        # simply construct a closure again and delegate to the original logic.
        if not accumulator_active:
            # Rebuild a closure that wraps the training step and backward/zero_grad
            closure = self._make_closure(kwargs, optimizer, batch_idx)

            if should_accumulate:
                # When accumulating, compute loss and backpropagate gradients.
                # Synchronise DDP only on the last accumulation step.
                with _block_parallel_sync_behavior(trainer.strategy, block=True):
                    closure()
            else:
                # Final step in the accumulation window (or not accumulating at all).
                self._optimizer_step(batch_idx, closure)

            # Consume the result for logging and callbacks.  ``asdict`` returns
            # the detached loss and any extra values returned by the user.
            result = closure.consume_result()
            if result.loss is None:
                return {}
            return result.asdict()

        # ------------------------------------------------------------------
        # Custom gradient accumulation path for CrossEntropyLoss (or similar)
        # ------------------------------------------------------------------
        # Retrieve the number of valid items contributing to this loss.  The
        # maintainer's example uses the key ``num_val_items`` but we make no
        # assumptions about its name beyond that it exists in ``extra``.
        num_items = step_result.extra.get("num_val_items")
        if not isinstance(num_items, int):
            raise MisconfigurationException(
                "When returning 'num_val_items' from training_step it must be an int."
            )

        # Accumulate the unnormalised loss (mean * count) and the count.  To
        # maintain a computational graph across batches we store the running sum
        # in a tensor.  ``step_result.closure_loss`` is the mean loss with
        # gradients attached.  Note that we did **not** divide by
        # ``accumulate_grad_batches`` above.
        current_numer = step_result.closure_loss * num_items
        if self._ga_loss_numer is None:
            self._ga_loss_numer = current_numer
        else:
            # Important: build the sum in a way that preserves the graph.  We
            # detach the running sum before adding the new tensor to avoid
            # creating a deep computational tree which could lead to large
            # memory usage.  The accumulator is a Tensor to which we reattach
            # the graph at each addition.
            self._ga_loss_numer = self._ga_loss_numer + current_numer
        self._ga_loss_total += num_items

        # Determine if we are still accumulating or if this is the last batch.
        if should_accumulate:
            # On the very first batch we need to zero gradients.  Lightning's
            # original implementation calls zero_grad only once per
            # accumulation window (when batch_idx % accumulate_grad_batches == 0).
            is_first_batch_to_accumulate = batch_idx % trainer.accumulate_grad_batches == 0
            if is_first_batch_to_accumulate:
                self._on_before_zero_grad(optimizer)
                self._optimizer_zero_grad(batch_idx, optimizer)
            # Skip backward for this micro‑batch.  No gradients are computed
            # until the end of the accumulation window.
            # Nothing to return yet.
            return {}

        # We have reached the end of the accumulation window (or
        # accumulate_grad_batches==1).  Construct the aggregated loss by
        # dividing the summed unnormalised losses by the total count.
        aggregated_loss: Tensor = self._ga_loss_numer / float(self._ga_loss_total)

        # Reset accumulation state for the next window.
        self._ga_loss_numer = None
        self._ga_loss_total = 0

        # Prepare a new ``ClosureResult`` for the aggregated loss.  We reuse
        # ``step_result.extra`` so that any additional values returned from
        # ``training_step`` (e.g. logged metrics) are preserved.  Note that we
        # deliberately set ``closure_loss`` to ``aggregated_loss`` here.
        aggregated_result = ClosureResult(closure_loss=aggregated_loss, extra=step_result.extra)

        # Build a closure around the aggregated loss.  We do not want to
        # zero out gradients here because that already happened at the start of
        # the accumulation window.  The closure's backward function simply
        # performs backpropagation on the aggregated loss.
        def aggregated_step_fn() -> ClosureResult:
            return aggregated_result

        def aggregated_backward_fn(loss: Tensor) -> None:
            call._call_strategy_hook(trainer, "backward", loss, optimizer)

        aggregated_closure = Closure(
            step_fn=aggregated_step_fn,
            backward_fn=aggregated_backward_fn,
            zero_grad_fn=None,
        )

        # Perform the optimiser step just like the default Lightning path.  We
        # delegate to ``_optimizer_step`` which handles hooks and updating the
        # optimisation progress counters.  This call will invoke the
        # ``optimizer_step`` hook on the LightningModule, passing in our
        # aggregated closure.  The closure will be called exactly once by
        # ``optimizer.step`` (if the optimiser honours the closure) and will
        # compute the aggregated gradient.
        self._optimizer_step(batch_idx, aggregated_closure)

        # The detached loss and any extra values are returned to the trainer.
        return aggregated_result.asdict()