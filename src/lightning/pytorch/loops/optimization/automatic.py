# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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

    It is created from the output of :meth:`~lightning.pytorch.core.LightningModule.training_step`.

    Attributes:
        closure_loss: The loss with a graph attached.
        loss: A detached copy of the closure loss.
        extra: Any keys other than the loss returned.

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
    def from_training_step_output(cls, training_step_output: STEP_OUTPUT, normalize: int = 1) -> "ClosureResult":
        closure_loss, extra = None, {}

        if isinstance(training_step_output, Mapping):
            closure_loss = training_step_output.get("loss")
            if closure_loss is None:
                raise MisconfigurationException(
                    "In automatic_optimization, when `training_step` returns a dict, the 'loss' key needs to be present"
                )
            extra = {k: v for k, v in training_step_output.items() if k != "loss"}
            # print(extra)
        elif isinstance(training_step_output, Tensor):
            closure_loss = training_step_output
        elif training_step_output is not None:
            raise MisconfigurationException(
                "In automatic optimization, `training_step` must return a Tensor, a dict, or None (where the step will"
                " be skipped)."
            )

        if closure_loss is not None:
            # accumulate the loss. If ``accumulate_grad_batches == 1``, no effect
            # note: avoid in-place operation `x /= y` here on purpose
            closure_loss = closure_loss / normalize
            # print(closure_loss)

        return cls(closure_loss, extra=extra)

    @override
    def asdict(self) -> dict[str, Any]:
        return {"loss": self.loss, **self.extra}


class Closure(AbstractClosure[ClosureResult]):
    """An implementation of a :class:`AbstractClosure` for automatic optimization in Lightning that combines three
    elementary closures into one: ``training_step``, ``backward`` and ``zero_grad``.

    The Closure gets created by the training loop(s) and is then passed to the
    :meth:`torch.optim.Optimizer.step` method. An optimizer is responsible for calling the closure and optionally
    do something with the output.

    Args:
        step_fn: This is typically the :meth:`lightning.pytorch.core.module.LightningModule.training_step
            wrapped with processing for its outputs
        backward_fn: A function that takes a loss value as input, performs back-propagation and returns the loss value.
            Can be set to ``None`` to skip the backward operation.
        zero_grad_fn: A function that zeroes the gradients. Can be set to ``None`` to skip zero_grad, for example
            when accumulating gradients.

    Example:

        closure = Closure()
        optimizer = torch.optim.Adam(...)
        optimizer.step(closure)
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
            self.warning_cache.warn("`training_step` returned `None`. If this was on purpose, ignore this warning...")

        if self._zero_grad_fn is not None:
            self._zero_grad_fn()

        if self._backward_fn is not None and step_output.closure_loss is not None:
            self._backward_fn(step_output.closure_loss)

        return step_output

    @override
    def __call__(self, *args: Any, **kwargs: Any) -> Optional[Tensor]:
        self._result = self.closure(*args, **kwargs)
        return self._result.loss


_OUTPUTS_TYPE = dict[str, Any]

def _fix_cross_entropy(batch, loss):
    """Adjust CrossEntropyLoss to handle gradient accumulation correctly.

    Converts mean reduction -> sum, counts valid (non-ignored) tokens,
    and returns (fixed_loss, num_items).
    """
    ignore_index = getattr(loss, "ignore_index", -100)
    reduction = getattr(loss, "reduction", "mean")

    # Detect tokens/items in batch
    num_items = None
    if isinstance(batch, dict) and "labels" in batch:
        labels = batch["labels"]
        if labels.ndim >= 2:
            num_items = int((labels != ignore_index).sum().item())
        else:
            num_items = labels.numel()

    # only fix mean reduction (sum already fine)
    if reduction == "mean" and num_items is not None and num_items > 0:
        loss = loss * num_items  # make it equivalent to summed loss
    return loss, num_items

class _AutomaticOptimization(_Loop):
    """Performs automatic optimization (forward, zero grad, backward, optimizer step)"""

    output_result_cls = ClosureResult

    def __init__(self, trainer: "pl.Trainer") -> None:
        super().__init__(trainer)
        self.optim_progress: _OptimizationProgress = _OptimizationProgress()
        self._skip_backward: bool = False
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
        print("aggregated_loss:", self._ga_loss_total)

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

    def _make_closure(self, kwargs: OrderedDict, optimizer: Optimizer, batch_idx: int) -> Closure:
        """Build a closure object that captures the given arguments and runs the `training_step` function and
        optionally other functions such as `backward` and `zero_grad`."""
        step_fn = self._make_step_fn(kwargs)
        backward_fn = self._make_backward_fn(optimizer)
        zero_grad_fn = self._make_zero_grad_fn(batch_idx, optimizer)
        return Closure(step_fn=step_fn, backward_fn=backward_fn, zero_grad_fn=zero_grad_fn)

    def _make_step_fn(self, kwargs: OrderedDict) -> Callable[[], ClosureResult]:
        """Build the step function that runs the `training_step` and processes its output."""
        return partial(self._training_step, kwargs)

    def _make_zero_grad_fn(self, batch_idx: int, optimizer: Optimizer) -> Optional[Callable[[], None]]:
        """Build a `zero_grad` function that zeroes the gradients before back-propagation.

        Returns ``None`` in the case backward needs to be skipped.

        """
        if self._skip_backward:
            return None

        is_first_batch_to_accumulate = batch_idx % self.trainer.accumulate_grad_batches == 0
        if not is_first_batch_to_accumulate:
            return None

        def zero_grad_fn() -> None:
            self._on_before_zero_grad(optimizer)
            self._optimizer_zero_grad(batch_idx, optimizer)

        return zero_grad_fn

    def _make_backward_fn(self, optimizer: Optimizer) -> Optional[Callable[[Tensor], None]]:
        """Build a `backward` function that handles back-propagation through the output produced by the `training_step`
        function.

        Returns ``None`` in the case backward needs to be skipped.

        """
        if self._skip_backward:
            return None

        def backward_fn(loss: Tensor) -> None:
            call._call_strategy_hook(self.trainer, "backward", loss, optimizer)

        return backward_fn

    def _optimizer_step(
        self,
        batch_idx: int,
        train_step_and_backward_closure: Callable[[], Optional[Tensor]],
    ) -> None:
        """Performs the optimizer step and some sanity checking.

        Args:
            batch_idx: the index of the current batch
            train_step_and_backward_closure: the closure function performing the train step and computing the
                gradients. By default, called by the optimizer (if possible)

        """
        trainer = self.trainer

        # wraps into LightningOptimizer only for running step
        optimizer = trainer.strategy._lightning_optimizers[0]

        # if `strategy.handles_gradient_accumulation`, this method will be called to route into the strategy, but we
        # need to check again if `should_accumulate` before increasing the counters
        should_accumulate = trainer.fit_loop._should_accumulate()
        if not should_accumulate:
            self.optim_progress.optimizer.step.increment_ready()

        # model hook
        call._call_lightning_module_hook(
            trainer,
            "optimizer_step",
            trainer.current_epoch,
            batch_idx,
            optimizer,
            train_step_and_backward_closure,
        )

        if not should_accumulate:
            self.optim_progress.optimizer.step.increment_completed()

    def _on_before_zero_grad(self, optimizer: torch.optim.Optimizer) -> None:
        """Calls the ``on_before_zero_grad`` hook.

        Args:
            optimizer: the current optimizer

        """
        trainer = self.trainer
        self.optim_progress.optimizer.zero_grad.increment_ready()
        call._call_callback_hooks(trainer, "on_before_zero_grad", optimizer)
        call._call_lightning_module_hook(trainer, "on_before_zero_grad", optimizer)
        self.optim_progress.optimizer.zero_grad.increment_started()

    def _optimizer_zero_grad(self, batch_idx: int, optimizer: torch.optim.Optimizer) -> None:
        """Zeroes out all gradients of parameters optimized by the current optimizer.

        Args:
            batch_idx: the index of the current batch
            optimizer: the current optimizer

        """
        trainer = self.trainer
        call._call_lightning_module_hook(trainer, "optimizer_zero_grad", trainer.current_epoch, batch_idx, optimizer)
        self.optim_progress.optimizer.zero_grad.increment_completed()

    def _training_step(self, kwargs: OrderedDict) -> ClosureResult:
        """Performs the actual train step with the tied hooks.

        Args:
            kwargs: the kwargs passed down to the hooks.

        Returns:
            A ``ClosureResult`` containing the training step output.

        """
        trainer = self.trainer

        training_step_output = call._call_strategy_hook(trainer, "training_step", *kwargs.values())
        self.trainer.strategy.post_training_step()  # unused hook - call anyway for backward compatibility

        if training_step_output is None and trainer.world_size > 1:
            raise RuntimeError(
                "Skipping the `training_step` by returning None in distributed training is not supported."
                " It is recommended that you rewrite your training logic to avoid having to skip the step in the first"
                " place."
            )

        return self.output_result_cls.from_training_step_output(training_step_output, trainer.accumulate_grad_batches)
