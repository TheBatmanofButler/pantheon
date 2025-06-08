from enum import Enum, auto
from contextlib import ExitStack
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List

import numpy as np
import torch.utils.data

import pantheon.gpt2.instrumentation.context as context
import pantheon.gpt2.core.config as config


class TrainingMode(Enum):
    OBSERVABILITY = auto()
    PERFORMANCE = auto()
    MEMORY = auto()
    CHECKPOINTED_SAVES = auto()


context_managers = {
    TrainingMode.PERFORMANCE: context.PerformanceContextManager,
    TrainingMode.MEMORY: context.MemoryContextManager,
    TrainingMode.OBSERVABILITY: context.ObservabilityContextManager,
}


class InstrumentedTrainer(ABC):
    def __init__(
        self,
        modes: List[TrainingMode],
        config: config.GPT2Config,
        save_fn: Callable[[], None],
    ) -> None:
        self.instrumentors = initialize_instrumentation(modes, config)

        self.modes = modes
        self.save_fn = save_fn
        self.config = config

    def train(self) -> None:
        with ExitStack() as manager_stack:
            for mode in self.instrumentors:
                self.instrumentors[mode].context = manager_stack.enter_context(
                    self.instrumentors[mode].manager
                )

            accuracy = np.nan

            for epoch in range(self.config.epochs):
                for step_index, batch in enumerate(self.train_loader):
                    if TrainingMode.PERFORMANCE in self.modes:
                        self.instrumentors[TrainingMode.PERFORMANCE].context.step()

                    loss = self.train_step(
                        step_index=step_index,
                        batch=batch,
                    )
                    print(
                        f"Epoch {epoch + 1}, Batch {step_index + 1}, loss: {loss:.3f}, accuracy: {accuracy:.3f}"
                    )

                    if (
                        self.config.max_batches_per_epoch
                        and step_index > self.config.max_batches_per_epoch
                    ):
                        break

                accuracy = self.evaluate()
                if TrainingMode.OBSERVABILITY in self.modes:
                    self.instrumentors[TrainingMode.OBSERVABILITY].context.log(
                        step=step_index,
                        content={"accuracy": accuracy},
                    )

                if TrainingMode.CHECKPOINTED_SAVES in self.modes:
                    print(f"Saving model params to disk for Epoch {epoch + 1}.")
                    self.save_fn()

        if TrainingMode.CHECKPOINTED_SAVES not in self.modes:
            print("Training complete. Saving model params to disk.")
            self.save_fn()

    @property
    @abstractmethod
    def train_loader(self) -> torch.utils.data.DataLoader:
        """Return the training data loader. Must be implemented by child classes."""
        pass

    @abstractmethod
    def train_step(
        self,
        step_index: int,
        batch: Dict[str, Any],
    ) -> float:
        """Execute a single training step. Must be implemented by child classes."""
        pass

    @abstractmethod
    def evaluate(self) -> float:
        """Evaluate the model and return accuracy. Must be implemented by child classes."""
        pass


def initialize_instrumentation(
    modes: List[TrainingMode],
    config: config.GPT2Config,
) -> Dict[TrainingMode, context.ManagedInstrumentor]:
    return {
        mode: context.ManagedInstrumentor(
            manager=context_managers[mode](config=config),
        )
        for mode in modes
        if mode in context_managers
    }
