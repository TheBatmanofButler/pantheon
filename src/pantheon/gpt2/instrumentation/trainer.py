from enum import Enum, auto
from contextlib import ExitStack
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List

import time
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
        train_start_time = time.time()

        with ExitStack() as manager_stack:
            for mode in self.instrumentors:
                self.instrumentors[mode].context = manager_stack.enter_context(
                    self.instrumentors[mode].manager
                )

            accuracy = np.nan

            step_index = 0
            for epoch in range(self.config.epochs):
                for batch_index, batch in enumerate(self.train_loader):
                    batch_start_time = time.time()
                    print(
                        f"Starting Epoch {epoch + 1}, Batch {batch_index + 1}"
                        f"\n  Batch start time (relative): {(batch_start_time - train_start_time) * 1000:.0f}ms"
                    )

                    loss = self.train_step(
                        step_index=step_index,
                        batch=batch,
                    )
                    if TrainingMode.OBSERVABILITY in self.modes:
                        self.instrumentors[TrainingMode.OBSERVABILITY].context.log(
                            step=step_index,
                            content={"train_loss": loss},
                        )

                    print(
                        f"Finished Epoch {epoch + 1}, Batch {batch_index + 1}"
                        f"\n  Loss: {loss:.3f}"
                        f"\n  Accuracy: {accuracy:.3f}"
                        f"\n  Batch process time: {(time.time() - batch_start_time) * 1000:.0f}ms"
                        f"\n  Current wall time: {(time.time() - train_start_time) * 1000:.0f}ms"
                    )
                    if TrainingMode.PERFORMANCE in self.modes:
                        self.instrumentors[TrainingMode.PERFORMANCE].context.step()

                    if (
                        self.config.max_batches_per_epoch
                        and batch_index == self.config.max_batches_per_epoch - 1
                    ):
                        print(
                            f"Ending training early. Saving model params to disk for Epoch {epoch + 1}, Batch {step_index + 1}."
                        )
                        self.save_fn()
                        break

                    step_index += 1

                if TrainingMode.MEMORY not in self.modes:
                    accuracy = self.evaluate()
                else:
                    print("Skipping evaluation for memory mode")

                if TrainingMode.OBSERVABILITY in self.modes:
                    self.instrumentors[TrainingMode.OBSERVABILITY].context.log(
                        step=step_index,
                        content={"accuracy": accuracy},
                    )

                if TrainingMode.CHECKPOINTED_SAVES in self.modes:
                    print(f"Saving model params to disk for Epoch {epoch + 1}.")
                    self.save_fn()

        if TrainingMode.CHECKPOINTED_SAVES not in self.modes:
            print("All epochs complete. Saving model params to disk.")
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
