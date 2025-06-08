import torch.optim
import torch
import torch.utils.data
from typing import List, Dict, Callable

import pantheon.gpt2.instrumentation.trainer as trainer
import pantheon.gpt2.core.device as device
import pantheon.gpt2.core.loss as loss_utils
import pantheon.gpt2.data.load as load
import pantheon.gpt2.core.config as config
import pantheon.gpt2.core.model as model


class Trainer(trainer.InstrumentedTrainer):
    def __init__(
        self,
        modes: List[trainer.TrainingMode],
        config: config.GPT2Config,
        save_fn: Callable[[], None],
        model: model.GPT2,
    ) -> None:
        super().__init__(modes, config, save_fn)

        self.config = config
        self.model = model

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self._train_loader, self.test_loader = load.build_dataloaders(config=config)

    @property
    def train_loader(self) -> torch.utils.data.DataLoader:
        return self._train_loader

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        step_index: int,
    ) -> torch.Tensor:
        accumulation_steps = self.config.accumulation_steps or 1

        tokens = batch["tokens"].to(device.device)
        logits = self.model(tokens)

        loss = -loss_utils.get_log_probs(logits, tokens).mean()
        loss /= accumulation_steps

        loss.backward()

        if (step_index + 1) % accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss

    def evaluate(self) -> float:
        self.model.eval()
        total_correct = 0
        total_samples = 0

        for batch in self.test_loader:
            tokens = batch["tokens"].to(device.device)
            logits = self.model(tokens)[:, :-1]
            predicted_tokens = logits.argmax(dim=-1)
            total_correct += (predicted_tokens == tokens[:, 1:]).sum().item()
            total_samples += tokens.size(0) * (tokens.size(1) - 1)

        accuracy = total_correct / total_samples

        self.model.train()

        return accuracy
