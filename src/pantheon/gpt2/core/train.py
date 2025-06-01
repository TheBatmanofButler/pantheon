import datasets
import torch.utils.data
import numpy as np
import transformer_lens.utils

import pantheon.gpt2.core.config as config
import pantheon.gpt2.core.tokenize as tokenize
import pantheon.gpt2.core.sample as sample
import pantheon.gpt2.core.device as device


class Trainer:
    def __init__(self, model, batch_size, epochs):
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs

        self.sampler = sample.Sampler(self.model)
        self.optimizer = torch.optim.AdamW(self.model.parameters())

        dataset = datasets.load_dataset("roneneldan/TinyStories", split="train")
        tokenized_dataset = transformer_lens.utils.tokenize_and_concatenate(
            dataset,
            tokenize.tokenizer,
            column_name="text",
            max_length=config.d_sequence,
            add_bos_token=True,
            num_proc=4,
        )

        dataset_dict = tokenized_dataset.train_test_split(test_size=config.test_size)
        self.train_loader = torch.utils.data.DataLoader(
            dataset_dict["train"],
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.test_loader = torch.utils.data.DataLoader(
            dataset_dict["test"],
            batch_size=self.batch_size,
            shuffle=False,
        )

        self.step = 0

    def train(self):
        accuracy = np.nan
        for epoch in range(self.epochs):
            for i, batch in enumerate(self.train_loader):
                loss = self.train_step(batch)
                print(
                    f"Epoch {epoch + 1}, Batch {i + 1}, loss: {loss:.3f}, accuracy: {accuracy:.3f}"
                )

            accuracy = self.evaluate()
            sample_text = self.sampler.sample("Is mayonnaise an instrument?")
            print(sample_text)

    def evaluate(self):
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

    def train_step(self, batch):
        tokens = batch["tokens"].to(device.device)
        logits = self.model(tokens)

        loss = -self.get_log_probs(logits, tokens).mean()
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        self.step += 1

        return loss

    def get_log_probs(self, logits, tokens):
        log_probs = logits.log_softmax(dim=-1)
        log_probs_for_tokens = (
            log_probs[:, :-1]
            .gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1))
            .squeeze(-1)
        )

        return log_probs_for_tokens
