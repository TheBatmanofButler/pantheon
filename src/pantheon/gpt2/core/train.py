import datasets
import numpy as np
import torch.optim
import torch.utils.data
import transformer_lens.utils
import wandb

import pantheon.gpt2.core.config as config
import pantheon.gpt2.core.device as device
import pantheon.gpt2.core.sample as sample
import pantheon.gpt2.core.tokenize as tokenize


class Trainer:
    def __init__(self, model, num_sequences_per_batch, epochs, save_fn=None):
        self.model = model
        self.num_sequences_per_batch = num_sequences_per_batch
        self.epochs = epochs
        self.save_fn = save_fn

        self.sampler = sample.Sampler(self.model)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.config["learning_rate"],
            weight_decay=config.config["weight_decay"],
        )

        dataset = datasets.load_dataset(config.config["dataset"], split="train")
        if config.config["limited_dataset_size"]:
            dataset = dataset.select(range(config.config["limited_dataset_size"]))

        tokenized_dataset = transformer_lens.utils.tokenize_and_concatenate(
            dataset,
            tokenize.tokenizer,
            column_name="text",
            max_length=config.config["context_window"],
            add_bos_token=True,
            num_proc=4,
        )

        dataset_dict = tokenized_dataset.train_test_split(
            test_size=config.config["test_size"]
        )
        self.train_loader = torch.utils.data.DataLoader(
            dataset_dict["train"],
            batch_size=self.num_sequences_per_batch,
            shuffle=True,
            pin_memory=True,
        )
        self.test_loader = torch.utils.data.DataLoader(
            dataset_dict["test"],
            batch_size=self.num_sequences_per_batch,
            shuffle=False,
            pin_memory=True,
        )

        self.step = 0
        self.run = None

    def train(self):
        # self.run = wandb.init(
        #     # Set the wandb entity where your project will be logged (generally your team name).
        #     entity="the-ganesh-ravichandran-none",
        #     # Set the wandb project where this run will be logged.
        #     project="gpt2",
        #     # Track hyperparameters and run metadata.
        #     config=config.config,
        # )

        accuracy = np.nan

        torch.cuda.memory._record_memory_history()

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=5),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler("."),
        ) as prof:
            for epoch in range(self.epochs):
                for step_index, batch in enumerate(self.train_loader):
                    loss = self.train_step(batch, step_index)
                    prof.step()
                    print(
                        f"Epoch {epoch + 1}, Batch {step_index + 1}, loss: {loss:.3f}, accuracy: {accuracy:.3f}"
                    )

                    if (
                        config.config["max_batches_per_epoch"]
                        and step_index > config.config["max_batches_per_epoch"]
                    ):
                        break

                accuracy = self.evaluate()
                # sample_text = self.sampler.sample("Is mayonnaise an instrument?")
                # print("\n")
                # print(sample_text)

                # if self.save_fn:
                #     print("Saving model params to disk.")
                #     self.save_fn()

        prof.export_memory_timeline("shapes.html", device="cuda:0")

        torch.cuda.memory._dump_snapshot("memory.pickle")

        torch.cuda.memory._record_memory_history(enabled=None)

        # self.run.finish()

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
        # self.run.log({"accuracy": accuracy}, step=self.step)

        self.model.train()

        return accuracy

    def train_step(self, batch, step_index):
        accumulation_steps = config.config["accumulation_steps"] or 1
        print(step_index, accumulation_steps)

        tokens = batch["tokens"].to(device.device)
        logits = self.model(tokens)

        loss = -self.get_log_probs(logits, tokens).mean()
        loss /= accumulation_steps

        loss.backward()

        if (step_index + 1) % accumulation_steps == 0:
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.step += 1
        # self.run.log(
        #     {"train_loss": loss},
        #     step=self.step,
        # )

        return loss

    def get_log_probs(self, logits, tokens):
        log_probs = logits.log_softmax(dim=-1)
        log_probs_for_tokens = (
            log_probs[:, :-1]
            .gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1))
            .squeeze(-1)
        )

        return log_probs_for_tokens
