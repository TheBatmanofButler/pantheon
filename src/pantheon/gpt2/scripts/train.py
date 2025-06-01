import argparse

import pantheon.gpt2.core.model as model
import pantheon.gpt2.core.tokenize as tokenize
import pantheon.gpt2.core.train as train
import pantheon.gpt2.core.config as config
import pantheon.gpt2.core.device as device


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompt", help="Prompt to pass into the model")
    args = parser.parse_args()

    gpt2 = model.GPT2(len(tokenize.tokenizer)).to(device.device)

    trainer = train.Trainer(
        model=gpt2,
        batch_size=config.batch_size,
        epochs=config.epochs,
    )
    trainer.train()
