import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import wandb

import pantheon.gpt2_jax.core.model as model_lib
import pantheon.gpt2_jax.core.config as config
import pantheon.gpt2_jax.data.load as load
import pantheon.gpt2_jax.data.save as save

key = jax.random.PRNGKey(1)
key, model_key = jax.random.split(key)
gpt2 = model_lib.GPT2(model_key)

train_dataloader, test_dataloader = load.build_dataloaders(config.gpt2_config)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=config.gpt2_config.learning_rate),
)
optimizer_state = optimizer.init(eqx.filter(gpt2, eqx.is_array))


def loss_fn(model, sample):
    logits = model(sample)

    predictions = logits[:-1]
    targets = sample[0][1:]

    losses = jnp.array(
        [
            -jax.nn.log_softmax(predictions[i], axis=-1)[targets[i]]
            for i in range(len(predictions))
        ]
    )

    return jnp.mean(losses)


def param_loss_fn(params, static, batch):
    model = eqx.combine(params, static)
    losses = jax.vmap(lambda sample: loss_fn(model, sample))(batch)

    return jnp.mean(losses)


@eqx.filter_jit
def update_step(model, optimizer_state, batch):
    params, static = eqx.partition(model, eqx.is_array)

    loss, grads = jax.value_and_grad(param_loss_fn)(params, static, batch)
    loss = jnp.clip(loss, a_max=100.0)

    updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
    params = eqx.apply_updates(params, updates)

    model = eqx.combine(params, static)

    return model, optimizer_state, loss


wandb.init(
    entity=config.gpt2_config.wandb_entity,
    project=config.gpt2_config.wandb_project,
    config=config.gpt2_config.to_dict(),
)


def evaluate(model, sample):
    logits = model(sample)

    predictions = logits[:-1]
    targets = sample[0][1:]
    correct = jnp.argmax(predictions, axis=-1) == targets

    total_correct = jnp.sum(correct)
    total_tokens = correct.size

    return total_correct / total_tokens


for batch_idx, batch in enumerate(train_dataloader):
    step = batch_idx + 1

    gpt2, optimizer_state, loss = update_step(gpt2, optimizer_state, batch)

    print(f"Batch {step}: Loss = {loss:.4f}")
    wandb.log(
        data={"train_loss": loss},
        step=step,
    )

    if step % 10 == 0:
        batch_accuracies = jax.vmap(lambda sample: evaluate(gpt2, sample))(
            next(iter(test_dataloader))
        )
        accuracy = jnp.mean(batch_accuracies)

        print(f"Accuracy = {accuracy}")
        wandb.log(
            data={"accuracy": accuracy},
            step=step,
        )

save.save_model(gpt2, config.gpt2_config.saved_model_name)
