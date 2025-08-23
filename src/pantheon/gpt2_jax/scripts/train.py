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
params = model_lib.init(model_key)

train_dataloader, test_dataloader = load.build_dataloaders(config.gpt2_config)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=config.gpt2_config.learning_rate),
)
optimizer_state = optimizer.init(params)

mesh = jax.make_mesh(
    (config.gpt2_config.num_devices,),
    axis_names=("batch",),
)

batch_sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec("batch"))
replicated_sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec())

params = jax.device_put(params, replicated_sharding)
optimizer_state = jax.device_put(optimizer_state, replicated_sharding)


def loss_fn(params, sample):
    logits = model_lib.forward(params, sample)

    predictions = logits[:-1]
    targets = sample[0][1:]
    attention_mask = sample[1][1:]

    losses = jnp.array(
        [
            -jax.nn.log_softmax(predictions[i], axis=-1)[targets[i]]
            for i in range(len(predictions))
        ]
    )

    masked_losses = losses * attention_mask
    total_loss = jnp.sum(masked_losses)
    total_tokens = jnp.sum(attention_mask)

    return jnp.where(total_tokens > 0, total_loss / total_tokens, 0.0)


def param_loss_fn(params, batch):
    losses = jax.vmap(lambda sample: loss_fn(params, sample))(batch)

    return jnp.mean(losses)


def compute_loss_and_grads(params, batch):
    loss, grads = jax.value_and_grad(param_loss_fn)(params, batch)

    loss = jax.lax.pmean(loss, axis_name="batch")
    grads = jax.lax.pmean(grads, axis_name="batch")

    return loss, grads


@jax.jit
def update_step(params, optimizer_state, batch):
    loss, grads = jax.shard_map(
        compute_loss_and_grads,
        mesh=mesh,
        in_specs=(replicated_sharding.spec, batch_sharding.spec),
        out_specs=(replicated_sharding.spec, replicated_sharding.spec),
    )(params, batch)

    updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
    params = optax.apply_updates(params, updates)

    return params, optimizer_state, loss


wandb.init(
    entity=config.gpt2_config.wandb_entity,
    project=config.gpt2_config.wandb_project,
    config=config.gpt2_config.to_dict(),
)


def evaluate(params, sample):
    logits = model_lib.forward(params, sample)

    predictions = logits[:-1]
    targets = sample[0][1:]
    correct = jnp.argmax(predictions, axis=-1) == targets

    total_correct = jnp.sum(correct)
    total_tokens = correct.size

    return total_correct / total_tokens


for batch_idx, batch in enumerate(train_dataloader):
    batch = jax.device_put(batch, batch_sharding)

    step = batch_idx + 1

    params, optimizer_state, loss = update_step(params, optimizer_state, batch)

    print(f"Batch {step}: Loss = {loss:.4f}")
    wandb.log(
        data={"train_loss": loss},
        step=step,
    )

    if step % 100 == 0:
        test_batch = next(iter(test_dataloader))
        test_batch = jax.device_put(test_batch, batch_sharding)

        eval_fn = jax.shard_map(
            lambda params, batch: jax.vmap(lambda sample: evaluate(params, sample))(
                batch
            ),
            mesh=mesh,
            in_specs=(replicated_sharding.spec, batch_sharding.spec),
            out_specs=batch_sharding.spec,
        )

        batch_accuracies = eval_fn(params, test_batch)
        accuracy = jnp.mean(batch_accuracies)

        print(f"Accuracy = {accuracy}")
        wandb.log(
            data={"accuracy": accuracy},
            step=step,
        )

        print("Saving model")
        cpu_params = jax.device_get(params)
        save.save_model(cpu_params, config.gpt2_config.saved_model_name)

wandb.finish()
