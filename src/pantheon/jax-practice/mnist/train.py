import time
import numpy as np
import jax.numpy as jnp
from jax.tree_util import tree_map
from jax import random, vmap, value_and_grad, jit
from jax.nn import swish, logsumexp, one_hot

from torch.utils.data import DataLoader, default_collate
from torchvision.datasets import MNIST


def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [
        random_layer_params(m, n, key) for m, n, key in zip(sizes[:-1], sizes[1:], keys)
    ]


num_pixels = 784
layer_sizes = [num_pixels, 512, 10]
step_size = 0.01
num_epochs = 30
batch_size = 32
n_targets = 10
params = init_network_params(layer_sizes, random.PRNGKey(0))


def numpy_collate(batch):
    return tree_map(np.asarray, default_collate(batch))


def flatten_and_cast(image):
    return np.ravel(np.array(image, dtype=jnp.float32))


mnist_dataset = MNIST("/tmp/mnist/", download=True, transform=flatten_and_cast)
training_generator = DataLoader(
    mnist_dataset, batch_size=batch_size, collate_fn=numpy_collate
)

train_images = np.asarray(mnist_dataset.data, dtype=np.float32).reshape(
    len(mnist_dataset.data), -1
)
train_labels = one_hot(
    np.asarray(mnist_dataset.targets),
    n_targets,
)

mnist_dataset_test = MNIST("/tmp/mnist/", download=True, train=False)
test_images = jnp.array(
    mnist_dataset_test.data.numpy().reshape(len(mnist_dataset_test.data), -1),
    dtype=jnp.float32,
)
test_labels = one_hot(
    np.asarray(mnist_dataset_test.targets),
    n_targets,
)


def predict(params, image):
    activations = image

    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = swish(outputs)

    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b

    return logits - logsumexp(logits)


batched_predict = vmap(predict, in_axes=(None, 0))


def loss(params, images, targets):
    preds = batched_predict(params, images)

    return -jnp.mean(preds * targets)


@jit
def update(params, x, y):
    loss_value, grads = value_and_grad(loss)(params, x, y)

    return [
        (w - step_size * dw, b - step_size * db)
        for (w, b), (dw, db) in zip(params, grads)
    ], loss_value


@jit
def accuracy(params, images, targets):
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(batched_predict(params, images), axis=1)

    return jnp.mean(predicted_class == target_class)


for epoch in range(num_epochs):
    start_time = time.time()
    losses = []
    for x, y in training_generator:
        y = one_hot(y, n_targets)
        params, loss_value = update(params, x, y)
        losses.append(loss_value)

    epoch_time = time.time() - start_time

    start_time = time.time()
    train_acc = accuracy(params, train_images, train_labels)
    test_acc = accuracy(params, test_images, test_labels)
    eval_time = time.time() - start_time

    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    print("Eval in {:0.2f} sec".format(eval_time))
    print("Training set loss {}".format(jnp.mean(jnp.array(losses))))
    print("Training set accuracy {}".format(train_acc))
    print("Test set accuracy {}".format(test_acc))
