


import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Optimizer
import numpy as np
from backpack import backpack, extend
from backpack.extensions import BatchGrad, BatchL2Grad
from backpack.utils.examples import get_mnist_dataloader
import time
NUM_EPOCHS = 1
PRINT_EVERY = 50
MAX_ITER = 500
BATCH_SIZE = 64
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np.random.seed(42)


def make_broadcastable(v, X):
 
    broadcasting_shape = (-1, *[1 for _ in X.shape[1:]])
    return v.reshape(broadcasting_shape)


def accuracy(output, targets):
    predictions = output.argmax(dim=1, keepdim=True).view_as(targets)
    return predictions.eq(targets).float().mean().item()


# %%
# Creating the model and loading some data
# ----------------------------------------
#
# We will use a small CNN with 2 convolutions, 2 linear layers,
# and feed it some MNIST data.


def make_small_cnn(outputs=10, channels=(16, 32), fc_dim=32, kernels=(8, 4)):
    return nn.Sequential(
        nn.ZeroPad2d((3, 4, 3, 4)),
        nn.Conv2d(1, channels[0], kernels[0], stride=2, padding=0),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=1),
        nn.Conv2d(channels[0], channels[1], kernels[1], stride=2, padding=0),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=1),
        nn.Flatten(),
        nn.Linear(channels[1] * 4 * 4, fc_dim),
        nn.ReLU(),
        nn.Linear(fc_dim, outputs),
    )


mnist_dataloader = get_mnist_dataloader(batch_size=BATCH_SIZE)

model = make_small_cnn().to(DEVICE)
loss_function = nn.CrossEntropyLoss().to(DEVICE)

# %%
# and we need to ``extend`` the model so that ``BackPACK`` knows about it.

model = extend(model)

# %%
# Computing clipped individual gradients
# -----------------------------------------------------------------
#
# Before writing the optimizer class, let's see how we can use ``BackPACK``
# on a single batch to compute the clipped gradients, without the overhead
# of the optimizer class.
#
# We take a single batch from the data loader, compute the loss,
# and use the ``with(backpack(...))`` syntax to activate two extensions;
# ``BatchGrad`` and ``BatchL2Grad``.

x, y = next(iter(mnist_dataloader))
x, y = x.to(DEVICE), y.to(DEVICE)

loss = loss_function(model(x), y)
with backpack(BatchL2Grad(), BatchGrad()):
    loss.backward()

# %%
# ``BatchGrad`` computes individual gradients and ``BatchL2Grad`` their norm (squared),
# which get stored in the ``grad_batch`` and ``batch_l2`` attributes of the parameters

for p in model.parameters():
    print(
        "{:28} {:32} {}".format(
            str(p.grad.shape), str(p.grad_batch.shape), str(p.batch_l2.shape)
        )
    )

# %%
# To compute the clipped gradients, we need to know the norms of the complete
# individual gradients, but ad the moment they are split across parameters,
# so let's reduce over the parameters

l2_norms_squared_all_params = torch.stack([p.batch_l2 for p in model.parameters()])
l2_norms = torch.sqrt(torch.sum(l2_norms_squared_all_params, dim=0))

# %%
# We can compute the clipping scaling factor for each gradient,
# given a maximum norm ``C``,
#
# .. math::
#
#     \\max(1, \Vert g_i \Vert/C),
#
# as a tensor of ``[N]`` elements.

C = 0.1
scaling_factors = torch.clamp_max(l2_norms / C, 1.0)

# %%
# All that remains is to multiply the individual gradients by those factors
# and sum them to get the update direction for that parameter.

for p in model.parameters():
    clipped_grads = p.grad_batch * make_broadcastable(scaling_factors, p.grad_batch)
    clipped_grad = torch.sum(clipped_grads, dim=0)


# %%
# Writing the optimizer
# ---------------------
# Let's do the same, but in an optimizer class.


import torch
from torch.optim.optimizer import Optimizer

class DP_SGD(Optimizer):
  
    def __init__(self, params, lr=0.1, max_norm=0.01, stddev=2.0, noise = 9):
        self.lr = lr
        self.max_norm = max_norm
        self.stddev = stddev
        self.noise_scale = noise
        super().__init__(params, dict())

    def step(self):
        """Performs a single optimization step.

        The function expects the gradients to have been computed by BackPACK
        and the parameters to have a ``batch_l2`` and ``grad_batch`` attribute.
        """
        l2_norms_all_params_list = []
        for group in self.param_groups:
            for p in group["params"]:
                l2_norms_all_params_list.append(p.batch_l2)

        l2_norms_all_params = torch.stack(l2_norms_all_params_list)
        total_norms = torch.sqrt(torch.sum(l2_norms_all_params, dim=0))
        scaling_factors = torch.clamp_max(total_norms / self.max_norm, 1.0)

        for group in self.param_groups:
            for p in group["params"]:
                clipped_grads = p.grad_batch * make_broadcastable(
                    scaling_factors, p.grad_batch
                )
                clipped_grad = torch.sum(clipped_grads, dim=0)
                noise_magnitude = self.stddev * self.max_norm * self.noise_scale
                noise = torch.randn_like(clipped_grad) * noise_magnitude

                perturbed_update = clipped_grad + noise

                p.data.add_(-self.lr * perturbed_update)



class DPSGD_JL(Optimizer):
    def __init__(self, params=model.parameters(), lr=0.01, epsilon=0.1, noise_scale=0.01, defaults=None):
        self.lr = lr
        self.epsilon = epsilon
        self.noise_scale = noise_scale
        if defaults is None:
            defaults = {}
        super(DPSGD_JL, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                norm = torch.norm(grad)
                if norm > self.epsilon:
                    clipped_norm = norm * self.epsilon / norm.clamp(max=self.epsilon)
                    grad = grad * clipped_norm / norm
                noise = torch.normal(0, self.noise_scale * torch.norm(grad), grad.shape)
                p.data = p.data - self.lr * (grad + noise)

# %%
# Running and plotting
# --------------------
# We can now run our optimizer on MNIST.
# create a new optimizer with the current noise value

opts = []

optimizer_standard = DP_SGD(model.parameters(), lr=0.1, max_norm=0.01, stddev=2.0, noise=0.1)
optimizer = DPSGD_JL(model.parameters(), lr=0.1, epsilon=4)

opts.append(optimizer)
opts.append(optimizer_standard)
all_losses = []
colors = ['g', 'b', 'm', 'c']

for Curr, color in zip(opts, colors):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
    # initialize the losses and accuracies
    losses = []
    accuracies = []

    # initialize the memory usage list
    memory_usage = []

    # initialize time
    start_time = time.time()
    iteration_times = []

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        for batch_idx, (x, y) in enumerate(mnist_dataloader):
            # convert tensors to CPU
            x, y = x.cpu(), y.cpu()

            Curr.zero_grad()

            outputs = model(x)
            loss = loss_function(outputs, y)

            with backpack(BatchGrad(), BatchL2Grad()):
                loss.backward()

            Curr.step()
            model.apply

            # logging
            losses.append(loss.detach().item())
            accuracies.append(accuracy(outputs, y))

            # record memory usage
            memory_usage.append(torch.max(torch.tensor([torch.cuda.max_memory_allocated(), 0])))

            # measure time
            iteration_time = time.time() - epoch_start_time
            iteration_times.append(iteration_time)

            if (batch_idx % PRINT_EVERY) == 0:
                print(
                    f"Epoch {epoch}/{NUM_EPOCHS} Iteration {batch_idx} "
                    f"Minibatch Loss {losses[-1]:.3f} Accuracy {accuracies[-1]:.3f} "
                    f"Time {iteration_time:.3f} sec"
                )

            if MAX_ITER is not None and batch_idx > MAX_ITER:
                break

    all_losses.append(losses)

    # plot the iteration times for the current optimizer
    plt.plot(iteration_times, color=color, label=f'optimizer ={Curr}')

    # print total time taken for the current optimizer
    total_time = time.time() - start_time
    print(f"Total time taken for optimizer {Curr}: {total_time:.3f} sec")
plt.title("Time Vs Iteration")
plt.xlabel("Iteration")
plt.ylabel("Time (sec)")
plt.legend()
plt.show()


# %%
