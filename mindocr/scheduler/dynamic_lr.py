"""Meta learning rate scheduler.

This module implements exactly the same learning rate scheduler as native PyTorch,
see `"torch.optim.lr_scheduler" <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_.
At present, only `constant_lr`, `linear_lr`, `polynomial_lr`, `exponential_lr`, `step_lr`, `multi_step_lr`,
`cosine_annealing_lr`, `cosine_annealing_warm_restarts_lr` are implemented. The number, name and usage of
the Positional Arguments are exactly the same as those of native PyTorch.

However, due to the constraint of having to explicitly return the learning rate at each step, we have to
introduce additional Keyword Arguments. There are only three Keyword Arguments introduced,
namely `lr`, `steps_per_epoch` and `epochs`, explained as follows:
`lr`: the basic learning rate when creating optim in torch.
`steps_per_epoch`: the number of steps(iterations) of each epoch.
`epochs`: the number of epoch. It and `steps_per_epoch` determine the length of the returned lrs.

Since most scheduler in PyTorch are coarse-grained, that is the learning rate is constant within a single epoch.
For non-stepwise scheduler, we introduce several fine-grained variation, that is the learning rate
is also changed within a single epoch. The function name of these variants have the `refined` keyword.
The implemented fine-grained variation are list as follows: `linear_refined_lr`, `polynomial_refined_lr`, etc.

"""
import math
from bisect import bisect_right

from mindspore import nn


def dynamic_lr(lr, end_lr, epochs, step_per_epoch, by_epoch=True):
    if by_epoch:
        lrs = nn.polynomial_decay_lr(
            lr, end_lr, total_step=step_per_epoch * epochs, step_per_epoch=step_per_epoch, decay_epoch=epochs, power=0.9
        )
        return lrs


def constant_lr(factor, total_iters, *, lr, steps_per_epoch, epochs):
    steps = steps_per_epoch * epochs
    lrs = []
    for i in range(steps):
        epoch_idx = math.floor(i / steps_per_epoch)
        if epoch_idx < total_iters:
            lrs.append(lr * factor)
        else:
            lrs.append(lr)
    return lrs


def linear_lr(start_factor, end_factor, total_iters, *, lr, steps_per_epoch, epochs):
    steps = steps_per_epoch * epochs
    lrs = []
    start_lr = lr * start_factor
    end_lr = lr * end_factor
    for i in range(steps):
        epoch_idx = math.floor(i / steps_per_epoch)
        multiplier = min(epoch_idx, total_iters) / total_iters
        lrs.append(start_lr + multiplier * (end_lr - start_lr))
    return lrs


def linear_refined_lr(start_factor, end_factor, total_iters, *, lr, steps_per_epoch, epochs):
    steps = steps_per_epoch * epochs
    lrs = []
    start_lr = lr * start_factor
    end_lr = lr * end_factor
    for i in range(steps):
        epoch_idx = i / steps_per_epoch
        multiplier = min(epoch_idx, total_iters) / total_iters
        lrs.append(start_lr + multiplier * (end_lr - start_lr))
    return lrs


def polynomial_lr(total_iters, power, *, lr, steps_per_epoch, epochs):
    steps = steps_per_epoch * epochs
    lrs = []
    for i in range(steps):
        epoch_idx = math.floor(i / steps_per_epoch)
        lrs.append(lr * (1 - min(epoch_idx, total_iters) / total_iters) ** power)
    return lrs


def polynomial_refined_lr(total_iters, power, *, lr, steps_per_epoch, epochs):
    steps = steps_per_epoch * epochs
    lrs = []
    for i in range(steps):
        epoch_idx = i / steps_per_epoch
        lrs.append(lr * (1 - min(epoch_idx, total_iters) / total_iters) ** power)
    return lrs


def exponential_lr(gamma, *, lr, steps_per_epoch, epochs):
    steps = steps_per_epoch * epochs
    lrs = []
    for i in range(steps):
        epoch_idx = math.floor(i / steps_per_epoch)
        lrs.append(lr * gamma**epoch_idx)
    return lrs


def exponential_refined_lr(gamma, *, lr, steps_per_epoch, epochs):
    steps = steps_per_epoch * epochs
    lrs = []
    for i in range(steps):
        epoch_idx = i / steps_per_epoch
        lrs.append(lr * gamma**epoch_idx)
    return lrs


def step_lr(step_size, gamma, *, lr, steps_per_epoch, epochs):
    steps = steps_per_epoch * epochs
    lrs = []
    for i in range(steps):
        epoch_idx = math.floor(i / steps_per_epoch)
        lrs.append(lr * gamma ** math.floor(epoch_idx / step_size))
    return lrs


def multi_step_lr(milestones, gamma, *, lr, steps_per_epoch, epochs):
    steps = steps_per_epoch * epochs
    milestones = sorted(milestones)
    lrs = []
    for i in range(steps):
        epoch_idx = math.floor(i / steps_per_epoch)
        lrs.append(lr * gamma ** bisect_right(milestones, epoch_idx))
    return lrs


def cosine_decay_lr(decay_epochs, eta_min, *, eta_max, steps_per_epoch, epochs, num_cycles=1, cycle_decay=1.0):
    """update every epoch"""
    tot_steps = steps_per_epoch * epochs
    lrs = []

    for c in range(num_cycles):
        lr_max = eta_max * (cycle_decay**c)
        delta = 0.5 * (lr_max - eta_min)
        for i in range(steps_per_epoch * decay_epochs):
            t_cur = math.floor(i / steps_per_epoch)
            t_cur = min(t_cur, decay_epochs)
            lr_cur = eta_min + delta * (1.0 + math.cos(math.pi * t_cur / decay_epochs))
            if len(lrs) < tot_steps:
                lrs.append(lr_cur)
            else:
                break

    if epochs > num_cycles * decay_epochs:
        for i in range((epochs - (num_cycles * decay_epochs)) * steps_per_epoch):
            lrs.append(eta_min)

    return lrs


def cosine_decay_refined_lr(decay_epochs, eta_min, *, eta_max, steps_per_epoch, epochs, num_cycles=1, cycle_decay=1.0):
    """update every step"""
    tot_steps = steps_per_epoch * epochs
    lrs = []

    for c in range(num_cycles):
        lr_max = eta_max * (cycle_decay**c)
        delta = 0.5 * (lr_max - eta_min)
        for i in range(steps_per_epoch * decay_epochs):
            t_cur = i / steps_per_epoch
            t_cur = min(t_cur, decay_epochs)
            lr_cur = eta_min + delta * (1.0 + math.cos(math.pi * t_cur / decay_epochs))
            if len(lrs) < tot_steps:
                lrs.append(lr_cur)
            else:
                break

    if epochs > num_cycles * decay_epochs:
        for i in range((epochs - (num_cycles * decay_epochs)) * steps_per_epoch):
            lrs.append(eta_min)

    return lrs


def cosine_annealing_lr(t_max, eta_min, *, eta_max, steps_per_epoch, epochs):
    steps = steps_per_epoch * epochs
    delta = 0.5 * (eta_max - eta_min)
    lrs = []
    for i in range(steps):
        t_cur = math.floor(i / steps_per_epoch)
        lrs.append(eta_min + delta * (1.0 + math.cos(math.pi * t_cur / t_max)))
    return lrs


# fmt: off
def cosine_annealing_warm_restarts_lr(te, tm, eta_min, *, eta_max, steps_per_epoch, epochs):
    delta = 0.5 * (eta_max - eta_min)
    tt = 0
    te_next = te
    lrs = []
    for epoch_idx in range(epochs):
        for batch_idx in range(steps_per_epoch):
            lrs.append(eta_min + delta * (1.0 + math.cos(tt)))
            tt = tt + math.pi / te / steps_per_epoch
            if tt >= math.pi:
                tt = tt - math.pi
        if epoch_idx + 1 == te_next:  # time to restart
            tt = 0                    # by setting to 0 we set lr to lr_max, see above
            te = te * tm              # change the period of restarts
            te_next = te_next + te    # note the next restart's epoch
    return lrs


if __name__ == "__main__":
    # Demonstrate how these schedulers work by printing & visualizing the returned list.
    import matplotlib.pyplot as plt
    table = (
        (("constant_lr", constant_lr(0.5, 4, lr=0.05, steps_per_epoch=2, epochs=10),),),
        (("linear_lr", linear_lr(0.5, 1.0, 4, lr=0.05, steps_per_epoch=2, epochs=10)),
         ("linear_refined_lr", linear_refined_lr(0.5, 1.0, 4, lr=0.05, steps_per_epoch=2, epochs=10)),),
        (("polynomial_lr", polynomial_lr(4, 1.0, lr=0.05, steps_per_epoch=2, epochs=10)),
         ("polynomial_refined_lr", polynomial_refined_lr(4, 1.0, lr=0.05, steps_per_epoch=2, epochs=10)),),
        (("exponential_lr", exponential_lr(0.9, lr=0.05, steps_per_epoch=2, epochs=10)),
         ("exponential_refined_lr", exponential_refined_lr(0.9, lr=0.05, steps_per_epoch=2, epochs=10)),),
        (("step_lr", step_lr(3, 0.5, lr=0.05, steps_per_epoch=2, epochs=10)),
         ("multi_step_lr", multi_step_lr([3, 6], 0.5, lr=0.05, steps_per_epoch=2, epochs=10)),),
        (("cosine_decay_lr", cosine_decay_lr(5, 1.0, eta_max=2.0, steps_per_epoch=2, epochs=10)),
         ("cosine_decay_refined_lr", cosine_decay_refined_lr(5, 1.0, eta_max=2.0, steps_per_epoch=2, epochs=10)),),
        (("cosine_annealing_lr", cosine_annealing_lr(5, 0.0, eta_max=1.0, steps_per_epoch=2, epochs=15)),
         (
             "cosine_annealing_warm_restarts_lr",
             cosine_annealing_warm_restarts_lr(5, 2, 0.0, eta_max=1.0, steps_per_epoch=2, epochs=15),
         ),)
    )
    for variants in table:
        n_variants = len(variants)
        fig = plt.figure(figsize=(4, 3 * n_variants))
        for ax_idx, (title, lrs_ms) in enumerate(variants, start=1):
            print(f"name: {title}\nlrs: {lrs_ms}")
            ax = plt.subplot(n_variants, 1, ax_idx)
            ax.plot(lrs_ms, marker="*")
            ax.set_title(title)
            ax.set_xlim(0, len(lrs_ms))  # n_steps
            ax.set_xlabel("step")
            ax.set_ylabel("lr")
        plt.tight_layout()

    # Compare the difference between cosine_annealing_lr and cosine_annealing_warm_restarts_lr.
    plt.figure()
    lrs_ms = cosine_annealing_lr(5, 0.0, eta_max=1.0, steps_per_epoch=10, epochs=35)
    plt.plot(lrs_ms)
    lrs_ms = cosine_annealing_warm_restarts_lr(5, 2, 0.0, eta_max=1.0, steps_per_epoch=10, epochs=35)
    plt.plot(lrs_ms)
    plt.xlabel("step")
    plt.ylabel("lr")
    plt.legend(["cosine_annealing_lr", "cosine_annealing_warm_restarts_lr"], loc="best")
    plt.title("cosine_annealing_lr vs. cosine_annealing_warm_restarts_lr")
    plt.show()
# fmt: on
