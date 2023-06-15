"""Scheduler Factory"""
from .dynamic_lr import (
    cosine_decay_lr,
    cosine_decay_refined_lr,
    dynamic_lr,
    exponential_lr,
    exponential_refined_lr,
    linear_lr,
    linear_refined_lr,
    multi_step_lr,
    polynomial_lr,
    polynomial_refined_lr,
    step_lr,
)

__all__ = ["create_scheduler"]


def create_scheduler(
    steps_per_epoch: int,
    scheduler: str = "constant",
    lr: float = 0.01,
    min_lr: float = 1e-6,
    warmup_epochs: int = 3,
    warmup_factor: float = 0.0,
    decay_epochs: int = 10,
    decay_rate: float = 0.9,
    milestones: list = None,
    num_epochs: int = 200,
    num_cycles: int = 1,
    cycle_decay: float = 1.0,
    lr_epoch_stair: bool = False,
):
    r"""Creates learning rate scheduler by name.

    Args:
        steps_per_epoch: number of steps per epoch.
        scheduler: scheduler name like 'constant', 'cosine_decay', 'step_decay',
            'exponential_decay', 'polynomial_decay', 'multi_step_decay'. Default: 'constant'.
        lr: learning rate value. Default: 0.01.
        min_lr: lower lr bound for 'cosine_decay' schedulers. Default: 1e-6.
        warmup_epochs: epochs to warmup LR, if scheduler supports. Default: 3.
        warmup_factor: the warmup phase of scheduler is a linearly increasing lr,
            the beginning factor is `warmup_factor`, i.e., the lr of the first step/epoch is lr*warmup_factor,
            and the ending lr in the warmup phase is lr. Default: 0.0
        decay_epochs: for 'cosine_decay' schedulers, decay LR to min_lr in `decay_epochs`.
            For 'step_decay' scheduler, decay LR by a factor of `decay_rate` every `decay_epochs`. Default: 10.
        decay_rate: LR decay rate (default: 0.9)
        milestones: list of epoch milestones for 'multi_step_decay' scheduler. Must be increasing.
        num_epochs: number of total epochs.
        lr_epoch_stair: If True, LR will be updated in the beginning of each new epoch
            and the LR will be consistent for each batch in one epoch.
            Otherwise, learning rate will be updated dynamically in each step. (default=False)
    Returns:
        Cell object for computing LR with input of current global steps
    """
    # check params
    if milestones is None:
        milestones = []

    if warmup_epochs + decay_epochs > num_epochs:
        print("[WARNING]: warmup_epochs + decay_epochs > num_epochs. Please check and reduce decay_epochs!")

    # lr warmup phase
    warmup_lr_scheduler = []
    if warmup_epochs > 0:
        if warmup_factor == 0 and lr_epoch_stair:
            print(
                "[WARNING]: The warmup factor is set to 0, lr of 0-th epoch is always zero! " "Recommend value is 0.01."
            )
        warmup_func = linear_lr if lr_epoch_stair else linear_refined_lr
        warmup_lr_scheduler = warmup_func(
            start_factor=warmup_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
            lr=lr,
            steps_per_epoch=steps_per_epoch,
            epochs=warmup_epochs,
        )

    # lr decay phase
    main_epochs = num_epochs - warmup_epochs
    if scheduler in ["cosine_decay", "warmup_cosine_decay"]:
        cosine_func = cosine_decay_lr if lr_epoch_stair else cosine_decay_refined_lr
        main_lr_scheduler = cosine_func(
            decay_epochs=decay_epochs,
            eta_min=min_lr,
            eta_max=lr,
            steps_per_epoch=steps_per_epoch,
            epochs=main_epochs,
            num_cycles=num_cycles,
            cycle_decay=cycle_decay,
        )
    elif scheduler == "exponential_decay":
        exponential_func = exponential_lr if lr_epoch_stair else exponential_refined_lr
        main_lr_scheduler = exponential_func(
            gamma=decay_rate, lr=lr, steps_per_epoch=steps_per_epoch, epochs=main_epochs
        )
    elif scheduler == "polynomial_decay":
        polynomial_func = polynomial_lr if lr_epoch_stair else polynomial_refined_lr
        main_lr_scheduler = polynomial_func(
            total_iters=main_epochs, power=decay_rate, lr=lr, steps_per_epoch=steps_per_epoch, epochs=main_epochs
        )
    elif scheduler == "dynamic_lr":
        main_lr_scheduler = dynamic_lr(lr, min_lr, num_epochs, steps_per_epoch)
    elif scheduler == "step_decay":
        main_lr_scheduler = step_lr(
            step_size=decay_epochs, gamma=decay_rate, lr=lr, steps_per_epoch=steps_per_epoch, epochs=main_epochs
        )
    elif scheduler == "multi_step_decay":
        main_lr_scheduler = multi_step_lr(
            milestones=milestones, gamma=decay_rate, lr=lr, steps_per_epoch=steps_per_epoch, epochs=main_epochs
        )
    elif scheduler == "constant":
        main_lr_scheduler = [lr for _ in range(steps_per_epoch * main_epochs)]
    else:
        raise ValueError(f"Invalid scheduler: {scheduler}")

    # combine
    lr_scheduler = warmup_lr_scheduler + main_lr_scheduler

    return lr_scheduler
