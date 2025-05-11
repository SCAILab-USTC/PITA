# optimizer_utils.py
import numpy as np
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, StepLR, LambdaLR, CosineAnnealingWarmRestarts, CyclicLR

def get_optimizer_and_scheduler(args, model, awl, train_loader):
    # Optimizer setup
    if args.opt.lower() == 'adam':
        optimizer = Adam([
            {"params": model.parameters(), "lr": args.lr, "betas": (args.beta1, args.beta2), "weight_decay": 1e-6},
            {"params": awl.parameters(), "lr": args.lr, "betas": (args.beta1, args.beta2), "weight_decay": 1e-6}
        ])
    elif args.opt.lower() == 'adamw':
        optimizer = AdamW([
            {"params": model.parameters(), "lr": args.lr, "betas": (args.beta1, args.beta2), "weight_decay": 1e-6},
            {"params": awl.parameters(), "lr": args.lr, "betas": (args.beta1, args.beta2), "weight_decay": 1e-6}
        ])
    else:
        raise NotImplementedError(f"Optimizer {args.opt} not supported.")

    # Scheduler setup
    if args.lr_method == 'cycle':
        print('Using cycle learning rate schedule')
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr,
            div_factor=1e4,
            pct_start=(args.warmup_epochs / args.epochs),
            final_div_factor=1e4,
            steps_per_epoch=len(train_loader),
            epochs=args.epochs
        )
    elif args.lr_method == 'step':
        print('Using step learning rate schedule')
        scheduler = StepLR(optimizer, step_size=args.step_size * len(train_loader), gamma=args.step_gamma)
    elif args.lr_method == 'warmup':
        print('Using warmup learning rate schedule')
        scheduler = LambdaLR(
            optimizer,
            lambda steps: min(
                (steps + 1) / (args.warmup_epochs * len(train_loader)),
                np.power(args.warmup_epochs * len(train_loader) / float(steps + 1), 0.5)
            )
        )
    elif args.lr_method == 'linear':
        print('Using linear learning rate schedule')
        scheduler = LambdaLR(
            optimizer,
            lambda steps: (1 - steps / (args.epochs * len(train_loader)))
        )
    elif args.lr_method == 'restart':
        print('Using cosine annealing with restarts')
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=len(train_loader) * args.step_size,
            eta_min=0.
        )
    elif args.lr_method == 'cyclic':
        print('Using cyclic LR')
        scheduler = CyclicLR(
            optimizer,
            base_lr=1e-5,
            max_lr=1e-3,
            step_size_up=args.step_size * len(train_loader),
            mode='triangular2',
            cycle_momentum=False
        )
    elif args.lr_method == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)

    else:
        raise NotImplementedError(f"LR method {args.lr_method} not supported.")

    return optimizer, scheduler
