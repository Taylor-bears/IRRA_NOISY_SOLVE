import torch

from .lr_scheduler import LRSchedulerWithWarmup


# 构建优化器（optimizer.step更新参数）
def build_optimizer(args, model):
    params = []

    print(f'Using {args.lr_factor} times learning rate for random init module ')
    
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = args.lr
        weight_decay = args.weight_decay

        # 不同层次的学习率不同
        # 对跨模态模块使用更大的学习率
        if "cross" in key:
            # use large learning rate for random initialized cross modal module
            lr =  args.lr * args.lr_factor # default 5.0
        # 对偏置项使用不同的学习率和权重衰减
        if "bias" in key:
            lr = args.lr * args.bias_lr_factor
            weight_decay = args.weight_decay_bias
        # 对分类头和MLM头使用更大的学习率
        if "classifier" in key or "mlm_head" in key:
            lr = args.lr * args.lr_factor
        
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    # 根据配置选择优化器类型（优化器是调用已知的，我们只需要传参，负责​​参数更新规则​​，比如梯度下降）
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=args.momentum
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            params,
            lr=args.lr,
            betas=(args.alpha, args.beta),
            eps=1e-3,
        )
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            params,
            lr=args.lr,
            betas=(args.alpha, args.beta),
            eps=1e-8,
        )
    else:
        NotImplementedError

    return optimizer


# 构建学习率调度器（schedule.step调整优化器中的学习率，也就是更新的步长）
def build_lr_scheduler(args, optimizer):
    return LRSchedulerWithWarmup(
        optimizer,
        milestones=args.milestones,
        gamma=args.gamma,
        warmup_factor=args.warmup_factor,
        warmup_epochs=args.warmup_epochs,
        warmup_method=args.warmup_method,
        total_epochs=args.num_epoch,
        mode=args.lrscheduler,
        target_lr=args.target_lr,
        power=args.power,
    )
