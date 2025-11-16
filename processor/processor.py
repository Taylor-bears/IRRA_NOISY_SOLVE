import logging
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator as EvaluatorExtended
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
import os

# 模型训练循环，包含前向传播、损失计算、反向传播、日志记录和周期性验证。
def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):
    # 初始化参数
    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0
    # 初始化日志和监控工具
    logger = logging.getLogger("IRRA.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "sdm_loss": AverageMeter(),
        "itc_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),
        "noise_loss": AverageMeter(),
        "consistency_loss": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter(),
        "noise_acc": AverageMeter()
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0

    # 训练循环
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()

        # 全局写时钟：每个epoch开始时更新，供数据集控制“前N轮”动态词库写入
        try:
            clock_path = getattr(args, 'attribute_vocab_path', 'utils/attribute_vocab.txt') + '.clock'
            with open(clock_path, 'w', encoding='utf-8') as f:
                f.write(str(epoch))
        except Exception:
            pass

        # 遍历训练数据
        for n_iter, batch in enumerate(train_loader): # n_iter是当前epoch中的迭代次数
            # batch的数据放到GPU上
            batch = {k: v.to(device) for k, v in batch.items()}
            # 前向传播，返回损失和指标
            ret = model(batch)
            # 动态warmup噪声损失：支持延后起始与线性预热
            warmup_epochs = getattr(args, 'noise_warmup_epochs', 0) # 表示噪声损失预热多少轮
            # 表示什么时候开始开启算入噪声损失，目的是为了恢复接近原 IRRA 检索水平，建立稳定的跨模态空间
            start_epoch_noise = getattr(args, 'noise_start_epoch', 0)
            if 'noise_loss' in ret:
                if warmup_epochs <= 0:
                    # 若不预热但设置了延后起始，则在起始轮前禁用
                    if start_epoch_noise and epoch < start_epoch_noise:
                        ret['noise_loss'] = ret['noise_loss'] * 0.0
                else:
                    # 在起始轮之前禁用；起始轮到起始轮+warmup线性提升到1
                    if start_epoch_noise and epoch < start_epoch_noise:
                        alpha = 0.0
                    else:
                        # 计算从起始轮开始的progress
                        base_epoch = start_epoch_noise if start_epoch_noise else 1
                        progress = (epoch - base_epoch) + (n_iter + 1) / max(1, len(train_loader))
                        alpha = min(1.0, max(0.0, progress / max(1e-6, warmup_epochs)))
                    ret['noise_loss'] = ret['noise_loss'] * alpha

            # 一致性损失：支持“起始轮数 + 线性预热”
            cons_warmup = getattr(args, 'consistency_warmup_epochs', 0)
            cons_start = getattr(args, 'consistency_start_epoch', 0)
            if 'consistency_loss' in ret:
                if cons_warmup <= 0:
                    # 无预热则只应用起始轮门控
                    if cons_start and epoch < cons_start:
                        ret['consistency_loss'] = ret['consistency_loss'] * 0.0
                else:
                    if cons_start and epoch < cons_start:
                        alpha = 0.0
                    else:
                        base_epoch = cons_start if cons_start else 1
                        progress = (epoch - base_epoch) + (n_iter + 1) / max(1, len(train_loader))
                        alpha = min(1.0, max(0.0, progress / max(1e-6, cons_warmup)))
                    ret['consistency_loss'] = ret['consistency_loss'] * alpha
            # 计算总损失
            total_loss = sum([v for k, v in ret.items() if "loss" in k])
            # 更新各项指标
            batch_size = batch['images'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            meters['sdm_loss'].update(ret.get('sdm_loss', 0), batch_size)
            meters['itc_loss'].update(ret.get('itc_loss', 0), batch_size)
            meters['id_loss'].update(ret.get('id_loss', 0), batch_size)
            meters['mlm_loss'].update(ret.get('mlm_loss', 0), batch_size)
            meters['noise_loss'].update(ret.get('noise_loss', 0), batch_size)
            meters['consistency_loss'].update(ret.get('consistency_loss', 0), batch_size)

            meters['img_acc'].update(ret.get('img_acc', 0), batch_size)
            meters['txt_acc'].update(ret.get('txt_acc', 0), batch_size)
            meters['mlm_acc'].update(ret.get('mlm_acc', 0), 1)
            meters['noise_acc'].update(ret.get('noise_acc', 0), 1)
            # 反向传播优化
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()
            # 打印日志
            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
        # 记录学习率和温度参数到TensorBoard
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)

        # 更新学习率
        scheduler.step()
        # 打印epoch完成信息
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        # 调用utils进行模型评估
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                # 基于 --mask_test_start_epoch 进行测试期噪声掩码的延迟启用
                mask_noise_flag = getattr(args, 'mask_noise_at_test', False)
                start_ep_gate = getattr(args, 'mask_test_start_epoch', 0)
                if mask_noise_flag:
                    if start_ep_gate > 0 and epoch < start_ep_gate:
                        evaluator.mask_noise = False
                        logger.info(f"[Eval Gate] epoch {epoch} < mask_test_start_epoch {start_ep_gate}: disable masking this eval")
                    else:
                        evaluator.mask_noise = True
                        logger.info(f"[Eval Gate] masking enabled at epoch {epoch} (start_gate={start_ep_gate})")
                else:
                    evaluator.mask_noise = False
                    logger.info(f"[Eval Gate] mask_noise_at_test flag False: masking disabled")
                # 切换模型为评估模式并计算指标
                if args.distributed:
                    top1 = evaluator.eval(model.module.eval())
                else:
                    top1 = evaluator.eval(model.eval())

                torch.cuda.empty_cache()
                # 保存最佳模型
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)
    # 训练结束，打印最佳结果
    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")


# 模型测试阶段的推理流程，计算检索指标
def do_inference(model, test_img_loader, test_txt_loader):

    logger = logging.getLogger("IRRA.test")
    logger.info("Enter inferencing")
    # 初始化评估器（计算Top-1、mAP等指标）
    # 按需启用测试时噪声掩码（从模型参数中读取）
    mask_noise = getattr(getattr(model, 'args', object()), 'mask_noise_at_test', True)
    # 根据实现选择评估器
    try:
        from utils.options import get_args as _get_args_for_eval
        args_obj = getattr(model, 'args', None)
        eval_impl = getattr(args_obj, 'eval_impl', 'extended') if args_obj else 'extended'
    except Exception:
        eval_impl = 'extended'
    if eval_impl == 'baseline':
        from utils.metrics_baseline import Evaluator as EvaluatorBaseline
        evaluator = EvaluatorBaseline(test_img_loader, test_txt_loader)
    else:
        evaluator = EvaluatorExtended(test_img_loader, test_txt_loader, mask_noise=mask_noise)
    # 评估模型性能
    top1 = evaluator.eval(model.eval())