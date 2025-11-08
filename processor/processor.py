import logging
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable

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
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter()
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0

    # 训练循环
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()

        # 遍历训练数据
        for n_iter, batch in enumerate(train_loader):
            # batch的数据放到GPU上
            batch = {k: v.to(device) for k, v in batch.items()}
            # 前向传播，返回损失和指标
            ret = model(batch)
            # 计算总损失
            total_loss = sum([v for k, v in ret.items() if "loss" in k])
            # 更新各项指标
            batch_size = batch['images'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            meters['sdm_loss'].update(ret.get('sdm_loss', 0), batch_size)
            meters['itc_loss'].update(ret.get('itc_loss', 0), batch_size)
            meters['id_loss'].update(ret.get('id_loss', 0), batch_size)
            meters['mlm_loss'].update(ret.get('mlm_loss', 0), batch_size)

            meters['img_acc'].update(ret.get('img_acc', 0), batch_size)
            meters['txt_acc'].update(ret.get('txt_acc', 0), batch_size)
            meters['mlm_acc'].update(ret.get('mlm_acc', 0), 1)
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
    evaluator = Evaluator(test_img_loader, test_txt_loader)
    # 评估模型性能
    top1 = evaluator.eval(model.eval())