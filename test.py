from prettytable import PrettyTable
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import numpy as np
import time
import os.path as op

from datasets import build_dataloader
from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
from utils.metrics import Evaluator
import argparse
from utils.iotools import load_train_configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IRRA Test")
    parser.add_argument("--config_file", default="logs/CUHK-PEDES/irra/configs.yaml")
    # 允许在测试阶段通过命令行覆盖部分与掩码相关的参数；不提供则沿用训练时保存的配置
    parser.add_argument(
        "--mask_noise_at_test", dest="mask_noise_at_test", action="store_true"
    )
    parser.add_argument(
        "--no-mask_noise_at_test", dest="mask_noise_at_test", action="store_false"
    )
    parser.add_argument("--mask_topk", type=int, default=None)
    parser.add_argument("--mask_prob_thresh", type=float, default=None)
    parser.add_argument("--mask_max_ratio", type=float, default=None)
    parser.add_argument("--mask_max_tokens", type=int, default=None)
    parser.set_defaults(mask_noise_at_test=None)
    cli_args = parser.parse_args()
    args = load_train_configs(cli_args.config_file)

    args.training = False
    # 覆盖配置：仅当命令行提供了值时
    if cli_args.mask_noise_at_test is not None:
        args.mask_noise_at_test = bool(cli_args.mask_noise_at_test)
    if cli_args.mask_topk is not None:
        args.mask_topk = int(cli_args.mask_topk)
    if cli_args.mask_prob_thresh is not None:
        args.mask_prob_thresh = float(cli_args.mask_prob_thresh)
    if cli_args.mask_max_ratio is not None:
        args.mask_max_ratio = float(cli_args.mask_max_ratio)
    if cli_args.mask_max_tokens is not None:
        args.mask_max_tokens = int(cli_args.mask_max_tokens)

    logger = setup_logger("IRRA", save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = "cuda"

    test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes=num_classes)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(args.output_dir, "best.pth"))
    model.to(device)
    do_inference(model, test_img_loader, test_txt_loader)