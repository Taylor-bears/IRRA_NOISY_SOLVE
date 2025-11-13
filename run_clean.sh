#!/bin/bash
DATASET_NAME="CUHK-PEDES"

CUDA_VISIBLE_DEVICES=3 \
/data1/baiyang/anaconda/envs/bear_irra/bin/python train.py \
--name irra_v_clean \
--img_aug \
--batch_size 64 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'sdm+mlm+id' \
--num_epoch 60 \
--use_clean_for_retrieval \
--eval_impl baseline
# --loss_names 'sdm+mlm+id' 暂时先关了mlm