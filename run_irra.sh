#!/bin/bash
DATASET_NAME="CUHK-PEDES"

CUDA_VISIBLE_DEVICES=3 \
/data1/baiyang/anaconda/envs/bear_irra/bin/python train.py \
--name irra_v1.2 \
--img_aug \
--batch_size 64 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'sdm+itc+id' \
--num_epoch 60 \
--noise_detection \
--noisy_train_json './data/CUHK-PEDES/bear_noisy_data/reid_rw_with_train_noisy_33821_20638_10297_3370/train_reid_rw_all.json' \
--noise_loss_weight 0.3 \
--noise_start_epoch 6 \
--noise_warmup_epochs 6 \
--use_clean_for_retrieval \
--consistency_loss_weight 0.05 \
--consistency_start_epoch 6 \
--consistency_warmup_epochs 6

# --loss_names 'sdm+mlm+id' 暂时先关了mlm