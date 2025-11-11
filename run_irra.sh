#!/bin/bash
DATASET_NAME="CUHK-PEDES"

CUDA_VISIBLE_DEVICES=3 \
/data1/baiyang/anaconda/envs/bear_irra/bin/python train.py \
--name irra_sanity \
--img_aug \
--batch_size 16 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'sdm+id' \
--num_epoch 1 \
--noise_detection \
--noisy_train_json './data/CUHK-PEDES/bear_noisy_data/reid_rw_with_train_noisy_33821_20638_10297_3370/train_reid_rw_all.json' \

# --loss_names 'sdm+mlm+id' 暂时先关了mlm