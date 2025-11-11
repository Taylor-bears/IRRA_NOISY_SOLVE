#!/bin/bash
DATASET_NAME="CUHK-PEDES"

CUDA_VISIBLE_DEVICES=0 \
python train.py \
--name irra \
--img_aug \
--batch_size 64 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'sdm+mlm+id' \
--num_epoch 60 \
--noise_detection True \
--noisy_train_json '\data\CUHK-PEDES\bear_noisy_data\reid_rw_with_train_noisy_33821_20638_10297_3370\train_reid_rw_all.json' \
--mask_noise_at_test True 