#!/bin/bash
# 正确的 Bash 环境变量写法
export NLTK_DATA="$PWD/nltk_data"
mkdir -p "$NLTK_DATA"
DATASET_NAME="CUHK-PEDES"

CUDA_VISIBLE_DEVICES=3 \
/data1/baiyang/anaconda/envs/bear_irra/bin/python train.py \
--name irra_v_mask_form_real_and_no_noisydetection \
--img_aug \
--batch_size 64 \
--dataset_name $DATASET_NAME \
--loss_names 'sdm+id+itc' \
--val_dataset test \
--num_epoch 60 \
--reid_raw 'reid_raw_0.2.json' \
--noisy_train_json './data/CUHK-PEDES/bear_noisy_data/reid_rw_with_train_noisy_33821_20638_10297_3370/train_reid_rw_all.json' \
--test_noisy_json './data/CUHK-PEDES/bear_noisy_data/reid_rw_with_test_noisy_3007_1906_944_299/test_reid_rw_all.json' \
--align_start_epoch 9 \
--itc_noisy_weight 0.5 \
--itc_mask_weight 1 \
--use_clean_for_retrieval \
--disable_consistency_loss \
--mask_test_start_epoch 17 \
--eval_impl mask_from_real \
--mask_noise_at_test \

# --noise_loss_weight 0.1 \
# --noise_detection \
# --noise_start_epoch 9 \
# --noise_warmup_epochs 8 \
# --eval_noise_acc_disable
# 目前不启用动态类别权重调整，关闭一致性损失