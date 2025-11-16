#!/bin/bash
DATASET_NAME="CUHK-PEDES"

CUDA_VISIBLE_DEVICES=3 \
/data1/baiyang/anaconda/envs/bear_irra/bin/python train.py \
--name irra_v1.6 \
--img_aug \
--batch_size 64 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'sdm+mlm+id' \
--val_dataset test \
--num_epoch 60 \
--reid_raw 'reid_raw_0.2.json' \
--noise_detection \
--noisy_train_json './data/CUHK-PEDES/bear_noisy_data/reid_rw_with_train_noisy_33821_20638_10297_3370/train_reid_rw_all.json' \
--test_noisy_json './data/CUHK-PEDES/bear_noisy_data/reid_rw_with_test_noisy_3007_1906_944_299/test_reid_rw_all.json' \
--noise_loss_weight 0.2 \
--noise_start_epoch 8 \
--noise_warmup_epochs 8 \
--use_clean_for_retrieval \
--disable_consistency_loss \
--consistency_loss_weight 0.02 \
--consistency_start_epoch 8 \
--consistency_warmup_epochs 8 \
--mask_test_start_epoch 17 \
--eval_impl extended \
--mask_noise_at_test \
--mask_strategy hard \
--noise_ctx topk_vote \
--mask_topk 32 \
--mask_prob_thresh 0.5 \
--mask_max_ratio 0.30 \
--mask_max_tokens 6 \
--enable_attribute_filter

# 目前不启用动态类别权重调整，关闭一致性损失