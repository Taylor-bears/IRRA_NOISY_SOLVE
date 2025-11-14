#!/bin/bash
DATASET_NAME="CUHK-PEDES"

CUDA_VISIBLE_DEVICES=3 \
/data1/baiyang/anaconda/envs/bear_irra/bin/python train.py \
--name irra_v1.4 \
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
--noise_loss_weight 0.2 \
--noise_start_epoch 8 \
--noise_warmup_epochs 8 \
--use_clean_for_retrieval \
--consistency_loss_weight 0.05 \
--consistency_start_epoch 8 \
--consistency_warmup_epochs 8 \
--eval_impl extended \
--mask_noise_at_test \
--mask_strategy soft \
--noise_ctx topk_vote \
--mask_topk 5 \
--mask_prob_thresh 0.55 \
--mask_max_ratio 0.30 \
--mask_max_tokens 12