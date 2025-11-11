#!/bin/bash
DATASET_NAME="CUHK-PEDES"

CUDA_VISIBLE_DEVICES=3 \
/data1/baiyang/anaconda/envs/bear_irra/bin/python test.py \
--config_file logs/CUHK-PEDES/20251111_150459_irra_sanity/configs.yaml \
--mask_noise_at_test \
--mask_topk 5 \
--mask_prob_thresh 0.5 \
--mask_max_ratio 0.3 \
--mask_max_tokens 12 