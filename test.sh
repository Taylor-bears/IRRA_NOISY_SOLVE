#!/bin/bash
DATASET_NAME="CUHK-PEDES"

CUDA_VISIBLE_DEVICES=3 \
/data1/baiyang/anaconda/envs/bear_irra/bin/python test.py \
--config_file logs/CUHK-PEDES/20251111_150459_irra_sanity/configs.yaml \
--mask_noise_at_test \
--mask_topk 5 \
--mask_prob_thresh 0.69 \
--mask_max_ratio 0.3 \
--mask_max_tokens 12 \
--mask_strategy soft \
--mask_soft_alpha_cap 0.35 \
--noise_ctx topk_mean \
--enable_attribute_filter \
--attribute_vocab_path "utils/attribute_vocab.txt" 

这里的部分参数，询问一下copilot怎么取比较合适，给他看一些例子！！！！！！！！！

目前enable_attribute_filter这个设置还是有很大的问题，因为里面的属性词并不是特别全面！！！！！！！！！！！！！