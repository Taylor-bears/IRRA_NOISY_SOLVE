# IRRA 模型架构详解

## 整体框架概览

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         IRRA Model Architecture                                  │
│   Cross-Modal Implicit Relation Reasoning and Aligning (CVPR 2023)             │
└─────────────────────────────────────────────────────────────────────────────────┘

╔═══════════════════════════════════════════════════════════════════════════════╗
║                            TRAINING STAGE                                      ║
╚═══════════════════════════════════════════════════════════════════════════════╝

┌──────────────┐          ┌──────────────────┐         ┌──────────────────┐
│  Image Input │          │ Vision Transformer│         │  Image Features  │
│  (B,3,384,   │  ───────>│   (ViT-B/16)     │────────>│   (B, 512)       │
│   128)       │          │  CLIP Pretrained  │         │  [CLS Token]     │
└──────────────┘          └──────────────────┘         └──────────────────┘
                                                                 │
                                                                 │
┌──────────────┐          ┌──────────────────┐         ┌───────▼──────────┐
│  Text Input  │          │ Text Transformer │         │  Text Features   │
│  (77 tokens) │  ───────>│  (12 Layers)     │────────>│   (B, 512)       │
│  Caption     │          │  CLIP Pretrained  │         │  [EOS Token]     │
└──────────────┘          └──────────────────┘         └───────┬──────────┘
                                                                 │
                                                                 │
╔════════════════════════════════════════════════════════════════════════════════╗
║                          MULTI-TASK LEARNING                                   ║
╚════════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────────┐
│ 1. MLM Module (Masked Language Modeling)                                       │
│    ┌────────────────────────────────────────────────────────┐                 │
│    │ • Masked Text Input → Text Encoder                     │                 │
│    │ • Cross-Modal Attention (Text → Image)                 │                 │
│    │ • 4-Layer Cross-Modal Transformer                      │                 │
│    │ • MLM Prediction Head: Dense → GELU → LN → FC(vocab)  │                 │
│    │ • Loss: CrossEntropy(predicted_tokens, original_tokens)│                 │
│    └────────────────────────────────────────────────────────┘                 │
│                              ↓                                                  │
│                        MLM Loss (L_mlm)                                         │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│ 2. SDM Module (Similarity Distribution Matching)                               │
│    ┌────────────────────────────────────────────────────────┐                 │
│    │ • Normalize Image & Text Features (L2 norm)            │                 │
│    │ • Compute Cosine Similarity Matrix: S = T_norm @ I_norm^T│              │
│    │ • Temperature Scaling: S' = S * logit_scale            │                 │
│    │ • Construct Ground Truth Distribution from PIDs        │                 │
│    │ • Loss: KL Divergence(softmax(S'), GT_distribution)   │                 │
│    └────────────────────────────────────────────────────────┘                 │
│                              ↓                                                  │
│                        SDM Loss (L_sdm)                                         │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│ 3. ITC Module (Image-Text Contrastive) [Optional]                             │
│    ┌────────────────────────────────────────────────────────┐                 │
│    │ • Standard CLIP Contrastive Loss (InfoNCE)             │                 │
│    │ • Positive pairs: matching image-text pairs            │                 │
│    │ • Loss: (CrossEntropy_i2t + CrossEntropy_t2i) / 2     │                 │
│    └────────────────────────────────────────────────────────┘                 │
│                              ↓                                                  │
│                        ITC Loss (L_itc)                                         │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│ 4. ID Module (Instance Recognition)                                            │
│    ┌────────────────────────────────────────────────────────┐                 │
│    │ • Linear Classifier: FC(512 → num_classes)             │                 │
│    │ • Separate Classification for Image & Text             │                 │
│    │ • Loss: (CE_image + CE_text) / 2                       │                 │
│    └────────────────────────────────────────────────────────┘                 │
│                              ↓                                                  │
│                        ID Loss (L_id)                                           │
└─────────────────────────────────────────────────────────────────────────────────┘

                                    ↓
                ┌──────────────────────────────────────┐
                │      Total Loss (Optimization)       │
                │  L_total = L_sdm + L_mlm + L_id      │
                │  (可配置不同的loss组合)                │
                └──────────────────────────────────────┘
                                    ↓
                            Backward & Update


╔═══════════════════════════════════════════════════════════════════════════════╗
║                           INFERENCE STAGE                                      ║
╚═══════════════════════════════════════════════════════════════════════════════╝

┌──────────────┐       ┌──────────────┐       ┌──────────────┐       ┌─────────────┐
│ Gallery      │       │   Encode     │       │  Normalize   │       │   Feature   │
│ Images       │ ─────>│   (ViT)      │──────>│  (L2 norm)   │──────>│   Gallery   │
│ (N images)   │       │              │       │              │       │   (N, 512)  │
└──────────────┘       └──────────────┘       └──────────────┘       └──────┬──────┘
                                                                              │
                                                                              │
┌──────────────┐       ┌──────────────┐       ┌──────────────┐       ┌──────▼──────┐
│ Query        │       │   Encode     │       │  Normalize   │       │   Compute   │
│ Text         │ ─────>│ (Text Trans) │──────>│  (L2 norm)   │──────>│ Similarity  │
│ (1 caption)  │       │              │       │              │       │   Matrix    │
└──────────────┘       └──────────────┘       └──────────────┘       └──────┬──────┘
                                                                              │
                                                                              │
                                                                     ┌────────▼────────┐
                                                                     │   Ranking &     │
                                                                     │   Retrieval     │
                                                                     │   (Top-K)       │
                                                                     └─────────────────┘
                                                                              │
                                                                              ▼
                                                          ┌──────────────────────────────┐
                                                          │  Evaluation Metrics:         │
                                                          │  • Rank-1, Rank-5, Rank-10  │
                                                          │  • mAP (mean Average Prec.) │
                                                          │  • mINP (mean Inverse Neg.) │
                                                          └──────────────────────────────┘
```

## 模型组件详解

### 1. 基础编码器 (CLIP Backbone)
- **Vision Transformer (ViT-B/16)**
  - 输入尺寸: 384×128 (Person Re-ID定制)
  - Patch size: 16×16
  - Stride: 16
  - 输出: 提取 [CLS] token作为图像表示
  
- **Text Transformer**
  - 12层Transformer
  - 输入: 77个token (CLIP标准)
  - 输出: 提取 [EOS] token作为文本表示

### 2. Cross-Modal Transformer (MLM专用)
```
输入: Masked Text Features + Image Features
    ↓
Cross-Modal Attention: Q=Text, K=Image, V=Image
    ↓
4层Transformer Encoder
    ↓
MLM Prediction Head
    ↓
预测被mask的token
```

### 3. 数据流程

**训练数据格式:**
```python
batch = {
    'images': Tensor (B, 3, 384, 128),      # 图像
    'caption_ids': Tensor (B, 77),          # 文本token
    'mlm_ids': Tensor (B, 77),              # 带mask的文本
    'mlm_labels': Tensor (B, 77),           # MLM标签
    'pids': Tensor (B,),                    # 行人ID
    'image_ids': Tensor (B,)                # 图像ID
}
```

**特征提取:**
```python
# 图像编码
image_features = vision_encoder(images)        # (B, 197, 512)
i_feat = image_features[:, 0, :]              # (B, 512) - CLS token

# 文本编码
text_features = text_encoder(caption_ids)      # (B, 77, 512)
t_feat = text_features[arange, eot_index]     # (B, 512) - EOS token
```

### 4. 损失函数细节

**SDM Loss (核心创新):**
```python
# 1. 归一化特征
i_norm = i_feat / ||i_feat||
t_norm = t_feat / ||t_feat||

# 2. 计算相似度矩阵
similarity = t_norm @ i_norm.T  # (B, B)

# 3. 温度缩放
logits = similarity * temperature  # temperature = 1/0.02

# 4. 构建标签分布
labels[i,j] = 1.0  if pid[i] == pid[j]
labels[i,j] = 0.3  if pid[i] == pid[j] and img_id[i] != img_id[j]  # 同人不同图
labels[i,j] = 0.0  otherwise

# 5. KL散度
loss = KL_div(softmax(logits), labels_normalized)
```

**MLM Loss:**
```python
# 15%的token被mask
# - 80%替换为[MASK]
# - 10%替换为随机token
# - 10%保持不变
loss = CrossEntropy(predicted_logits, original_tokens)
```

**ID Loss:**
```python
img_logits = classifier(i_feat)  # (B, num_ids)
txt_logits = classifier(t_feat)  # (B, num_ids)
loss = (CE(img_logits, pids) + CE(txt_logits, pids)) / 2
```

### 5. 训练配置

**优化器设置:**
```python
# 不同模块使用不同学习率
- Base CLIP: lr = 1e-5
- Cross-Modal Module: lr = 5e-5 (5倍)
- Classifier/MLM Head: lr = 5e-5 (5倍)
- Bias: lr = 2e-5 (2倍)
```

**学习率调度:**
```python
# Warmup: 5 epochs (linear from 0.1*lr to lr)
# Main: Cosine annealing (60 epochs)
```

**数据采样:**
```python
# Identity Sampling (推荐)
batch_size = 64
num_instance = 4
→ 每个batch包含16个不同ID，每个ID 4个样本
```

### 6. 评估指标

- **Rank-K**: 前K个检索结果中是否包含正确匹配
- **mAP**: 平均精度均值 (考虑所有正确匹配的排序)
- **mINP**: 平均逆负惩罚 (更关注困难样本)

### 7. 性能对比 (CUHK-PEDES)

| Method   | Rank-1    | Rank-5    | Rank-10   | mAP       |
| -------- | --------- | --------- | --------- | --------- |
| CLIP     | 68.19     | 86.47     | 91.47     | 61.12     |
| **IRRA** | **73.38** | **89.93** | **93.71** | **66.13** |

## 关键创新点

1. **SDM损失**: 使用KL散度匹配相似度分布，比对比损失更柔和
2. **Cross-Modal Transformer**: 显式建模图文交互
3. **多任务学习**: MLM增强文本理解，ID增强判别能力
4. **完整CLIP**: 使用全部特征而非仅[CLS]/[EOS]

## 代码结构对应

```
model/
  ├── clip_model.py       → Vision/Text Encoder (CLIP基础)
  ├── build.py            → IRRA模型主体 (多任务模块)
  └── objectives.py       → 损失函数实现

datasets/
  ├── build.py            → 数据加载器
  ├── cuhkpedes.py        → CUHK-PEDES数据集
  └── bases.py            → 数据集基类 (MLM mask)

processor/
  └── processor.py        → 训练/推理循环

solver/
  ├── build.py            → 优化器构建
  └── lr_scheduler.py     → 学习率调度

utils/
  ├── metrics.py          → 评估指标计算
  └── checkpoint.py       → 模型保存加载
```

## 使用示例

**训练:**
```bash
python train.py \
  --name irra \
  --img_aug \
  --batch_size 64 \
  --MLM \
  --loss_names 'sdm+mlm+id' \
  --dataset_name 'CUHK-PEDES' \
  --num_epoch 60
```

**测试:**
```bash
python test.py --config_file 'logs/CUHK-PEDES/model/configs.yaml'
```
