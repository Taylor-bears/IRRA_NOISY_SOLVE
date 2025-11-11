"""
IRRA模型架构可视化
使用matplotlib绘制IRRA (Cross-Modal Implicit Relation Reasoning and Aligning) 模型的整体架构图
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(20, 14))
ax.set_xlim(0, 20)
ax.set_ylim(0, 14)
ax.axis('off')

# 定义颜色方案
color_input = '#E8F4F8'
color_encoder = '#B8E6F0'
color_feature = '#FFE6CC'
color_loss = '#FFD9D9'
color_module = '#D4E6D4'
color_output = '#E6D4F0'

# ==================== 标题 ====================
ax.text(10, 13.5, 'IRRA Model Architecture', 
        ha='center', va='center', fontsize=24, fontweight='bold')
ax.text(10, 13, 'Cross-Modal Implicit Relation Reasoning and Aligning for Text-to-Image Person Retrieval', 
        ha='center', va='center', fontsize=12, style='italic')

# ==================== 输入层 ====================
# 图像输入
img_input = FancyBboxPatch((0.5, 10.5), 2.5, 1.2, 
                           boxstyle="round,pad=0.1", 
                           edgecolor='black', facecolor=color_input, linewidth=2)
ax.add_patch(img_input)
ax.text(1.75, 11.3, 'Image Input', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(1.75, 10.9, '(B, 3, 384, 128)', ha='center', va='center', fontsize=9)

# 文本输入
txt_input = FancyBboxPatch((0.5, 8.8), 2.5, 1.2, 
                           boxstyle="round,pad=0.1", 
                           edgecolor='black', facecolor=color_input, linewidth=2)
ax.add_patch(txt_input)
ax.text(1.75, 9.6, 'Text Input', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(1.75, 9.2, 'Caption (77 tokens)', ha='center', va='center', fontsize=9)

# ==================== CLIP编码器 ====================
# Vision Transformer
vit_box = FancyBboxPatch((4, 10.3), 3, 1.6, 
                         boxstyle="round,pad=0.1", 
                         edgecolor='blue', facecolor=color_encoder, linewidth=2.5)
ax.add_patch(vit_box)
ax.text(5.5, 11.4, 'Vision Transformer', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(5.5, 11, 'ViT-B/16', ha='center', va='center', fontsize=9)
ax.text(5.5, 10.7, '(Pretrained CLIP)', ha='center', va='center', fontsize=8, style='italic')

# Text Transformer
txt_transformer = FancyBboxPatch((4, 8.6), 3, 1.6, 
                                 boxstyle="round,pad=0.1", 
                                 edgecolor='blue', facecolor=color_encoder, linewidth=2.5)
ax.add_patch(txt_transformer)
ax.text(5.5, 9.7, 'Text Transformer', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(5.5, 9.3, '12-layer Transformer', ha='center', va='center', fontsize=9)
ax.text(5.5, 9, '(Pretrained CLIP)', ha='center', va='center', fontsize=8, style='italic')

# 箭头：输入到编码器
arrow1 = FancyArrowPatch((3, 11.1), (4, 11.1), 
                        arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
ax.add_artist(arrow1)
arrow2 = FancyArrowPatch((3, 9.4), (4, 9.4), 
                        arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
ax.add_artist(arrow2)

# ==================== 特征提取 ====================
# 图像特征
img_feat = FancyBboxPatch((8, 10.5), 2.5, 1.2, 
                          boxstyle="round,pad=0.1", 
                          edgecolor='orange', facecolor=color_feature, linewidth=2)
ax.add_patch(img_feat)
ax.text(9.25, 11.3, 'Image Features', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(9.25, 10.9, 'I_feat (B, 512)', ha='center', va='center', fontsize=9)

# 文本特征
txt_feat = FancyBboxPatch((8, 8.8), 2.5, 1.2, 
                          boxstyle="round,pad=0.1", 
                          edgecolor='orange', facecolor=color_feature, linewidth=2)
ax.add_patch(txt_feat)
ax.text(9.25, 9.6, 'Text Features', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(9.25, 9.2, 'T_feat (B, 512)', ha='center', va='center', fontsize=9)

# 箭头：编码器到特征
arrow3 = FancyArrowPatch((7, 11.1), (8, 11.1), 
                        arrowstyle='->', mutation_scale=20, linewidth=2, color='orange')
ax.add_artist(arrow3)
arrow4 = FancyArrowPatch((7, 9.4), (8, 9.4), 
                        arrowstyle='->', mutation_scale=20, linewidth=2, color='orange')
ax.add_artist(arrow4)

# ==================== 多任务模块 ====================
# MLM分支 (上方)
mlm_box = FancyBboxPatch((11.5, 10.2), 3.5, 2, 
                         boxstyle="round,pad=0.1", 
                         edgecolor='green', facecolor=color_module, linewidth=2.5)
ax.add_patch(mlm_box)
ax.text(13.25, 11.8, 'MLM Module', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(13.25, 11.4, 'Cross-Modal Transformer', ha='center', va='center', fontsize=9)
ax.text(13.25, 11.05, '• Cross Attention (T→I)', ha='center', va='center', fontsize=8)
ax.text(13.25, 10.75, '• 4-layer Transformer', ha='center', va='center', fontsize=8)
ax.text(13.25, 10.45, '• MLM Prediction Head', ha='center', va='center', fontsize=8)

# SDM/ITC分支 (中间)
sdm_box = FancyBboxPatch((11.5, 7.8), 3.5, 2, 
                         boxstyle="round,pad=0.1", 
                         edgecolor='purple', facecolor=color_module, linewidth=2.5)
ax.add_patch(sdm_box)
ax.text(13.25, 9.4, 'Cross-Modal Matching', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(13.25, 9, 'Similarity Computation', ha='center', va='center', fontsize=9)
ax.text(13.25, 8.65, '• Normalize Features', ha='center', va='center', fontsize=8)
ax.text(13.25, 8.35, '• Cosine Similarity', ha='center', va='center', fontsize=8)
ax.text(13.25, 8.05, '• Temperature Scaling', ha='center', va='center', fontsize=8)

# ID分类分支 (下方)
id_box = FancyBboxPatch((11.5, 5.5), 3.5, 2, 
                        boxstyle="round,pad=0.1", 
                        edgecolor='red', facecolor=color_module, linewidth=2.5)
ax.add_patch(id_box)
ax.text(13.25, 7.1, 'ID Classification', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(13.25, 6.7, 'Linear Classifier', ha='center', va='center', fontsize=9)
ax.text(13.25, 6.35, '• FC Layer (512→N_ids)', ha='center', va='center', fontsize=8)
ax.text(13.25, 6.05, '• Separate for I & T', ha='center', va='center', fontsize=8)
ax.text(13.25, 5.75, '• Instance Recognition', ha='center', va='center', fontsize=8)

# 箭头：特征到各模块
arrow5 = FancyArrowPatch((10.5, 11.1), (11.5, 11.5), 
                        arrowstyle='->', mutation_scale=20, linewidth=2, color='green')
ax.add_artist(arrow5)
arrow6 = FancyArrowPatch((10.5, 10.1), (11.5, 8.8), 
                        arrowstyle='->', mutation_scale=20, linewidth=2, color='purple')
ax.add_artist(arrow6)
arrow7 = FancyArrowPatch((10.5, 9.4), (11.5, 6.5), 
                        arrowstyle='->', mutation_scale=20, linewidth=2, color='red')
ax.add_artist(arrow7)

# ==================== 损失函数 ====================
# MLM Loss
mlm_loss = FancyBboxPatch((16, 10.8), 2, 0.8, 
                          boxstyle="round,pad=0.05", 
                          edgecolor='darkgreen', facecolor=color_loss, linewidth=2)
ax.add_patch(mlm_loss)
ax.text(17, 11.2, 'MLM Loss', ha='center', va='center', fontsize=10, fontweight='bold')

# SDM Loss
sdm_loss = FancyBboxPatch((16, 9.2), 2, 0.8, 
                          boxstyle="round,pad=0.05", 
                          edgecolor='darkviolet', facecolor=color_loss, linewidth=2)
ax.add_patch(sdm_loss)
ax.text(17, 9.6, 'SDM Loss', ha='center', va='center', fontsize=10, fontweight='bold')

# ITC Loss
itc_loss = FancyBboxPatch((16, 8.4), 2, 0.8, 
                          boxstyle="round,pad=0.05", 
                          edgecolor='darkviolet', facecolor=color_loss, linewidth=2)
ax.add_patch(itc_loss)
ax.text(17, 8.8, 'ITC Loss', ha='center', va='center', fontsize=10, fontweight='bold')

# ID Loss
id_loss = FancyBboxPatch((16, 6.3), 2, 0.8, 
                         boxstyle="round,pad=0.05", 
                         edgecolor='darkred', facecolor=color_loss, linewidth=2)
ax.add_patch(id_loss)
ax.text(17, 6.7, 'ID Loss', ha='center', va='center', fontsize=10, fontweight='bold')

# 箭头：模块到损失
arrow8 = FancyArrowPatch((15, 11.2), (16, 11.2), 
                        arrowstyle='->', mutation_scale=20, linewidth=2, color='darkgreen')
ax.add_artist(arrow8)
arrow9 = FancyArrowPatch((15, 9), (16, 9.6), 
                        arrowstyle='->', mutation_scale=20, linewidth=2, color='darkviolet')
ax.add_artist(arrow9)
arrow10 = FancyArrowPatch((15, 8.5), (16, 8.8), 
                         arrowstyle='->', mutation_scale=20, linewidth=2, color='darkviolet')
ax.add_artist(arrow10)
arrow11 = FancyArrowPatch((15, 6.5), (16, 6.7), 
                         arrowstyle='->', mutation_scale=20, linewidth=2, color='darkred')
ax.add_artist(arrow11)

# ==================== 总损失 ====================
total_loss = FancyBboxPatch((16.5, 4.5), 3, 1, 
                            boxstyle="round,pad=0.1", 
                            edgecolor='black', facecolor='#FFB6C1', linewidth=3)
ax.add_patch(total_loss)
ax.text(18, 5.2, 'Total Loss', ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(18, 4.8, 'L = L_SDM + L_MLM + L_ID', ha='center', va='center', fontsize=9)

# 箭头：各损失到总损失
arrow12 = FancyArrowPatch((17, 10.8), (17.5, 5.5), 
                         arrowstyle='->', mutation_scale=20, linewidth=1.5, 
                         color='black', linestyle='dashed')
ax.add_artist(arrow12)
arrow13 = FancyArrowPatch((17, 9.2), (17.7, 5.5), 
                         arrowstyle='->', mutation_scale=20, linewidth=1.5, 
                         color='black', linestyle='dashed')
ax.add_artist(arrow13)
arrow14 = FancyArrowPatch((17, 8.4), (17.9, 5.5), 
                         arrowstyle='->', mutation_scale=20, linewidth=1.5, 
                         color='black', linestyle='dashed')
ax.add_artist(arrow14)
arrow15 = FancyArrowPatch((17, 7.1), (18.2, 5.5), 
                         arrowstyle='->', mutation_scale=20, linewidth=1.5, 
                         color='black', linestyle='dashed')
ax.add_artist(arrow15)

# ==================== 推理阶段 ====================
inference_box = FancyBboxPatch((0.5, 3.5), 19, 0.05, 
                               boxstyle="round,pad=0", 
                               edgecolor='gray', facecolor='gray', linewidth=1)
ax.add_patch(inference_box)
ax.text(10, 3.8, 'Training Stage ↑        |        ↓ Inference Stage', 
        ha='center', va='center', fontsize=11, fontweight='bold', color='gray')

# 推理流程
# 图像库
gallery_box = FancyBboxPatch((1, 1.5), 2.5, 1.2, 
                             boxstyle="round,pad=0.1", 
                             edgecolor='blue', facecolor=color_input, linewidth=2)
ax.add_patch(gallery_box)
ax.text(2.25, 2.3, 'Gallery Images', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(2.25, 1.9, '(N images)', ha='center', va='center', fontsize=8)

# 查询文本
query_box = FancyBboxPatch((5, 1.5), 2.5, 1.2, 
                           boxstyle="round,pad=0.1", 
                           edgecolor='blue', facecolor=color_input, linewidth=2)
ax.add_patch(query_box)
ax.text(6.25, 2.3, 'Query Text', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(6.25, 1.9, '(Text description)', ha='center', va='center', fontsize=8)

# 特征提取
encode_box = FancyBboxPatch((9, 1.5), 2.5, 1.2, 
                            boxstyle="round,pad=0.1", 
                            edgecolor='orange', facecolor=color_feature, linewidth=2)
ax.add_patch(encode_box)
ax.text(10.25, 2.3, 'Feature Encoding', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(10.25, 1.9, '& Normalization', ha='center', va='center', fontsize=8)

# 相似度计算
similarity_box = FancyBboxPatch((13, 1.5), 2.5, 1.2, 
                                boxstyle="round,pad=0.1", 
                                edgecolor='purple', facecolor=color_module, linewidth=2)
ax.add_patch(similarity_box)
ax.text(14.25, 2.3, 'Similarity Matrix', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(14.25, 1.9, 'Cosine Similarity', ha='center', va='center', fontsize=8)

# 检索结果
result_box = FancyBboxPatch((17, 1.5), 2.5, 1.2, 
                            boxstyle="round,pad=0.1", 
                            edgecolor='green', facecolor=color_output, linewidth=2)
ax.add_patch(result_box)
ax.text(18.25, 2.3, 'Ranking Results', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(18.25, 1.9, 'Top-K Images', ha='center', va='center', fontsize=8)

# 推理箭头
arrow16 = FancyArrowPatch((3.5, 2.1), (5, 2.1), 
                         arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
ax.add_artist(arrow16)
arrow17 = FancyArrowPatch((7.5, 2.1), (9, 2.1), 
                         arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
ax.add_artist(arrow17)
arrow18 = FancyArrowPatch((11.5, 2.1), (13, 2.1), 
                         arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
ax.add_artist(arrow18)
arrow19 = FancyArrowPatch((15.5, 2.1), (17, 2.1), 
                         arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
ax.add_artist(arrow19)

# ==================== 评估指标 ====================
metrics_box = FancyBboxPatch((7, 0.2), 6, 0.7, 
                             boxstyle="round,pad=0.1", 
                             edgecolor='black', facecolor='#FFFACD', linewidth=2)
ax.add_patch(metrics_box)
ax.text(10, 0.7, 'Evaluation Metrics: Rank-1, Rank-5, Rank-10, mAP, mINP', 
        ha='center', va='center', fontsize=10, fontweight='bold')

# ==================== 图例 ====================
legend_elements = [
    mpatches.Patch(facecolor=color_input, edgecolor='black', label='Input Data'),
    mpatches.Patch(facecolor=color_encoder, edgecolor='black', label='CLIP Encoders'),
    mpatches.Patch(facecolor=color_feature, edgecolor='black', label='Features'),
    mpatches.Patch(facecolor=color_module, edgecolor='black', label='Task Modules'),
    mpatches.Patch(facecolor=color_loss, edgecolor='black', label='Loss Functions'),
    mpatches.Patch(facecolor=color_output, edgecolor='black', label='Output/Results'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9, ncol=2,
          bbox_to_anchor=(0.98, 0.98), framealpha=0.9)

# ==================== 注释说明 ====================
ax.text(10, 12.5, '① Training: Multi-task learning with SDM + MLM + ID losses', 
        ha='center', va='center', fontsize=9, style='italic', 
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

ax.text(10, 0.05, '② Inference: Extract features → Compute similarity → Rank and retrieve', 
        ha='center', va='center', fontsize=9, style='italic',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.tight_layout()
plt.savefig('d:/大三资料/AIGC科研/IRRA_NOISY_SOLVE/IRRA_Architecture_Diagram.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("架构图已保存到: IRRA_Architecture_Diagram.png")
plt.show()
