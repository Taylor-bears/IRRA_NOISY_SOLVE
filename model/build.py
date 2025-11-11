from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict


class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        # num是模型需要区分的决策选项数量​，这是不同于特征维度的
        self.num_classes = num_classes
        self._set_task()

        # 加载clip预训练（model是模型主体，cfg是模型的配置字典）
        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim'] # 特征维度
        # 温度参数
        self.logit_scale = torch.ones([]) * (1 / args.temperature) 

        # 分类头
        #  ID分类任务，行人重识别
        if 'id' in args.loss_names:
            # 线性分类器：将CLIP特征映射到行人ID类别
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            # 初始化分类器权重
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)
        # 跨模态模块：供 MLM 或 噪声检测 复用
        if ('mlm' in args.loss_names) or getattr(args, 'noise_detection', False):
            # 跨模态注意力层（文本→图像）
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            # 跨模态Transformer编码器
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5
            
            # 定义层归一化模块
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            # 初始化Transformer权重
            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # 初始化跨模态注意力权重
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            # 当使用MLM任务时，构建MLM头
            if 'mlm' in args.loss_names:
                # MLM预测头：4层结构（线性→GELU→归一化→线性）
                self.mlm_head = nn.Sequential(
                    OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                                ('gelu', QuickGELU()),
                                ('ln', LayerNorm(self.embed_dim)),
                                ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
                # 初始化MLM头权重
                nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
                nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

            # 当启用噪声检测时，构建二分类头
            if getattr(args, 'noise_detection', False):
                self.noise_detection_head = nn.Sequential(
                    OrderedDict([
                        ('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                        ('gelu', QuickGELU()),
                        ('ln', LayerNorm(self.embed_dim)),
                        ('fc', nn.Linear(self.embed_dim, 2))
                    ])
                )
                # 初始化噪声检测头权重
                nn.init.normal_(self.noise_detection_head.dense.weight, std=fc_std)
                nn.init.normal_(self.noise_detection_head.fc.weight, std=proj_std)

    # 设置当前任务
    def _set_task(self):
        # 解析损失函数列表（如"id+mlm+itc"）
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    # 跨模态特征交互方法
    def cross_former(self, q, k, v):
        # 输入：查询（文本）、键（图像）、值（图像）
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        # Transformer编码
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
         # 输出归一化
        x = self.ln_post(x)
        return x

    # 图像编码（取CLS令牌）
    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        return x[:, 0, :].float()
        # return x.float() # for CLIP ResNet visual model

    # 文本编码（取EOS令牌）
    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    # 前向传播
    def forward(self, batch):
        # 存储所有返回值的字典
        ret = dict()
        # 获取输入数据
        images = batch['images']
        caption_ids = batch['caption_ids']
        # 提取图像和文本特征（通过CLIP基模型）
        image_feats, text_feats = self.base_model(images, caption_ids)
        # 提取有效特征
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
        # 温度参数
        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})
        # ---- 多任务损失计算 ----
        if 'itc' in self.current_task:
            ret.update({'itc_loss':objectives.compute_itc(i_feats, t_feats, logit_scale)})
        
        if 'sdm' in self.current_task:
            ret.update({'sdm_loss':objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})

        if 'cmpm' in self.current_task:
            ret.update({'cmpm_loss':objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})
        
        if 'id' in self.current_task:
            # 图像身份分类：使用分类器预测行人ID
            image_logits = self.classifier(i_feats.half()).float()
            # 文本身份分类
            text_logits = self.classifier(t_feats.half()).float()
            # # 计算身份识别损失，损失是用来学习的
            ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'])*self.args.id_loss_weight})
            # 计算预测结果（取概率最大的类别）
            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)
            # 计算准确率指标，准确率是用来评估的
            # 计算图像分类准确率
            image_precision = (image_pred == batch['pids']).float().mean()
            # 计算文本分类准确率
            text_precision = (text_pred == batch['pids']).float().mean()
            # 记录准确率指标
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})
        
        if 'mlm' in self.current_task:
            mlm_ids = batch['mlm_ids']

            mlm_feats = self.base_model.encode_text(mlm_ids)
            # 跨模态注意力：使用图像特征来帮助恢复被掩码的文本
            x = self.cross_former(mlm_feats, image_feats, image_feats)
            # MLM头部：将特征映射到词汇表空间
            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]
            # 重塑为二维张量以便计算损失
            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})
            # 计算MLM准确率指标
            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})

        # 噪声检测分支（token-level 二分类）：输入为噪声文本与对应图像
        if getattr(self.args, 'noise_detection', False) and ('noise_labels' in batch): # new
            # 使用原始caption_ids（噪声文本）对应的文本序列特征作为query
            # text_feats 来自 self.base_model(images, caption_ids) 已计算，为整序列 (B, L, D)
            # 跨模态融合：文本特征作为query，图像特征作为key和value
            fused = self.cross_former(text_feats, image_feats, image_feats)  # (B, L, D)
            # 噪声检测头部：每个token位置进行二分类（噪声/干净）
            noise_logits = self.noise_detection_head(fused)  # (B, L, 2)
            noise_labels = batch['noise_labels']  # (B, L)
            attribute_mask = batch.get('attribute_mask', torch.ones_like(noise_labels, dtype=torch.float))
            noise_loss = objectives.compute_noise_detection(noise_logits.float(), noise_labels, attribute_mask) * getattr(self.args, 'noise_loss_weight', 1.0) # 这里的权重有待调整
            ret.update({'noise_loss': noise_loss})

            # 仅在mask位置统计准确率
            with torch.no_grad():
                pred = noise_logits.argmax(dim=-1)
                mask = attribute_mask > 0
                if mask.sum() > 0:
                    acc = (pred[mask] == noise_labels[mask]).float().mean()
                else:
                    acc = torch.tensor(0.0, device=pred.device)
            ret.update({'noise_acc': acc})

        return ret # 返回所有损失和指标


def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model