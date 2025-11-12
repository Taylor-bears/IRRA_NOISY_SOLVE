from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1)  # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [
        tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.0)
        for i, match_row in enumerate(matches)
    ]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices


class Evaluator:
    def __init__(self, img_loader, txt_loader, mask_noise=False):
        self.img_loader = img_loader  # gallery
        self.txt_loader = txt_loader  # query
        self.logger = logging.getLogger("IRRA.eval")
        self.mask_noise = mask_noise

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats = [], [], [], []

        # ===== 图像特征提取阶段 =====
        # 先处理图像 (收集全局与序列特征；序列特征用于构造伪图像上下文)
        image_seq_feats_all = []
        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image(
                    img
                )  # (B_img, D)，encode_image相比于base_model.encode_image是全局特征
                # 需要跨模态时调用 base_model.encode_image 获得序列特征
                if (
                    self.mask_noise
                ):  # 测试需要掩码噪声时，则需要全部图像序列特征，而不仅仅是全局特征
                    full_img_feat = model.base_model.encode_image(
                        img
                    )  # (B_img, L_img, D)
                    image_seq_feats_all.append(full_img_feat)
            gids.append(pid.view(-1))  # 展平
            gfeats.append(img_feat)
        gids = torch.cat(gids, 0)  # 按行拼接
        gfeats = torch.cat(gfeats, 0)

        # 归一化一份gallery特征，用于快速Top-K检索构造伪图像上下文
        gfeats_norm = F.normalize(gfeats, p=2, dim=1)
        image_seq_feats_cat = None
        if self.mask_noise and len(image_seq_feats_all) > 0:
            image_seq_feats_cat = torch.cat(
                image_seq_feats_all, 0
            )  # (N_gallery, L_img, D)

        # 打印一次测试时掩码配置，便于排查是否真的生效
        if self.mask_noise:
            args_obj = getattr(model, 'args', object())
            self.logger.info(f"[Eval] mask_noise_at_test=True strategy={getattr(args_obj,'mask_strategy','hard')} ctx={getattr(args_obj,'noise_ctx','topk_mean')} topk={getattr(args_obj,'mask_topk',5)}")
        else:
            self.logger.info("[Eval] mask_noise_at_test=False (no token masking)")

        # ===== 文本特征提取准备 =====
        # 文本特征（可选噪声掩码）
        tokenizer = None
        mask_token_id = None
        eot_token_id = None
        if self.mask_noise:
            from utils.simple_tokenizer import SimpleTokenizer

            tokenizer = SimpleTokenizer()
            mask_token_id = tokenizer.encoder.get("<|mask|>")
            eot_token_id = tokenizer.encoder.get("<|endoftext|>")
            # 属性词过滤：仅当开启时才生效
            args_obj = getattr(model, 'args', object())
            enable_attr_filter = getattr(args_obj, 'enable_attribute_filter', False)
            attr_set = set()
            if enable_attr_filter:
                vocab_path = getattr(args_obj, 'attribute_vocab_path', 'utils/attribute_vocab.txt')
                try:
                    with open(vocab_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            w = line.strip().lower()
                            if w:
                                attr_set.add(w)
                    self.logger.info(f"[Eval] Attribute filter enabled; loaded {len(attr_set)} words from {vocab_path}")
                except Exception as e:
                    self.logger.warning(f"[Eval] Attribute vocab not loaded from {vocab_path}: {e}; disabling filter")
                    enable_attr_filter = False

            def _allowed_token(token_id: int) -> bool:
                if not enable_attr_filter:
                    return True
                # 根据token_id从tokenizer的解码器中查找对应的文本内容
                tok = tokenizer.decoder.get(int(token_id), '')
                # 去除token中的词尾标记'</w>'
                base = tok.replace('</w>', '')
                # 遍历每个字符，只保留字母字符
                base = ''.join([c for c in base if c.isalpha()]).lower()
                # 仅允许长度>=3的字母词，且在属性词表
                return (len(base) >= 3) and (base in attr_set)

        # ===== 文本特征提取阶段 =====
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            if self.mask_noise and hasattr(model, 'noise_detection_head'):
                with torch.no_grad():
                    args_obj = getattr(model, 'args', object())
                    # 获取上下文模式配置，默认为'topk_mean'
                    ctx_mode = getattr(args_obj, 'noise_ctx', 'topk_mean')
                    seq_text_feats = model.base_model.encode_text(caption)  # (B, L_t, D)
                    if ctx_mode == 'none' or image_seq_feats_cat is None:
                        # 直接使用文本序列特征，不进行跨模态融合
                        fused = seq_text_feats
                        noise_logits = model.noise_detection_head(fused)
                        prob = torch.softmax(noise_logits, dim=-1)[..., 1]
                    else:
                        # 提取未掩码的文本全局特征
                        text_feat_nomask = model.encode_text(caption)  # (B, D)
                        text_feat_nomask_n = F.normalize(text_feat_nomask, p=2, dim=1)
                        # 计算文本特征与所有图库图像特征的相似度矩阵
                        sim = text_feat_nomask_n @ gfeats_norm.t()  # (B, N_gallery)
                        # 获取配置的Top-K值
                        k_cfg = getattr(args_obj, 'mask_topk', 5)
                        # 计算实际的K值，确保不超过图库大小
                        k = min(k_cfg if k_cfg > 0 else 5, gfeats_norm.size(0))
                        # 获取相似度最高的K个图像的索引
                        topk_idx = torch.topk(sim, k=k, dim=1, largest=True, sorted=True).indices  # (B, k)
                        Bq = caption.size(0)
                        L_img = image_seq_feats_cat.size(1)
                        D = image_seq_feats_cat.size(2)
                        if ctx_mode == 'topk_vote':
                            # K个图像 → K次融合 → K次预测 → 概率平均
                            # 针对每个Top-K图像，分别计算噪声概率，然后对概率做平均（或投票）
                            vote_probs = []
                            for kk in range(k):
                                # 批量取出每个样本对应的第kk个Top-K图像序列特征来和文本融合
                                sel_seq = image_seq_feats_cat[topk_idx[:, kk]]  # (B, L_img, D)
                                fused_k = model.cross_former(seq_text_feats, sel_seq, sel_seq)  # (B, L_t, D)
                                logits_k = model.noise_detection_head(fused_k)
                                prob_k = torch.softmax(logits_k, dim=-1)[..., 1]  # (B, L_t)
                                vote_probs.append(prob_k) # 收集每个Top-K图像对应的噪声概率
                            prob = torch.stack(vote_probs, dim=0).mean(0)  # (B, L_t)
                        else:
                            # 默认 topk_mean：先对Top-K图像序列取均值，再一次融合预测
                            # K个图像 → 平均 → 1次融合 → 1次预测
                            # 创建空的伪图像特征张量
                            pseudo_img = torch.empty((Bq, L_img, D), device=device, dtype=image_seq_feats_cat.dtype)
                            for b in range(Bq):
                                # 从图库中获取与当前文本最相似的K个图像的序列特征
                                seqs = image_seq_feats_cat[topk_idx[b]]  # (k, L_img, D)
                                # 对K个图像的序列特征求平均，得到伪图像上下文
                                pseudo_img[b] = seqs.mean(0)
                            fused = model.cross_former(seq_text_feats, pseudo_img, pseudo_img)  # (B, L_t, D)
                            noise_logits = model.noise_detection_head(fused)
                            prob = torch.softmax(noise_logits, dim=-1)[..., 1]  # (B, L_t)
                    strategy = getattr(args_obj, 'mask_strategy', 'hard')
                    prob_thresh = getattr(args_obj, 'mask_prob_thresh', 0.5)
                    max_ratio = getattr(args_obj, 'mask_max_ratio', 0.3)
                    max_tokens_cap = getattr(args_obj, 'mask_max_tokens', 0)
                    alpha_cap = getattr(args_obj, 'mask_soft_alpha_cap', 0.3)
                    B, L = caption.size()
                    # 根据不同的掩码策略进行处理
                    if strategy == 'none':
                        # 不进行任何掩码，直接使用原始文本提取特征
                        text_feat = model.encode_text(caption)
                    elif strategy == 'hard':
                        # 硬掩码策略：直接将噪声token替换为[MASK]
                        masked_caption = caption.clone()
                        for b in range(B):
                            end_pos = L - 1
                            # 找到结束标记位置，限定有效范围
                            if eot_token_id is not None:
                                eot_positions = (masked_caption[b] == eot_token_id).nonzero(as_tuple=False)
                                if eot_positions.numel() > 0:
                                    end_pos = int(eot_positions[0].item())
                            valid_range = (1, max(1, end_pos)) # [0]为开头，[1]为结束
                            if valid_range[1] <= valid_range[0]:
                                continue
                            cand_idx = torch.arange(valid_range[0], valid_range[1], device=device)
                            # 获取有效位置的噪声概率
                            cand_prob = prob[b, cand_idx]
                            # 属性词过滤
                            if enable_attr_filter:
                                allow_mask = []
                                for t_id in caption[b, cand_idx].tolist():
                                    allow_mask.append(_allowed_token(int(t_id)))
                                allow_mask = torch.tensor(allow_mask, device=device, dtype=torch.bool)
                                cand_idx = cand_idx[allow_mask]
                                cand_prob = cand_prob[allow_mask]
                            # 筛选出超过噪声阈值的token
                            keep = cand_prob > prob_thresh
                            if keep.any():
                                cand_idx = cand_idx[keep]
                                cand_prob = cand_prob[keep]
                                # 按概率降序排序
                                sorted_prob, order = torch.sort(cand_prob, descending=True)
                                # 重新排列索引
                                cand_idx = cand_idx[order]
                                # 计算最大掩码数量（基于比例）
                                max_mask_ratio = max(1, int((valid_range[1]-valid_range[0]) * max_ratio))
                                # 计算实际的最大掩码数量，受比例和绝对数量的限制
                                max_mask = min(max_mask_ratio, max_tokens_cap) if max_tokens_cap > 0 else max_mask_ratio
                                cand_idx = cand_idx[:max_mask]
                                for t in cand_idx.tolist():
                                    if masked_caption[b, t] != 0 and mask_token_id is not None:
                                        # 将token替换为掩码token
                                        masked_caption[b, t] = mask_token_id
                        text_feat = model.encode_text(masked_caption)
                    else: # soft
                        # 软掩码策略：对噪声token进行加权处理
                        seq_text_feats_full = model.base_model.encode_text(caption)  # (B, L, D)
                        # 初始化权重张量，全为1
                        weights = torch.ones_like(prob)
                        for b in range(B):
                            end_pos = L - 1
                            if eot_token_id is not None:
                                eot_positions = (caption[b] == eot_token_id).nonzero(as_tuple=False)
                                if eot_positions.numel() > 0:
                                    end_pos = int(eot_positions[0].item())
                            valid_range = (1, max(1, end_pos))
                            if valid_range[1] <= valid_range[0]:
                                continue
                            cand_idx = torch.arange(valid_range[0], valid_range[1], device=device)
                            cand_prob = prob[b, cand_idx]
                            # 属性词过滤
                            if enable_attr_filter:
                                allow_mask = []
                                for t_id in caption[b, cand_idx].tolist():
                                    allow_mask.append(_allowed_token(int(t_id)))
                                allow_mask = torch.tensor(allow_mask, device=device, dtype=torch.bool)
                                cand_idx = cand_idx[allow_mask]
                                cand_prob = cand_prob[allow_mask]
                            mask_sel = cand_prob > prob_thresh
                            if mask_sel.any():
                                sel_idx = cand_idx[mask_sel]
                                sel_prob = cand_prob[mask_sel]
                                max_soft_ratio = max(1, int((valid_range[1]-valid_range[0]) * max_ratio)) 
                                max_soft = min(max_soft_ratio, max_tokens_cap) if max_tokens_cap > 0 else max_soft_ratio
                                sorted_p, order = torch.sort(sel_prob, descending=True)
                                sel_idx = sel_idx[order][:max_soft]
                                sel_prob = sorted_p[:max_soft]
                                # 计算软掩码权重（概率越高，权重越低）
                                alpha = torch.clamp(sel_prob, max=alpha_cap)
                                # 更新权重张量
                                weights[b, sel_idx] = 1.0 - alpha
                        eps = 1e-6
                        # 在最后一个维度上增加一维，用于广播
                        weights = weights.unsqueeze(-1)
                        # 计算加权平均特征
                        # 使用加权平均融合文本局部特征来获得新的全局文本特征
                        weighted_feats = (seq_text_feats_full * weights).sum(1) / (weights.sum(1) + eps)
                        text_feat = weighted_feats.float()
            else:
                with torch.no_grad():
                    text_feat = model.encode_text(caption)
            qids.append(pid.view(-1))  # 展平
            qfeats.append(text_feat)
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)

        return qfeats, gfeats, qids, gids

    def eval(self, model, i2t_metric=False):
        # 步骤1：计算所有特征嵌入
        qfeats, gfeats, qids, gids = self._compute_embedding(model)
        # 步骤2：特征归一化
        qfeats = F.normalize(qfeats, p=2, dim=1)  # text features
        gfeats = F.normalize(gfeats, p=2, dim=1)  # image features
        # 步骤3：计算相似度矩阵
        similarity = qfeats @ gfeats.t()
        # 步骤4：计算文本到图像检索指标
        t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(
            similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True
        )
        t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table.add_row(["t2i", t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])
        # 步骤6：可选计算图像到文本检索指标
        if i2t_metric:
            # 转置相似度矩阵：图像作为查询，文本作为候选
            i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(
                similarity=similarity.t(),
                q_pids=gids,
                g_pids=qids,
                max_rank=10,
                get_mAP=True,
            )
            i2t_cmc, i2t_mAP, i2t_mINP = (
                i2t_cmc.numpy(),
                i2t_mAP.numpy(),
                i2t_mINP.numpy(),
            )
            table.add_row(
                ["i2t", i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP]
            )
        # table.float_format = '.4'
        table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        self.logger.info("\n" + str(table))

        return t2i_cmc[0]
