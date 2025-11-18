from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging
from utils.simple_tokenizer import SimpleTokenizer


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
        # 支持单路或多路文本loader（如 noisy/clean/[mask]）
        self.txt_loaders = txt_loader if isinstance(txt_loader, dict) else { 'noisy': txt_loader }
        self.logger = logging.getLogger("IRRA.eval")
        self.mask_noise = mask_noise

    # 目前，我们使用_compute_embedding主要用于提取一次 gallery 全局特征与 gids，里面的TopK以及hard掩码逻辑已经转移到了eval()，因为我们直接用这些来算最终的相似矩阵
    def _compute_embedding(self, model, txt_loader, mask_override: bool | None = None):
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
        mask_active = self.mask_noise if mask_override is None else bool(mask_override)
        if mask_active:
            tokenizer = SimpleTokenizer()
            mask_token_id = tokenizer.encoder.get("<|mask|>")
            eot_token_id = tokenizer.encoder.get("<|endoftext|>")
            # 属性词过滤：仅当开启时才生效
            args_obj = getattr(model, 'args', object())
            enable_attr_filter = getattr(args_obj, 'enable_attribute_filter', False)
            # 若开启属性过滤，使用 attribute_extractor 的 build_attribute_mask 生成 token 级允许掩码
            if enable_attr_filter:
                from utils.attribute_extractor import build_attribute_mask

                # 输入文本的token序列，输出属性token的位置
                def _compute_allowed_mask_for_row(row_tokens: torch.Tensor) -> torch.Tensor:
                    """基于 BPE token 重建词与span，调用 attribute_extractor.build_attribute_mask 获取允许掩码。
                    返回长度为L的bool张量，True表示该token位置可作为属性词进行掩码候选。
                    """
                    # 确定有效范围（去掉起始特殊符号，截止到第一个eot之前）
                    L = int(row_tokens.size(0))
                    end_pos = L - 1
                    if eot_token_id is not None:
                        eot_positions = (row_tokens == eot_token_id).nonzero(as_tuple=False)
                        if eot_positions.numel() > 0:
                            end_pos = int(eot_positions[0].item())
                    start = 1
                    if end_pos <= start:
                        return torch.zeros(L, dtype=torch.bool, device=row_tokens.device)
                    # 重建词序列与对应token span（相对去掉起始特殊符号后的索引）
                    words: list[str] = [] # 存储重建的完整单词
                    spans: list[tuple[int,int]] = [] # 存储每个单词对应的token位置范围
                    cur_chars: list[str] = [] # 当前正在构建的单词字符列表
                    cur_start: int | None = None # 当前单词的起始token索引
                    for rel_idx, t in enumerate(row_tokens[start:end_pos].tolist()):
                        piece = tokenizer.decoder.get(int(t), '')
                        # 如果是新单词的开始，记录起始位置
                        if cur_start is None:
                            cur_start = rel_idx
                        # 去除BPE结尾标记
                        seg = piece.replace('</w>', '')
                        cur_chars.append(seg)
                        # 通过BPE词尾标记判断是否是单词结尾，因为一个单词可能被拆成多个BPE片段
                        end_of_word = piece.endswith('</w>')
                        if end_of_word: # 遇到结尾
                            # 将字符列表拼接成完整单词
                            w = ''.join(cur_chars)
                            words.append(w)
                            spans.append((cur_start, rel_idx + 1))  # 记录该单词对应的token范围
                             # 重置当前单词状态，准备构建下一个单词
                            cur_chars = []
                            cur_start = None
                    # 收尾：若最后一个词无</w>也收集
                    if cur_chars and cur_start is not None:
                        w = ''.join(cur_chars)
                        words.append(w)
                        spans.append((cur_start, (row_tokens[start:end_pos].size(0))))
                    # 调用属性提取器生成token级掩码，offset=1 用于对齐起始特殊符号的偏移
                    mask_list = build_attribute_mask(words, spans, text_length=L, offset=1)
                    return torch.tensor(mask_list, dtype=torch.bool, device=row_tokens.device)

        # ===== 文本特征提取阶段 =====
        for pid, caption in txt_loader:
            caption = caption.to(device)
            if mask_active and hasattr(model, 'noise_detection_head'):
                with torch.no_grad():
                    args_obj = getattr(model, 'args', object())
                    # 获取上下文模式配置，默认为'topk_mean'
                    ctx_mode = getattr(args_obj, 'noise_ctx', 'topk_vote')
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
                            # 利用 gather 直接一次性取出所有候选序列再求平均，减少 Python 循环
                            # 索引操作：对每个索引值，从image_seq_feats_cat中提取对应的图像序列，topk_idx[0] = [23, 45, 67, 89, 12]，那么gathered[0] 包含image_seq_feats_cat[23]······[12]
                            gathered = image_seq_feats_cat[topk_idx]  # (B, k, L_img, D)
                            # 在第k维度上求均值，得到每一个caption的伪图像上下文
                            pseudo_img = gathered.mean(1)  # (B, L_img, D)
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
                            # 预先计算整句允许掩码（属性词/短语）
                            allowed_mask_full = None
                            if enable_attr_filter:
                                allowed_mask_full = _compute_allowed_mask_for_row(masked_caption[b])
                            cand_idx = torch.arange(valid_range[0], valid_range[1], device=device)
                            # 获取有效位置的噪声概率
                            cand_prob = prob[b, cand_idx]
                            # 属性词过滤
                            if enable_attr_filter and allowed_mask_full is not None:
                                allow_mask = allowed_mask_full[cand_idx]
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
                            # 预先计算整句允许掩码（属性词/短语）
                            allowed_mask_full = None
                            if enable_attr_filter:
                                allowed_mask_full = _compute_allowed_mask_for_row(caption[b])
                            cand_idx = torch.arange(valid_range[0], valid_range[1], device=device)
                            cand_prob = prob[b, cand_idx]
                            # 属性词过滤
                            if enable_attr_filter and allowed_mask_full is not None:
                                allow_mask = allowed_mask_full[cand_idx]
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
        results = []
        # 针对可用的文本路分别评估：noisy(不掩码)、clean(不掩码)、mask(对noisy掩码)
        routes = []
        if 'noisy' in self.txt_loaders:
            routes.append(('t2i-noisy', self.txt_loaders['noisy'], False))
        if 'clean' in self.txt_loaders:
            routes.append(('t2i-clean', self.txt_loaders['clean'], False))
        # 掩码行：使用 noisy 文本并强制启用掩码（仅当外部允许mask时）
        if self.mask_noise and ('noisy' in self.txt_loaders):
            routes.append(('t2i-mask', self.txt_loaders['noisy'], True))

        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        first_gfeats = None
        first_gids = None

        for label, txt_loader, force_mask in routes:
            if label == 't2i-mask':
                if not force_mask:
                    table.add_row([label, "--", "--", "--", "--", "--"])
                    continue
                # 新的掩码评估路径：S_final = (S0 + Σ S_k) / (K+1)
                device = next(model.parameters()).device
                model = model.eval()
                # 步骤一：准备 gallery 全局特征与归一化（优先复用首轮结果）
                # 计算相似度用的是图像的全局特征
                if first_gfeats is None:
                    # 通过无掩码路径先取到 gfeats/gids
                    qfeats_tmp, gfeats_tmp, qids_tmp, gids_tmp = self._compute_embedding(model, txt_loader, mask_override=False)
                    first_gfeats, first_gids = gfeats_tmp, gids_tmp
                gfeats = first_gfeats
                gids = first_gids
                gfeats_norm = F.normalize(gfeats, p=2, dim=1) # 后续所有得到的文本特征tfeats_norm都需要和它计算相似度，而它是不变的

                # 步骤二：若需要跨模态融合进行噪声预测，准备图像序列特征（仅一次）
                # cross_former用的是图像的序列特征
                image_seq_feats_all = []
                with torch.no_grad():
                    for _, img in self.img_loader:
                        img = img.to(device)
                        full_img_feat = model.base_model.encode_image(img)  # (B_img, L_img, D)
                        image_seq_feats_all.append(full_img_feat)
                image_seq_feats_cat = torch.cat(image_seq_feats_all, 0) if len(image_seq_feats_all) > 0 else None

                # 3) 步骤三：构造属性掩码相关配置
                args_obj = getattr(model, 'args', object())
                k_cfg = getattr(args_obj, 'mask_topk', 5)
                prob_thresh = getattr(args_obj, 'mask_prob_thresh', 0.5)
                max_ratio = getattr(args_obj, 'mask_max_ratio', 0.3)
                max_tokens_cap = getattr(args_obj, 'mask_max_tokens', 0)
                enable_attr_filter = getattr(args_obj, 'enable_attribute_filter', False)

                tokenizer = SimpleTokenizer()
                mask_token_id = tokenizer.encoder.get("<|mask|>")
                eot_token_id = tokenizer.encoder.get("<|endoftext|>")

                if enable_attr_filter:
                    from utils.attribute_extractor import build_attribute_mask

                    # 与embedding阶段里一致，True表示该token位置可作为属性词进行掩码候选
                    def _compute_allowed_mask_for_row(row_tokens: torch.Tensor) -> torch.Tensor:
                        L = int(row_tokens.size(0))
                        end_pos = L - 1
                        if eot_token_id is not None:
                            eot_positions = (row_tokens == eot_token_id).nonzero(as_tuple=False)
                            if eot_positions.numel() > 0:
                                end_pos = int(eot_positions[0].item())
                        start = 1
                        if end_pos <= start:
                            return torch.zeros(L, dtype=torch.bool, device=row_tokens.device)
                        words = []
                        spans = []
                        cur_chars = []
                        cur_start = None
                        for rel_idx, t in enumerate(row_tokens[start:end_pos].tolist()):
                            piece = tokenizer.decoder.get(int(t), '')
                            if cur_start is None:
                                cur_start = rel_idx
                            seg = piece.replace('</w>', '')
                            cur_chars.append(seg)
                            if piece.endswith('</w>'):
                                w = ''.join(cur_chars)
                                words.append(w)
                                spans.append((cur_start, rel_idx + 1))
                                cur_chars = []
                                cur_start = None
                        if cur_chars and cur_start is not None:
                            w = ''.join(cur_chars)
                            words.append(w)
                            spans.append((cur_start, (row_tokens[start:end_pos].size(0))))
                        mask_list = build_attribute_mask(words, spans, text_length=L, offset=1)
                        return torch.tensor(mask_list, dtype=torch.bool, device=row_tokens.device)

                # 若存在clean文本通道，基于原始字符串构造difflib的token级GT（与训练一致）
                has_clean = ('clean' in self.txt_loaders)
                clean_dataset = self.txt_loaders['clean'].dataset if has_clean else None
                noisy_dataset = txt_loader.dataset

                # 步骤四：计算最终相似矩阵 S_final，同时统计每个K路径的噪声预测精度（可选）
                all_rows = []  # 收集每个batch的 S_final_batch
                all_qids = []
                # 统计每个K路径的噪声预测准确率：累计正确数与总数
                noise_correct_per_k = None
                noise_total_per_k = None
                global_idx = 0 # 表示当前处理到第几个文本样本了，主要用于噪声预测准确率记录处理文本的个数
                with torch.no_grad():
                    # txt_loader来自于Dataloder，逐batch处理，所以这里的pid, caption是一个batch内的数据
                    for pid, caption in txt_loader:
                        caption = caption.to(device)
                        B, L = caption.size()
                        # A) 提取噪声文本全局特征计算原始相似矩阵 S0_batch
                        text_feat_nomask = model.encode_text(caption)  # (B,D)
                        text_feat_nomask_n = F.normalize(text_feat_nomask, p=2, dim=1)
                        S0_batch = text_feat_nomask_n @ gfeats_norm.t()  # (B, N_gallery)

                        # B) 选取相似图像中的 Top-K 个
                        sim_for_topk = S0_batch  # 已经是归一化后的余弦相似
                        k = min(k_cfg if k_cfg > 0 else 5, gfeats_norm.size(0))
                        topk_idx = torch.topk(sim_for_topk, k=k, dim=1, largest=True, sorted=True).indices  # (B,k)

                        # 提取噪声文本序列特征用于后续跨模态融合，预测噪声损失
                        seq_text_feats = model.base_model.encode_text(caption)  # (B, L_t, D)

                        # 使用difflib在词级对齐并映射到token级GT；可选属性过滤
                        gt_noise_batch = None  # (B,L) long(0/1)
                        eval_mask_batch = None  # (B,L) bool
                        if has_clean:
                            noisy_caps_strs = noisy_dataset.captions[global_idx:global_idx+B]
                            clean_caps_strs = clean_dataset.captions[global_idx:global_idx+B]
                            gt_list = [] # 相当于noisy_label
                            msk_list = [] # 相当于noisy文本的attribute_mask
                            import regex as re
                            import difflib
                            for b in range(B):
                                noisy_cap = noisy_caps_strs[b]
                                clean_cap = clean_caps_strs[b]
                                word_pattern = r"[A-Za-z0-9]+|[^\sA-Za-z0-9]+"
                                clean_words = re.findall(word_pattern, clean_cap)
                                noisy_words = re.findall(word_pattern, noisy_cap)
                                matcher = difflib.SequenceMatcher(a=clean_words, b=noisy_words)
                                noise_word_flags = [0] * len(noisy_words)
                                for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                                    if tag == 'replace' or tag == 'insert':
                                        for j in range(j1, j2):
                                            if 0 <= j < len(noisy_words):
                                                noise_word_flags[j] = 1
                                # 将 noisy_words 映射到 token span
                                noisy_token_ids = []
                                word_to_token_span = []
                                cur = 0
                                for w in noisy_words:
                                    t_ids = tokenizer.encode(w)
                                    noisy_token_ids.extend(t_ids)
                                    word_to_token_span.append((cur, cur + len(t_ids)))
                                    cur += len(t_ids)
                                # 有效范围mask：从1到第一个eot-1
                                eval_mask = torch.zeros(L, dtype=torch.bool, device=device)
                                end_pos = L - 1
                                if eot_token_id is not None:
                                    eot_positions = (caption[b] == eot_token_id).nonzero(as_tuple=False)
                                    if eot_positions.numel() > 0:
                                        end_pos = int(eot_positions[0].item())
                                start = 1
                                if end_pos > start:
                                    eval_mask[start:end_pos] = True
                                # token级GT
                                gt_noise = torch.zeros(L, dtype=torch.long, device=device)
                                for wi, flag in enumerate(noise_word_flags):
                                    if flag == 1 and wi < len(word_to_token_span):
                                        t_start = 1 + word_to_token_span[wi][0]
                                        t_end = 1 + word_to_token_span[wi][1]
                                        t_start = min(t_start, L)
                                        t_end = min(t_end, L)
                                        if t_start < t_end:
                                            gt_noise[t_start:t_end] = 1
                                # 属性过滤：仅在属性token上统计
                                if enable_attr_filter:
                                    from utils.attribute_extractor import build_attribute_mask
                                    attr_mask_list = build_attribute_mask(noisy_words, word_to_token_span, text_length=L, offset=1)
                                    attr_mask = torch.tensor(attr_mask_list, dtype=torch.bool, device=device)
                                    eval_mask = eval_mask & attr_mask
                                gt_list.append(gt_noise)
                                msk_list.append(eval_mask)
                            gt_noise_batch = torch.stack(gt_list, dim=0) if gt_list else None
                            eval_mask_batch = torch.stack(msk_list, dim=0) if msk_list else None

                        # C) 针对每个路径 i 产生 masked caption，并计算对应相似矩阵 S_i
                        S_acc_batch = S0_batch.clone()
                        for kk in range(k):
                            # 选出第kk个Top-K图像序列特征
                            sel_seq = image_seq_feats_cat[topk_idx[:, kk]]  # (B, L_img, D)
                            fused_k = model.cross_former(seq_text_feats, sel_seq, sel_seq)  # (B, L_t, D)
                            logits_k = model.noise_detection_head(fused_k)
                            prob_k = torch.softmax(logits_k, dim=-1)[..., 1]  # (B, L_t)

                            # 若有GT，统计该路径噪声预测精度
                            if (gt_noise_batch is not None) and (eval_mask_batch is not None):
                                if noise_correct_per_k is None:
                                    noise_correct_per_k = [0 for _ in range(k)] # [kk]记录第kk个路径上所有batch样本的正确预测数量总和
                                    noise_total_per_k = [0 for _ in range(k)] # [kk]记录第kk个路径上所有batch样本的评估token总数
                                pred_k = (prob_k > prob_thresh) # (B,L) bool
                                correct = ((pred_k == gt_noise_batch.bool()) & eval_mask_batch).sum().item() # item将结果转为标量，所以说每个样本的指标都是累加到一起的，一整个batch的
                                total = eval_mask_batch.sum().item()
                                noise_correct_per_k[kk] += correct
                                noise_total_per_k[kk] += total

                            # 生成该路径的 masked captions
                            masked_caption = caption.clone()
                            for b in range(B): # 针对batch内每个样本，相当于每个样本的txt与其第kk相似的图像融合掩码得到batch个mask_caption
                                end_pos = L - 1
                                if eot_token_id is not None:
                                    eot_positions = (masked_caption[b] == eot_token_id).nonzero(as_tuple=False)
                                    if eot_positions.numel() > 0: 
                                        end_pos = int(eot_positions[0].item())
                                valid_range = (1, max(1, end_pos))
                                if valid_range[1] <= valid_range[0]:
                                    continue
                                cand_idx = torch.arange(valid_range[0], valid_range[1], device=device)
                                cand_prob = prob_k[b, cand_idx]
                                if enable_attr_filter:
                                    allow_mask_full = _compute_allowed_mask_for_row(masked_caption[b])
                                    allow_mask = allow_mask_full[cand_idx]
                                    cand_idx = cand_idx[allow_mask]
                                    cand_prob = cand_prob[allow_mask]
                                keep = cand_prob > prob_thresh
                                if keep.any():
                                    cand_idx = cand_idx[keep]
                                    cand_prob = cand_prob[keep]
                                    _, order = torch.sort(cand_prob, descending=True)
                                    cand_idx = cand_idx[order]
                                    max_mask_ratio = max(1, int((valid_range[1]-valid_range[0]) * max_ratio))
                                    max_mask = min(max_mask_ratio, max_tokens_cap) if max_tokens_cap > 0 else max_mask_ratio
                                    cand_idx = cand_idx[:max_mask]
                                    for t in cand_idx.tolist():
                                        if masked_caption[b, t] != 0 and mask_token_id is not None:
                                            masked_caption[b, t] = mask_token_id

                            # Topk每个mask_caption都要计算自己路径的相似矩阵 S_k_batch
                            text_feat_k = model.encode_text(masked_caption)
                            text_feat_k_n = F.normalize(text_feat_k, p=2, dim=1)
                            S_k_batch = text_feat_k_n @ gfeats_norm.t()
                            S_acc_batch = S_acc_batch + S_k_batch # S_acc_batch一开始是S0_batch，后续累加每个S_k_batch

                        # D) 求平均，汇入全局
                        S_final_batch = S_acc_batch / float(k + 1)
                        all_rows.append(S_final_batch) # 收集每个batch的最终相似矩阵
                        all_qids.append(pid.view(-1)) # 收集query ids
                        global_idx += B

                similarity = torch.cat(all_rows, dim=0)
                qids = torch.cat(all_qids, dim=0)
                gfeats = first_gfeats
                gids = first_gids
                # 用最终相似矩阵计算指标
                t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
                t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
                table.add_row([label, t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])
                # 打印每个K路径的token级噪声预测精度列表
                if noise_correct_per_k is not None and noise_total_per_k is not None and sum(noise_total_per_k) > 0:
                    acc_list = []
                    for kk in range(len(noise_correct_per_k)):
                        if noise_total_per_k[kk] > 0:
                            acc_k = noise_correct_per_k[kk] / max(1, noise_total_per_k[kk])
                        else:
                            acc_k = 0.0
                        acc_list.append(acc_k)
                    acc_mean = float(np.mean(acc_list)) if len(acc_list) > 0 else 0.0 # 平均值
                    acc_std = float(np.std(acc_list)) if len(acc_list) > 0 else 0.0 # 标准差
                    self.logger.info(f"[NoiseEval] token-acc per-K: {[round(a,4) for a in acc_list]} (mean={acc_mean:.4f}, std={acc_std:.4f}, K={len(acc_list)})")
            else:
                # 保持原有单路（noisy/clean）逻辑
                # 步骤1：计算所有特征嵌入（图像库仅在首轮提取一次以节省时间）
                if first_gfeats is None:
                    qfeats, gfeats, qids, gids = self._compute_embedding(model, txt_loader, mask_override=False)
                    first_gfeats, first_gids = gfeats, gids
                else:
                    qfeats, gfeats, qids, gids = self._compute_embedding(model, txt_loader, mask_override=False)
                    gfeats, gids = first_gfeats, first_gids
                # 步骤2：特征归一化并评估
                qfeats = F.normalize(qfeats, p=2, dim=1)
                gfeats = F.normalize(gfeats, p=2, dim=1)
                similarity = qfeats @ gfeats.t()
                t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
                t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
                table.add_row([label, t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])

        # 若未启用掩码而又存在noisy文本，则补一行占位符“--”以满足展示需求
        if ('noisy' in self.txt_loaders) and (not self.mask_noise):
            table.add_row(["t2i-mask", "--", "--", "--", "--", "--"])
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
        def _fmt_float_or_str(val):
            try:
                return f"{float(val):.3f}"
            except (TypeError, ValueError):
                return str(val)

        table.custom_format["R1"] = lambda f, v: _fmt_float_or_str(v)
        table.custom_format["R5"] = lambda f, v: _fmt_float_or_str(v)
        table.custom_format["R10"] = lambda f, v: _fmt_float_or_str(v)
        table.custom_format["mAP"] = lambda f, v: _fmt_float_or_str(v)
        table.custom_format["mINP"] = lambda f, v: _fmt_float_or_str(v)
        self.logger.info("\n" + str(table))
        # 返回第一行(noisy)的R1作为主监控指标
        try:
            return float(table._rows[0][1])
        except Exception:
            return 0.0