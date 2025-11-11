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

    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices


class Evaluator():
    def __init__(self, img_loader, txt_loader, mask_noise=False):
        self.img_loader = img_loader # gallery
        self.txt_loader = txt_loader # query
        self.logger = logging.getLogger("IRRA.eval")
        self.mask_noise = mask_noise

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats = [], [], [], []

        # 先处理图像 (如果需要平均图像特征用于噪声预测)
        image_feats_all = []
        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img)  # (B_img, D)，encode_image相比于base_model.encode_image是全局特征
                # 需要跨模态时调用 base_model.encode_image 获得序列特征
                if self.mask_noise: # 测试需要掩码噪声时，则需要全部图像序列特征，而不仅仅是全局特征
                    full_img_feat = model.base_model.encode_image(img)  # (B_img, L_img, D)
                    image_feats_all.append(full_img_feat)
            gids.append(pid.view(-1)) # 展平
            gfeats.append(img_feat)
        gids = torch.cat(gids, 0) # 按行拼接
        gfeats = torch.cat(gfeats, 0)

        avg_img_seq = None
        if self.mask_noise and len(image_feats_all) > 0:
            # 这里计算所有图像序列特征的均值（平均图像特征），用于后续文本噪声掩码预测
            # 对于某个文本，我们并不知道判断它是否有噪声，具体是用哪张图像来辅助，此时使用所有图像的均值作为辅助
            # ---------------------------思考！！！！！我认为使用平均图像不如使用pid相同的图像，这样才有意义，描述同一个人然后取均值来判断---------------------------
            avg_img_seq = torch.cat(image_feats_all, 0).mean(0, keepdim=True)  # (1, L_img, D)

        # 文本特征（可选噪声掩码）
        tokenizer = None
        mask_token_id = None
        eot_token_id = None
        if self.mask_noise:
            from utils.simple_tokenizer import SimpleTokenizer
            tokenizer = SimpleTokenizer()
            mask_token_id = tokenizer.encoder.get('<|mask|>')
            eot_token_id = tokenizer.encoder.get('<|endoftext|>')

        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            if self.mask_noise and avg_img_seq is not None and hasattr(model, 'noise_detection_head'):
                # 获得文本序列特征
                with torch.no_grad():
                    # 步骤1: 提取完整文本序列特征（用于token级噪声检测）
                    seq_text_feats = model.base_model.encode_text(caption)  # (B, L_t, D)
                    # 步骤2: 构造伪图像序列（平均图像序列复制到batch维）
                    pseudo_img = avg_img_seq.repeat(seq_text_feats.size(0), 1, 1)
                    # 步骤3: 跨模态融合
                    fused = model.cross_former(seq_text_feats, pseudo_img, pseudo_img)  # (B, L_t, D)
                    # 步骤4: 噪声分类
                    noise_logits = model.noise_detection_head(fused)  # (B, L_t, 2)
                    pred_noise = noise_logits.argmax(-1)  # (B, L_t)
                    # 步骤5: 应用掩码（仅在存在mask_token时启用）
                    if mask_token_id is not None:
                        # 步骤6: 基于 <|endoftext|> 定位有效词区间，避免覆盖到 padding(0)
                        masked_caption = caption.clone()
                        B, L = masked_caption.size()
                        for b in range(B):
                            # 默认有效结束位置为 L-1（末位通常是eot或padding），实际以首次出现的eot为准
                            end_pos = L - 1
                            if eot_token_id is not None:
                                eot_positions = (masked_caption[b] == eot_token_id).nonzero(as_tuple=False)
                                if eot_positions.numel() > 0:
                                    end_pos = int(eot_positions[0].item())
                            # 仅在 [1, end_pos) 范围内进行掩码，跳过起始<sot>与结尾<eot>
                            for t in range(1, max(1, end_pos)):
                                if pred_noise[b, t] == 1 and masked_caption[b, t] != 0:
                                    masked_caption[b, t] = mask_token_id
                        text_feat = model.encode_text(masked_caption)
                    else:
                        text_feat = model.encode_text(caption)
            else:
                with torch.no_grad():
                    text_feat = model.encode_text(caption)
            qids.append(pid.view(-1)) # 展平
            qfeats.append(text_feat)
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)

        return qfeats, gfeats, qids, gids
    
    def eval(self, model, i2t_metric=False):
        # 步骤1：计算所有特征嵌入
        qfeats, gfeats, qids, gids = self._compute_embedding(model)
        # 步骤2：特征归一化
        qfeats = F.normalize(qfeats, p=2, dim=1) # text features
        gfeats = F.normalize(gfeats, p=2, dim=1) # image features
        # 步骤3：计算相似度矩阵
        similarity = qfeats @ gfeats.t()
        # 步骤4：计算文本到图像检索指标
        t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
        t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table.add_row(['t2i', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])
        # 步骤6：可选计算图像到文本检索指标
        if i2t_metric:
            # 转置相似度矩阵：图像作为查询，文本作为候选
            i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=similarity.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True)
            i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
            table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP])
        # table.float_format = '.4'
        table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        self.logger.info('\n' + str(table))
        
        return t2i_cmc[0]
