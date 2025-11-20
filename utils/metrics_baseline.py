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


class Evaluator:
    """Baseline评估器：兼容新的多路文本loader结构（dict: noisy/clean）。
    仅做标准图文相似度检索，不做噪声预测与掩码。
    行为：
      - 若传入 txt_loader 为 dict，则分别计算每一路并输出多行表格（t2i-noisy / t2i-clean）。
      - 若仅有单一路，则输出一行 t2i。
      - 返回 clean 路的 R1 （若存在），否则返回第一条的 R1 作为 top1。
    """
    def __init__(self, img_loader, txt_loader):
        self.img_loader = img_loader  # gallery
        # 统一转成dict结构，便于后续多路循环
        if isinstance(txt_loader, dict):
            self.txt_loaders = txt_loader
        else:
            self.txt_loaders = { 't2i': txt_loader }
        self.logger = logging.getLogger("IRRA.eval")

    def _compute_image_feats(self, model):
        device = next(model.parameters()).device
        gids, gfeats = [], []
        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img)
            gids.append(pid.view(-1))
            gfeats.append(img_feat)
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)
        return gfeats, gids

    def _compute_text_feats(self, model, loader):
        device = next(model.parameters()).device
        qids, qfeats = [], []
        for pid, caption in loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_text(caption)
            qids.append(pid.view(-1))
            qfeats.append(text_feat)
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)
        return qfeats, qids

    def eval(self, model, i2t_metric=False):
        model.eval()
        gfeats, gids = self._compute_image_feats(model)
        gfeats = F.normalize(gfeats, p=2, dim=1)

        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        top1_return = None
        # 遍历每一路文本loader
        for name, loader in self.txt_loaders.items():
            qfeats, qids = self._compute_text_feats(model, loader)
            qfeats = F.normalize(qfeats, p=2, dim=1)
            similarity = qfeats @ gfeats.t()
            cmc, mAP, mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
            cmc, mAP, mINP = cmc.numpy(), mAP.numpy(), mINP.numpy()
            row_name = 't2i' if name == 't2i' and len(self.txt_loaders) == 1 else f't2i-{name}'
            table.add_row([row_name, cmc[0], cmc[4], cmc[9], mAP, mINP])
            if top1_return is None or name == 'clean':
                top1_return = cmc[0]

        if i2t_metric and 't2i' in [r[0] for r in table._rows]:
            # 可选的 i2t 反向检索，仅在单一路时启用；多路文本意义不大
            if len(self.txt_loaders) == 1:
                # 重新计算单一路的文本特征（已在循环中算过，可复用；这里简单重算保持清晰）
                only_loader = list(self.txt_loaders.values())[0]
                qfeats, qids = self._compute_text_feats(model, only_loader)
                qfeats = F.normalize(qfeats, p=2, dim=1)
                similarity = qfeats @ gfeats.t()
                i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=similarity.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True)
                i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
                table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP])

        # prettytable格式化
        table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        self.logger.info('\n' + str(table))
        return top1_return if top1_return is not None else 0.0
