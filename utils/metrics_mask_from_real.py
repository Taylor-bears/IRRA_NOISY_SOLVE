from prettytable import PrettyTable
import torch
import torch.nn.functional as F
import logging
import regex as re
import difflib
from utils.simple_tokenizer import SimpleTokenizer


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        _, indices = torch.topk(similarity, k=max_rank, dim=1, largest=True, sorted=True)
    pred_labels = g_pids[indices.cpu()]
    matches = pred_labels.eq(q_pids.view(-1, 1))

    all_cmc = matches[:, :max_rank].cumsum(1)
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)
    tmp_cmc = matches.cumsum(1)
    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.0) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices


class Evaluator:
    """
    构造“真实差异掩码”的评估器：
    - 若 noisy 与 clean 文本完全一致：mask 文本 = 原 noisy 文本（不做掩码）
    - 若不一致：用 difflib 对齐找出 noisy 相对 clean 的差异词，将对应 token span 置为 <|mask|>

    不使用图像上下文与噪声头，前向路径与 IRRA 基线一致，仅换一套“mask 文本”。
    输出三路：t2i-noisy, t2i-clean(若有), t2i-mask。
    """

    def __init__(self, img_loader, txt_loaders, mask_noise=False):
        self.img_loader = img_loader
        self.txt_loaders = txt_loaders if isinstance(txt_loaders, dict) else {'noisy': txt_loaders}
        self.logger = logging.getLogger("IRRA.eval")
        self._tokenizer = SimpleTokenizer()
        self._mask_id = self._tokenizer.encoder.get("<|mask|>")
        self._eot_id = self._tokenizer.encoder.get("<|endoftext|>")
        self.mask_noise = mask_noise

    def _compute_image_embeddings(self, model):
        model = model.eval()
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

    def _mask_caption_from_real_diff(self, noisy_text: str, clean_text: str, cap_tokens: torch.Tensor) -> torch.Tensor:
        """
        基于 noisy vs clean 的真实差异，在 cap_tokens 上将对应 token 位置替换为 <|mask|>。
        若二者相同，则直接返回原 tokens。
        """
        if noisy_text == clean_text:
            return cap_tokens
        # 1) 词级对齐
        word_pattern = r"[A-Za-z0-9]+|[^\sA-Za-z0-9]+"
        clean_words = re.findall(word_pattern, clean_text)
        noisy_words = re.findall(word_pattern, noisy_text)
        matcher = difflib.SequenceMatcher(a=clean_words, b=noisy_words)
        noise_word_flags = [0] * len(noisy_words)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag in ('replace', 'insert'):
                for j in range(j1, j2):
                    if 0 <= j < len(noisy_words):
                        noise_word_flags[j] = 1
        # 2) 将 noisy_words 映射到 token span
        noisy_token_ids = []
        word_to_token_span = []
        cur = 0
        for w in noisy_words:
            t_ids = self._tokenizer.encode(w)
            noisy_token_ids.extend(t_ids)
            word_to_token_span.append((cur, cur + len(t_ids)))
            cur += len(t_ids)
        # 3) 基于 span 在 cap_tokens 中替换为 <|mask|>
        masked = cap_tokens.clone()
        L = int(masked.size(0))
        # 查找 eot 截止位置
        end_pos = L - 1
        if self._eot_id is not None:
            eot_positions = (masked == self._eot_id).nonzero(as_tuple=False)
            if eot_positions.numel() > 0:
                end_pos = int(eot_positions[0].item())
        start = 1
        if end_pos <= start:
            return masked
        for wi, flag in enumerate(noise_word_flags):
            if flag == 1 and wi < len(word_to_token_span):
                t_start = 1 + word_to_token_span[wi][0]
                t_end = 1 + word_to_token_span[wi][1]
                t_start = min(t_start, L)
                t_end = min(t_end, L)
                if t_start < t_end:
                    # 仅在有效范围内替换
                    t_end_clip = min(t_end, end_pos)
                    if t_start < t_end_clip and self._mask_id is not None:
                        masked[t_start:t_end_clip] = self._mask_id
        return masked

    def eval(self, model, i2t_metric=False):
        model = model.eval()
        device = next(model.parameters()).device

        # 图库特征一次性提取
        gfeats, gids = self._compute_image_embeddings(model)
        gfeats = F.normalize(gfeats, p=2, dim=1)

        routes = []
        if 'noisy' in self.txt_loaders:
            routes.append(('t2i-noisy', self.txt_loaders['noisy']))
        if 'clean' in self.txt_loaders:
            routes.append(('t2i-clean', self.txt_loaders['clean']))
        # t2i-mask 基于 noisy 与 clean 的真实差异构造
        do_mask = ('noisy' in self.txt_loaders) and ('clean' in self.txt_loaders) and self.mask_noise

        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])

        # 评估 noisy/clean
        for label, txt_loader in routes:
            qids, qfeats = [], []
            for pid, caption in txt_loader:
                caption = caption.to(device)
                with torch.no_grad():
                    text_feat = model.encode_text(caption)
                qids.append(pid.view(-1))
                qfeats.append(text_feat)
            qids = torch.cat(qids, 0)
            qfeats = F.normalize(torch.cat(qfeats, 0), p=2, dim=1)
            similarity = qfeats @ gfeats.t()
            t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
            t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
            table.add_row([label, t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])

        # 评估 t2i-mask（真实差异掩码）
        if do_mask:
            noisy_ds = self.txt_loaders['noisy'].dataset
            clean_ds = self.txt_loaders['clean'].dataset
            qids, qfeats = [], []
            global_idx = 0
            for pid, caption in self.txt_loaders['noisy']:
                caption = caption.to(device)
                B, L = caption.size()
                masked_batch = caption.clone()
                for b in range(B):
                    noisy_txt = noisy_ds.captions[global_idx + b]
                    clean_txt = clean_ds.captions[global_idx + b]
                    masked_batch[b] = self._mask_caption_from_real_diff(noisy_txt, clean_txt, caption[b])
                global_idx += B
                with torch.no_grad():
                    text_feat = model.encode_text(masked_batch)
                qids.append(pid.view(-1))
                qfeats.append(text_feat)
            qids = torch.cat(qids, 0)
            qfeats = F.normalize(torch.cat(qfeats, 0), p=2, dim=1)
            similarity = qfeats @ gfeats.t()
            t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
            t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
            table.add_row(['t2i-mask', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])
        else:
            table.add_row(['t2i-mask', "--", "--", "--", "--", "--"])  # 尚未开启mask文本时

        # 打印表
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

        try:
            # 返回第一行(noisy)的R1作为主指标
            return float(table._rows[0][1])
        except Exception:
            return 0.0
