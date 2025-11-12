from typing import List
from torch.utils.data import Dataset
import os.path as osp
import logging
import torch
from utils.iotools import read_image
from utils.simple_tokenizer import SimpleTokenizer
from prettytable import PrettyTable
import random
import regex as re
import copy
from utils.iotools import read_json
import difflib

class BaseDataset(object):
    """
    图文检索/重识别数据集的基类，提供基础数据结构和通用方法
    """
    logger = logging.getLogger("IRRA.dataset")

    def show_dataset_info(self):
        """
        打印数据集统计信息（训练集/验证集/测试集的ID数量、图像数量、文本数量）
        使用PrettyTable格式化为表格输出，便于直观查看数据分布
        """
        num_train_pids, num_train_imgs, num_train_captions = len(
            self.train_id_container), len(self.train_annos), len(self.train)
        num_test_pids, num_test_imgs, num_test_captions = len(
            self.test_id_container), len(self.test_annos), len(
                self.test['captions'])
        num_val_pids, num_val_imgs, num_val_captions = len(
            self.val_id_container), len(self.val_annos), len(
                self.val['captions'])

        # TODO use prettytable print comand line table

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(
            ['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(
            ['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        self.logger.info('\n' + str(table))


def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    """
    将文本描述转换为token序列，并填充/截断到固定长度【用到utils定义的SimpleTokenizer()工具】
    """
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    # 将文本转换为token序列，并添加起始/结束标记
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]
    # 初始化全零张量
    result = torch.zeros(text_length, dtype=torch.long)
    # 处理超长文本，采用截断策略
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    # 将token序列填充到结果张量中，如果不够长则保持后续部分为0
    result[:len(tokens)] = torch.tensor(tokens)
    return result


class ImageTextDataset(Dataset):
    """
    图文匹配数据集类，实现将原始数据转换为模型可处理的格式（图像+文本token）
    """
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # 解包原始数据（假设每个样本是四元组）
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)
         # 应用图像变换（比如转为张量)
        if self.transform is not None:
            img = self.transform(img)

        tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        # 封装为字典格式，ret指return
        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': tokens,
        }

        return ret


class ImageDataset(Dataset):
    """
    纯图像数据集加载类，返回图像及其对应的身份标签
    """
    def __init__(self, image_pids, img_paths, transform=None):
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_pids)
    # 需要注意，不同的index，一定对应不同的img_Path，但pid可能相同，因为不同图片可能表示的是同一个身份
    def __getitem__(self, index):
        pid, img_path = self.image_pids[index], self.img_paths[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return pid, img


class TextDataset(Dataset):
    """
    文本数据集加载类，返回文本描述及其对应的身份标签
    主要功能：将原始文本转换为token序列，并统一到固定长度
    """
    def __init__(self,
                 caption_pids,
                 captions,
                 text_length: int = 77,
                 truncate: bool = True):
        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, caption = self.caption_pids[index], self.captions[index]

        caption = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        return pid, caption


class ImageTextMLMDataset(Dataset):
    """
    图文数据集类（带MLM任务），
    同时支持：图像-文本匹配任务和掩码语言建模（MLM）任务
    返回包含图像、原始文本token和掩码文本token的字典
    """
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate

        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        
        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        # 生成掩码文本和标签，用于MLM任务
        mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(caption_tokens.cpu().numpy())

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': caption_tokens,
            'mlm_ids': mlm_tokens,
            'mlm_labels': mlm_labels
        }

        return ret
    
    # 构建随机掩码token序列和标签的方法
    def _build_random_masked_tokens_and_labels(self, tokens):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405
        
        labels = []
        # tokens是一个ID序列，token是某一个id，这里用mask对随机id替换
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random() #生成一个​[0.0, 1.0)内的浮点数
                # 15%概率选中当前token进行掩码
                if prob < 0.15:
                    prob /= 0.15 # 归一化到[0,1]

                    # 80%概率替换为[MASK]
                    if prob < 0.8:
                        tokens[i] = mask

                    # 10%概率替换为随机token,从 token_range列表中​随机选择一个元素
                    elif prob < 0.9:
                        tokens[i] = random.choice(token_range)

                    # 剩余10%保持原token不变

                    # 记录原始token作为标签
                    labels.append(token)
                else:
                    # 非掩码位置标签为0（计算损失时忽略）
                    labels.append(0)
            else:
                labels.append(0) # 特殊token不参与MLM
        
        # 确保至少掩码一个token（避免全零标签）
        if all(l == 0 for l in labels):
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens), torch.tensor(labels)


# new
class ImageTextNoiseDetectionDataset(Dataset):
    """
    用于噪声词检测任务的图文数据集。
    直接读取包含 clean captions 与 noisy captions (captions_rw) 的训练 JSON。

    说明：为了避免依赖外部属性词词表，初版实现采用 clean/noisy token 的逐位差异作为监督信号，
    仅在差异位置参与监督，满足“尽可能只在属性相关处优化”的目标；后续可加入更细粒度的属性词mask。
    """
    # 这里我们没有直接用dataset，是因为cuhkpedes产生的dataset是基于reid_raw的，里面并没有captions_rw字段，所以这里init阶段，我们需要重新自己构建samples内容（也就是dataset）
    def __init__(self,
                 noisy_json_path: str,
                 img_dir: str,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        super().__init__()
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

        annos = read_json(noisy_json_path)
        # 只保留训练split，且要求同时包含 captions 与 captions_rw
        train_annos = [a for a in annos if a.get('split') == 'train' and 'captions' in a and 'captions_rw' in a]

        self.samples = [] # 存储所有样本的列表
        image_id = 0
        for anno in train_annos:
            pid = int(anno['id']) - 1  # 训练pid从0开始
            img_path = osp.join(img_dir, anno['file_path'])
            clean_caps = anno['captions']
            noisy_caps = anno['captions_rw']
            m = min(len(clean_caps), len(noisy_caps)) # 对齐：按索引一一对应
            # training时的dataset应当将captions全部展开为独立样本，这与cuhkpedes.py中train下的dataset思路一致
            for i in range(m):
                self.samples.append((pid, image_id, img_path, clean_caps[i], noisy_caps[i]))
            image_id += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        pid, image_id, img_path, clean_cap, noisy_cap = self.samples[index]

        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

            # 为了更稳健地识别哪些词被替换，我们首先在词(level)层面进行对齐，
            # 然后将被判定为替换的词映射回它们对应的token id范围，生成token-level的noise_labels。

            # 1) 词级切分（保留字母/数字的词）
            word_pattern = r"[A-Za-z0-9]+|[^\sA-Za-z0-9]+" # 
            clean_words = re.findall(word_pattern, clean_cap)
            noisy_words = re.findall(word_pattern, noisy_cap)

            # 2) 使用SequenceMatcher对词序列进行对齐，标记 noisy_words 中的 replace/insert 操作为噪声
            matcher = difflib.SequenceMatcher(a=clean_words, b=noisy_words)
            noise_word_flags = [0] * len(noisy_words) # 初始化全0标记
            # tag的类型可以是'replace','delete','insert','equal'，[i1,i2]是clean_words中的索引范围，[j1,j2]是noisy_words中的索引范围，举例运行过程
            # 开始部分相同，('equal', 0, 10, 0, 10) "The man is facing away and walking forward. He has a"
            # # 第一个差异：black backpack → brown shoulder bag ('replace', 10, 12, 10, 13)  clean[10:12] = ['black', 'backpack']  noisy[10:13] = ['brown', 'shoulder', 'bag']
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                # tag in ('replace','delete','insert','equal')
                if tag == 'replace': # 我们当前的模式只用到replace
                    # noisy words in [j1, j2) are considered modified/noisy
                    for j in range(j1, j2):
                        noise_word_flags[j] = 1 

            # 3) 将 noisy_words 按词逐个 token 化，并记录每个词对应的 token id 范围
            sot = self.tokenizer.encoder.get("<|startoftext|>")
            eot = self.tokenizer.encoder.get("<|endoftext|>")

            noisy_token_ids = []
            word_to_token_span = []  # list of (start_idx, end_idx) in noisy_token_ids for each noisy_word
            cur = 0
            for w in noisy_words:
                # encode 单词
                token_ids = self.tokenizer.encode(w)
                noisy_token_ids.extend(token_ids)
                word_to_token_span.append((cur, cur + len(token_ids)))
                cur += len(token_ids)
            # 上面这一步可以得到 noisy_token_ids = [1000, 2000, 3000, 4000, 5000]word_to_token_span = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]

            # 构造带special tokens的最终token序列，并截断/填充到text_length
            tokens = [sot] + noisy_token_ids + [eot]
            if len(tokens) > self.text_length:
                tokens = tokens[:self.text_length]
                tokens[-1] = eot
            tokens_tensor = torch.zeros(self.text_length, dtype=torch.long)
            tokens_tensor[:len(tokens)] = torch.tensor(tokens, dtype=torch.long)

            # 4) 生成 token-level 的 noise_labels（默认全0），并将被标注为噪声的词对应的token位置设为1
            # 前面的noise_word_flags是以词为单位标注的，但是一个word可能对应多个token，所以这里我们构建token级的noise_labels用到了word_to_token_span
            noise_labels = torch.zeros(self.text_length, dtype=torch.long)
            for wi, flag in enumerate(noise_word_flags):
                if flag == 1 and wi < len(word_to_token_span):
                    start, end = word_to_token_span[wi]
                    # token indices in tokens_tensor are offset by +1 because of sot at position 0
                    t_start = 1 + start # 这里的1是因为开头的<sot>占了一个位置
                    t_end = 1 + end
                    # clip to text_length
                    t_start = min(t_start, self.text_length)
                    t_end = min(t_end, self.text_length)
                    if t_start < t_end:
                        noise_labels[t_start:t_end] = 1

            # attribute_mask：目前与noise_labels一致（仅在检测到差异的token上计算损失）
            attribute_mask = noise_labels.clone().float()

            noisy_tokens = tokens_tensor

            # 额外：构造 clean 文本的 token 序列（用于一致性损失）
            clean_tokens = tokenize(clean_cap, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': noisy_tokens,     # 输入使用带噪声的tokens
            'noise_labels': noise_labels,    # 监督：1=噪声token位置
            'attribute_mask': attribute_mask, # 仅在差异位置计算loss
            'clean_caption_ids': clean_tokens # clean 文本tokens，用于一致性损失
        }

        return ret