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