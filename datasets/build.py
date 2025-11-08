import logging
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from datasets.sampler import RandomIdentitySampler
from datasets.sampler_ddp import RandomIdentitySampler_DDP
from torch.utils.data.distributed import DistributedSampler

from utils.comm import get_world_size

from .bases import ImageDataset, TextDataset, ImageTextDataset, ImageTextMLMDataset

from .cuhkpedes import CUHKPEDES
from .icfgpedes import ICFGPEDES
from .rstpreid import RSTPReid

__factory = {'CUHK-PEDES': CUHKPEDES, 'ICFG-PEDES': ICFGPEDES, 'RSTPReid': RSTPReid}


def build_transforms(img_size=(384, 128), aug=False, is_train=True):
    """构建图像预处理变换管道
    Args:
        img_size: 图像尺寸 (height, width)
        aug: 是否使用数据增强
        is_train: 是否为训练模式
    """
    height, width = img_size

    # ImageNet标准归一化参数
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    if not is_train: 
        # 验证/测试时的变换：仅调整大小+归一化
        transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return transform

    # 训练时
    if aug:
        # 增强版数据增强：水平翻转+填充+随机裁剪+随机擦除
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5), # 翻转：真实场景中，物体可能出现在左侧或右侧
            T.Pad(10), # 填充
            T.RandomCrop((height, width)), # 裁剪，与上面填充配合，模拟目标位置的多样性
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(scale=(0.02, 0.4), value=mean), # 随机遮挡
        ])
    else:
        # 基础数据增强：仅水平翻转
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    return transform


# 收集函数，将字典列表转换为以键为单位的张量列表
def collate(batch):
    """
    原本的batch是​​列表，每个元素是一个​字典，如下：
    batch = [ 
        {"images": tensor1, "caption_ids": tensor2, "pids": 101},  # 样本1
        {"images": tensor3, "caption_ids": tensor4, "pids": 102},  # 样本2
        # ...  具体的字典形式看bases里的ret格式
    ]，经过collate后变为一个字典​​，每个​​键​​对应一个张量列表​​，将所有样本的该字段堆叠起来。
    {
        "images": torch.stack([tensor1, tensor3, ...]),  # 所有图像的张量
        "caption_ids": torch.stack([tensor2, tensor4, ...]),  # 所有文本的张量
        "pids": torch.tensor([101, 102, ...]),  # 所有ID的张量
    }
    """
    keys = set([key for b in batch for key in b.keys()])
    # turn list of dicts data structure to dict of lists data structure
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
 
    batch_tensor_dict = {}
    for k, v in dict_batch.items():
        if isinstance(v[0], int):
            batch_tensor_dict.update({k: torch.tensor(v)})
        elif torch.is_tensor(v[0]):
             batch_tensor_dict.update({k: torch.stack(v)})
        else:
            raise TypeError(f"Unexpect data type: {type(v[0])} in a batch.")

    return batch_tensor_dict


# 构建数据加载器
def build_dataloader(args, tranforms=None):
    """
    ①随机采样，假设数据集有1000张图，属于100个不同的人，每个ID 10张图，若batch_size=8，
    Batch 1: [ID7, ID32, ID15, ID89, ID4, ID63, ID12, ID55]  # 完全随机
    Batch 2: [ID22, ID1, ID1, ID76, ID90, ID34, ID8, ID19]   # 可能重复ID

    ②身份采样，每个batch固定包含N个不同ID，每个ID采样K个实例，batch_size = N * K，设 batch_size=8, num_instance=2→ 4 个 ID × 2 张图/ID
    Batch 1: [ID7_img1, ID7_img2, ID32_img1, ID32_img2, ID15_img1, ID15_img2, ID89_img1, ID89_img2]
    Batch 2: [ID4_img1, ID4_img2, ID63_img1, ID63_img2, ID12_img1, ID12_img2, ID55_img1, ID55_img2]
    
    ③分布式身份采样，协调各 GPU 的采样，保证全局 batch 仍符合 N个ID × K个实例，假设 2 个 GPU，batch_size=8（单卡 mini_batch_size=4），num_instance=2
    GPU 1​​: [ID7_img1, ID7_img2, ID32_img1, ID32_img2]
​    ​GPU 2​​: [ID15_img1, ID15_img2, ID89_img1, ID89_img2]
    """
    logger = logging.getLogger("IRRA.dataset")

    num_workers = args.num_workers
    dataset = __factory[args.dataset_name](root=args.root_dir)
    num_classes = len(dataset.train_id_container)
    
    if args.training: # 训练模式
        # 构建训练和验证的图像变换
        train_transforms = build_transforms(img_size=args.img_size,
                                            aug=args.img_aug,
                                            is_train=True)
        val_transforms = build_transforms(img_size=args.img_size,
                                          is_train=False)
        # 根据是否启用MLM选择不同的数据集类（来自于bases定义）
        if args.MLM:
            train_set = ImageTextMLMDataset(dataset.train,
                                     train_transforms,
                                     text_length=args.text_length)
        else:
            train_set = ImageTextDataset(dataset.train,
                                     train_transforms,
                                     text_length=args.text_length)
        
        # 使用 "identity" 采样器，按身份/ID采样
        if args.sampler == 'identity': 
            if args.distributed: # 分布式训练模式
                logger.info('using ddp random identity sampler')
                logger.info('DISTRIBUTED TRAIN START')
                mini_batch_size = args.batch_size // get_world_size() # 单卡分配的batch大小
                # TODO wait to fix bugs
                # 身份采样器：确保每个batch包含多个ID，每个ID的多个实例
                data_sampler = RandomIdentitySampler_DDP(
                    dataset.train, args.batch_size, args.num_instance)
                # 批量采样器：将采样结果按mini_batch_size分块
                batch_sampler = torch.utils.data.sampler.BatchSampler(
                    data_sampler, mini_batch_size, True)
            # 单机训练模式
            else:
                logger.info(
                    f'using random identity sampler: batch_size: {args.batch_size}, id: {args.batch_size // args.num_instance}, instance: {args.num_instance}'
                )
                train_loader = DataLoader(train_set,
                                          batch_size=args.batch_size,
                                          sampler=RandomIdentitySampler(
                                              dataset.train, args.batch_size,
                                              args.num_instance),
                                          num_workers=num_workers,
                                          collate_fn=collate)
        # 使用随机采样器
        elif args.sampler == 'random':
            # TODO add distributed condition
            logger.info('using random sampler')
            # 完全随机打乱数据，无ID约束
            train_loader = DataLoader(train_set,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      collate_fn=collate)
        else:
            logger.error('unsupported sampler! expected softmax or triplet but got {}'.format(args.sampler))


        # 构建验证集（使用测试集或专用验证集）
        ds = dataset.val if args.val_dataset == 'val' else dataset.test
        # 验证集图像和文本数据集
        val_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],
                                   val_transforms)
        val_txt_set = TextDataset(ds['caption_pids'],
                                  ds['captions'],
                                  text_length=args.text_length)
        # 验证集数据加载器
        val_img_loader = DataLoader(val_img_set,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)
        val_txt_loader = DataLoader(val_txt_set,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)

        return train_loader, val_img_loader, val_txt_loader, num_classes

    else: # 测试模式
        # 构建测试变换
        if tranforms:
            test_transforms = tranforms
        else:
            test_transforms = build_transforms(img_size=args.img_size,
                                               is_train=False)

        ds = dataset.test
        # 测试集图像和文本数据集
        test_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],
                                    test_transforms)
        test_txt_set = TextDataset(ds['caption_pids'],
                                   ds['captions'],
                                   text_length=args.text_length)
        # 测试集数据加载器
        test_img_loader = DataLoader(test_img_set,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)
        test_txt_loader = DataLoader(test_txt_set,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)
        return test_img_loader, test_txt_loader, num_classes
