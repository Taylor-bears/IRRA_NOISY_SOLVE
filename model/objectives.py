import torch
import torch.nn as nn
import torch.nn.functional as F

# SDM损失函数 跨模态语义对齐（替代 CLIP 原余弦对比）
def compute_sdm(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    # 构造同/不同身份矩阵
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t() # 通过广播减法，计算每个样本对的pid差值
    labels = (pid_dist == 0).float() #先是返回bool阵，然后再转为float

    # 这里的image_id就是图像的编号，之所以会出现同一个pid，但是imageid不同，因为不同的拍摄角度会使得同一种样本有多个图像
    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        # 此时labels会得到三种情况：同一张图片（1），不同图片但同一身份也就是多角度（factor），不同身份（0）
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    # 归一化，分母是计算两特征的范数（沿第一维度计算，保持维度一致）
    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    # 由于前面已经归一化，所以点积得到的即为余弦相似度，注意t2i与i2t相反
    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    # 温度缩放，放大差距，这是对比学习中常用的，温度过小，模型会极度关注最相似的样本，容易过拟合；过大，相当于不怎么区分，无法区分正负样本
    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # 真相分布归一化成概率分布，这里的归一化是使每一行和为1
    labels_distribute = labels / labels.sum(dim=1)

    # KL散度损失
    # 图像到文本每行相似度作softmax，得到概率分布
    i2t_pred = F.softmax(image_proj_text, dim=1)
    # 乘上对数概率差值，得到交叉熵KL散度
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    # 每行求和再取均值
    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss


# MLM损失函数 图文语义重建
def compute_mlm(scores, labels):
    ce = nn.CrossEntropyLoss(ignore_index=0)
    return ce(scores, labels)


# ITC损失，跨模态对比学习损失函数
def compute_itc(image_features, text_features, logit_scale):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(image_features.device)

    
    # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = logit_scale * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t =F.cross_entropy(logits_per_text, labels)
    loss = (loss_i +  loss_t)/2

    return loss


# ID实例识别损失函数，用于实例级对齐​​
def compute_id(image_logits, text_logits, labels):
    """
    Instance loss proposed at http://arxiv.org/abs/1711.05535
    """
    criterion = nn.CrossEntropyLoss(reduction="mean")

    # 计算图像模态的分类损失和文本模态的分类损失
    loss = criterion(image_logits, labels) + criterion(text_logits, labels)
    
    return loss / 2


# CMPM损失函数 跨模态投影匹配
def compute_cmpm(image_embeddings, text_embeddings, labels, epsilon=1e-8):
    """
    Cross-Modal Projection Matching Loss(CMPM)
    :param image_embeddings: Tensor with dtype torch.float32
    :param text_embeddings: Tensor with dtype torch.float32
    :param labels: Tensor with dtype torch.int32
    :return:
        i2t_loss: cmpm loss for image projected to text
        t2i_loss: cmpm loss for text projected to image
        pos_avg_sim: average cosine-similarity for positive pairs
        neg_avg_sim: averate cosine-similarity for negative pairs
    """

    batch_size = image_embeddings.shape[0]
    labels_reshape = torch.reshape(labels, (batch_size, 1))
    labels_dist = labels_reshape - labels_reshape.t()
    labels_mask = (labels_dist == 0).float()

    image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
    image_proj_text = torch.matmul(image_embeddings, text_norm.t())
    text_proj_image = torch.matmul(text_embeddings, image_norm.t())

    # normalize the true matching distribution
    labels_mask_norm = labels_mask / labels_mask.norm(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + epsilon))

    cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return cmpm_loss