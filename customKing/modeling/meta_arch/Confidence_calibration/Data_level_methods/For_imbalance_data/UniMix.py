'''
Reference paper: "Towards Calibrated Model for Long-Tailed Visual Recognition from Prior Perspective"
Reference code: https://github.com/XuZhengzhuo/Prior-LT/blob/main/core.py
'''


import torch
import numpy as np
from numpy.random import choice
from customKing.modeling.meta_arch.build import META_ARCH_REGISTRY
import torch.nn as nn
import torch.nn.functional as F


def unimix_sampler(batch_size, labels, cls_num_list, tau):
    idx = np.linspace(0, batch_size - 1, batch_size)
    cls_num = np.array(cls_num_list)
    idx_prob = cls_num[labels.cpu().numpy()]
    idx_prob = np.power(idx_prob, tau, dtype=float)
    idx_prob = idx_prob / np.sum(idx_prob)
    idx = choice(idx, batch_size, p=idx_prob)
    idx = torch.Tensor(idx).type(torch.LongTensor)
    return idx


def unimix_factor(labels_1, labels_2, cls_num_list, alpha):
    cls_num_list = np.array(cls_num_list)
    n_i = cls_num_list[labels_1.cpu().numpy()]
    n_j = cls_num_list[labels_2.cpu().numpy()]
    lam = n_j / (n_i + n_j)
    lam = [np.random.beta(alpha, alpha) + t for t in lam]
    lam = np.array([t - 1 if t > 1 else t for t in lam])
    return torch.Tensor(lam).cuda()

@META_ARCH_REGISTRY.register()
class unimix(nn.Module):
    def __init__(self,cfg) -> None:
        super().__init__()
        self.alpha = 0.8
        self.tau = -0.5
        self.cfg = cfg

    def forward(self, images, labels, cls_num_list):

        batch_size = images.size()[0]

        index = unimix_sampler(batch_size, labels, cls_num_list, self.tau)
        images_1, images_2 = images, images[index, :]
        labels_1, labels_2 = labels, labels[index]

        lam = unimix_factor(labels_1, labels_2, cls_num_list, self.alpha)

        mixed_images = torch.zeros_like(images)

        if len(images_1.shape)==4:
            for i, s in enumerate(lam):
                mixed_images[i, :, :, :] = images_1[i, :, :, :] * s + images_2[
                    i, :, :, :] * (1 - s)
        if len(images_1.shape)==3:
            for i, s in enumerate(lam):
                mixed_images[i, :, :] = images_1[i, :, :] * s + images_2[
                    i, :, :] * (1 - s)
        if len(images_1.shape)==2:
            for i, s in enumerate(lam):
                mixed_images[i, :] = images_1[i, :] * s + images_2[
                    i, :] * (1 - s)

        mixed_images = mixed_images[:batch_size].cuda()

        labels_1, labels_2 = labels_1, labels_2[:batch_size]

        return mixed_images, labels_1, labels_2, lam
    
@META_ARCH_REGISTRY.register()
class Bayias_compensated_loss(nn.Module):
    def __init__(self,
                 train_cls_num_list=None,
                 inf_lable_distrbution=None,
                 weight=None):
        super(Bayias_compensated_loss, self).__init__()

        self.weight = weight

        self.train_cnl = train_cls_num_list
        self.prior = np.log(self.train_cnl / sum(self.train_cnl))
        self.prior = torch.from_numpy(self.prior).type(torch.cuda.FloatTensor)

        self.inf = inf_lable_distrbution
        self.inf = np.log(self.inf / sum(self.inf))
        self.inf = torch.from_numpy(self.inf).type(torch.cuda.FloatTensor)

    def forward(self, x, target):
        logits = x + self.prior - self.inf
        loss = F.cross_entropy(logits,
                               target,
                               weight=self.weight,
                               reduction='none')
        return loss