'''
Reference paper: "Improving Calibration for Long-Tailed Recognition"
Reference code: "https://github.com/dvlab-research/MiSLAS"
'''

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from customKing.modeling.meta_arch.build import META_ARCH_REGISTRY

class MLPClassifier(nn.Module):
    def __init__(self, cfg):
        super(MLPClassifier, self).__init__()
        assert len(cfg.DATASETS.INPUTE_SIZE) == 1, "Only supports one-dimensional data!"

        self.backbone = nn.Sequential(nn.Linear(cfg.DATASETS.INPUTE_SIZE[0], cfg.DATASETS.INPUTE_SIZE[0]),
                                       nn.Sigmoid(),
                                       nn.Linear(cfg.DATASETS.INPUTE_SIZE[0], 2*cfg.DATASETS.INPUTE_SIZE[0]),
                                       nn.Sigmoid())
        self.class_head = nn.Linear(2*cfg.DATASETS.INPUTE_SIZE[0], cfg.MODEL.NUM_CLASS)


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class LabelAwareSmoothing(nn.Module):
    def __init__(self, cls_num_list, smooth_head=0.2, smooth_tail=0.0, shape='concave', power=None):
        super(LabelAwareSmoothing, self).__init__()

        n_1 = max(cls_num_list)
        n_K = min(cls_num_list)

        if shape == 'concave':
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * np.sin((np.array(cls_num_list) - n_K) * np.pi / (2 * (n_1 - n_K)))

        elif shape == 'linear':
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * (np.array(cls_num_list) - n_K) / (n_1 - n_K)

        elif shape == 'convex':
            self.smooth = smooth_head + (smooth_head - smooth_tail) * np.sin(1.5 * np.pi + (np.array(cls_num_list) - n_K) * np.pi / (2 * (n_1 - n_K)))

        elif shape == 'exp' and power is not None:
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * np.power((np.array(cls_num_list) - n_K) / (n_1 - n_K), power)

        self.smooth = torch.from_numpy(self.smooth)
        self.smooth = self.smooth.float()
        if torch.cuda.is_available():
            self.smooth = self.smooth.cuda()

    def forward(self, x, target):
        smoothing = self.smooth[target]
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()

class LearnableWeightScaling(nn.Module):
    def __init__(self, num_classes):
        super(LearnableWeightScaling, self).__init__()
        self.learned_norm = nn.Parameter(torch.ones(1, num_classes))

    def forward(self, x):
        return self.learned_norm * x

class MiSLAS(nn.Module):
    def __init__(self,cfg,cls_num_list,network) -> None:
        super().__init__()
        self.cfg = cfg
        self.Two_stage = True
        self.cls_num_list = cls_num_list
        self.backbone = network.backbone
        self.class_head = network.class_head
        self.lws_model = LearnableWeightScaling(cfg.MODEL.NUM_CLASS)
        self.need_epoch = False
        self.output_uncertainty = False
    def forward(self,x,y,stage = None):
        if self.training:
            assert stage != None, "This is a two-stage method!"
            if stage == 1:
                criterion = nn.CrossEntropyLoss().to(x.device)
                mixed_x, y_a, y_b, lam = mixup_data(x, y)
                x = self.backbone(mixed_x)
                x = self.class_head(x)
                loss = mixup_criterion(criterion,x,y_a,y_b,lam)
                return x,loss
            elif stage == 2:
                criterion = LabelAwareSmoothing(self.cls_num_list).to(x.device)
                mixed_x, y_a, y_b, lam = mixup_data(x, y)
                x = self.backbone(mixed_x).detach()
                x = self.class_head(x)
                x = self.lws_model(x)
                loss = mixup_criterion(criterion,x,y_a,y_b,lam)
                return x,loss
        else:
            x = self.backbone(x)
            x = self.class_head(x)
            return x

@META_ARCH_REGISTRY.register()
def MLP_MiSLAS(cfg,cls_num_list):
    MLP = MLPClassifier(cfg)
    return MiSLAS(cfg,cls_num_list,MLP)