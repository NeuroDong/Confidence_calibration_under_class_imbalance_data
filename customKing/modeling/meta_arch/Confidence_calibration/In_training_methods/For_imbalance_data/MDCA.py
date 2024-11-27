import torch
import torch.nn as nn
import logging
from customKing.modeling.meta_arch.build import META_ARCH_REGISTRY

# MLP classifier
class MLPClassifier(nn.Module):
    def __init__(self, cfg):
        super(MLPClassifier, self).__init__()
        assert len(cfg.DATASETS.INPUTE_SIZE) == 1, "Only supports one-dimensional data!"
        self.fc1 = nn.Linear(cfg.DATASETS.INPUTE_SIZE[0], cfg.DATASETS.INPUTE_SIZE[0]) 
        self.activation = nn.Sigmoid()  
        self.fc2 = nn.Linear(cfg.DATASETS.INPUTE_SIZE[0], 2*cfg.DATASETS.INPUTE_SIZE[0]) 
        self.fc3 = nn.Linear(2*cfg.DATASETS.INPUTE_SIZE[0], cfg.MODEL.NUM_CLASS) 

    def _forward_impl(self,x):
        out = self.activation(self.fc1(x))
        out = self.activation(self.fc2(out))
        predict = self.fc3(out)
        return predict

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.activation(self.fc2(out))
        predict = self.fc3(out)
        return predict

class LabelSmoothingLoss(nn.Module):
    def __init__(self, alpha=0.0, dim=-1, **kwargs):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - alpha
        self.alpha = alpha
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        num_classes = pred.shape[self.dim]
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.alpha / (num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, **kwargs):
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        logging.info("using gamma={}".format(gamma))

    def forward(self, input, target):

        target = target.view(-1,1)

        logpt = torch.nn.functional.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        loss = -1 * (1-pt)**self.gamma * logpt
        
        return loss.mean()

class CrossEntropy(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(CrossEntropy, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input, target):
        return self.criterion(input, target)

class MDCA(torch.nn.Module):
    def __init__(self):
        super(MDCA,self).__init__()

    def forward(self , output, target):
        output = torch.softmax(output, dim=1)
        # [batch, classes]
        loss = torch.tensor(0.0).cuda()
        batch, classes = output.shape
        for c in range(classes):
            avg_count = (target == c).float().mean()
            avg_conf = torch.mean(output[:,c])
            loss += torch.abs(avg_conf - avg_count)
        denom = classes
        loss /= denom
        return loss

class ClassficationAndMDCA(nn.Module):
    def __init__(self, cfg, network, alpha=0.1, beta=1.0, gamma=1.0):
        super(ClassficationAndMDCA, self).__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.classification_loss = FocalLoss(gamma=self.gamma)
        self.MDCA = MDCA()
        self.classifier = network

        self.need_epoch = False
        self.output_uncertainty = False
        self.Two_stage = False


    def forward(self, x, targets):
        logits = self.classifier(x)

        if self.training:
            loss_cls = self.classification_loss(logits, targets)
            loss_cal = self.MDCA(logits, targets)
            loss = loss_cls + self.beta * loss_cal
            return logits,loss
        else:
            return logits
    
@META_ARCH_REGISTRY.register()
def MLP_MDCA(cfg):
    MLP = MLPClassifier(cfg)
    return ClassficationAndMDCA(cfg,MLP)