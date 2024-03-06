# from https://github.com/Virusdoll/Active-Negative-Loss/blob/main/loss.py
import torch
import torch.nn.functional as F

class MeanAbsoluteError(torch.nn.Module):
    def __init__(self, num_classes) -> None:
        super(MeanAbsoluteError, self).__init__()
        self.num_classes = num_classes
    
    def forward(self, pred, labels):
        # assume labels are also NxC and sum to 1
        # assume device management is done by torch
        pred =  F.softmax(pred, dim=1)
        mae = 1. - torch.sum(labels * pred, dim=1)
        return mae.mean()

class RevserseCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
    
    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_clamp = torch.clamp(labels, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_clamp), dim=1))
        return rce.mean()

class NormalizedCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes) -> None:
        super(NormalizedCrossEntropy, self).__init__()
        self.num_classes = num_classes
    
    def forward(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        nce = -1 * torch.sum(labels * pred, dim=1) / (- pred.sum(dim=1))
        return nce.mean()
    
class NormalizedNegativeCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.min_prob = 1e-7
        self.A = - torch.tensor(self.min_prob).log()
    
    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = pred.clamp(min=self.min_prob, max=1)
        pred = self.A + pred.log() # - log(1e-7) - (- log(p(k|x)))
        nnce = 1 - (labels * pred).sum(dim=1) / pred.sum(dim=1)
        return nnce.mean()

class ActivePassiveLoss(torch.nn.Module):
    def __init__(self, active_loss, passive_loss,
                 alpha=1., beta=1.) -> None:
        super(ActivePassiveLoss, self).__init__()
        self.active_loss = active_loss
        self.passive_loss = passive_loss
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, pred, labels):
        return self.alpha * self.active_loss(pred, labels) \
            + self.beta * self.passive_loss(pred, labels)

class ActiveNegativeLoss(torch.nn.Module):
    def __init__(self, active_loss, negative_loss,
                 alpha=1., beta=1.) -> None:
        super().__init__()
        self.active_loss = active_loss
        self.negative_loss = negative_loss
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, pred, labels):
        al = self.active_loss(pred, labels)
        nl = self.negative_loss(pred, labels)
        
        loss = self.alpha * al + self.beta * nl
        
        return loss

# Help Function

def _apl(active_loss, passive_loss):
    return ActivePassiveLoss(active_loss,
                             passive_loss,
                             alpha = 1., #config['alpha'],
                             beta = 1.) #config['beta'])

def _anl(active_loss, negative_loss):
    return ActiveNegativeLoss(active_loss,
                              negative_loss,
                              alpha = 1., #config['alpha'],
                              beta = 1.) #config['beta'],
                              #delta = config['delta'])

# Loss

def mae(num_classes):
    return MeanAbsoluteError(num_classes)

def ce():
    return CrossEntropy()

def rce(num_classes):
    return RevserseCrossEntropy(num_classes)

def nce(num_classes):
    return NormalizedCrossEntropy(num_classes)

def nnce(num_classes):
    return NormalizedNegativeCrossEntropy(num_classes)
    
def nce_rce(num_classes):
    return _apl(nce(num_classes), rce(num_classes))

# Active Negative Loss

def anl_ce(num_classes):
    return _anl(nce(num_classes), nnce(num_classes))
