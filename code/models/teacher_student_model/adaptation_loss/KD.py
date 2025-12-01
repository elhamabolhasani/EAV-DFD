import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    """ Distilling the Knowledge in a Neural Network """
    def __init__(self, t, weighted=True):
        super(Loss, self).__init__()
        self.T = float(t)
        self.weighted = weighted
        self.KLDiv = nn.KLDivLoss(reduction="batchmean")

    def forward(self, out_dict):
        y_s, y_t = out_dict['y_s'], out_dict['y_t']
        p_s = F.log_softmax(y_s / self.T, dim=0)
        p_t = F.softmax(y_t / self.T, dim=0)
        KL_distil = (self.T ** 2) * self.KLDiv(p_s, p_t)
        return KL_distil
