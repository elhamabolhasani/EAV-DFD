import torch.nn as nn
import torch

class Loss(nn.Module):
    """ MSE Loss """
    def __init__(self, temperature):
        super(Loss, self).__init__()
        self.temperature = temperature

    def forward(self, teacher_logits, student_logits):
        soft_targets = nn.functional.softmax(teacher_logits / self.temperature, dim=-1)
        soft_prob = nn.functional.log_softmax(student_logits / self.temperature, dim=-1)
        soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (self.temperature**2)
        return soft_targets_loss