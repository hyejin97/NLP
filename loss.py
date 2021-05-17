import torch
import torch.nn.functional as F
from torch import nn

device = torch.device("cuda")

class ScoreLabelSmoothedCrossEntropyLoss(nn.Module):
    '''
        Assuming that even number is female and odd number is male 
    '''
    def __init__(self, temp, num_classes, device='cuda'):
        super(ScoreLabelSmoothedCrossEntropyLoss, self).__init__()
        self.temp = temp
        self.num_classes = num_classes

    def forward(self, y_hat, y):
        y_hat = F.log_softmax(y_hat, dim=-1)
        y_one_hot = F.one_hot(y, num_classes=self.num_classes).to(device)
        y_smooth = torch.zeros((y.size()[0], self.num_classes)).to(device)
        for b in range(y.size()[0]):
            for i in range(y[b].item() % 2, self.num_classes, 2):
                if i != y[b].item():
                    y_smooth[b][i] = self.temp / (int(self.num_classes / 2) - 1)

        y_smooth += (1 - self.temp) * y_one_hot
        loss = (-y_smooth * y_hat).sum(dim=-1).mean()
        return loss
