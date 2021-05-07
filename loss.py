class ScoreLabelSmoothedCrossEntropyLoss(nn.Module):
    def __init__(self, temp, power, num_classes, device='cuda'):
        super(ScoreLabelSmoothedCrossEntropyLoss, self).__init__()
        self.temp = temp
        self.power = -power
        self.num_classes = num_classes
        self.D = torch.zeros((num_classes, num_classes), device=device)
        for i in range(0, num_classes):
            for j in range(0, num_classes):
                self.D[i, j] = abs(i - j)

    def forward(self, y_hat, y):
        y_hat = F.log_softmax(y_hat, dim=-1)
        y_one_hot = F.one_hot(y, num_classes=self.num_classes)
        y_smooth = (self.D[y] + y_one_hot).pow(self.power) - y_one_hot
        y_smooth /= y_smooth.sum(dim=-1, keepdim=True)
        y_smooth *= self.temp
        y_smooth += (1 - self.temp) * y_one_hot
        loss = (-y_smooth * y_hat).sum(dim=-1).mean()
        return loss