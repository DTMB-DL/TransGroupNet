import torch
import torch.nn as nn

__all__ = ["loss_CE_and_MSE"]


class loss_CE_and_MSE(nn.Module):
    def __init__(self, alpha=0.1):
        super(loss_CE_and_MSE, self).__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(self, pred, label, ori):  # pred(out1,out2)
        loss1 = self.ce(pred[1], label)
        loss2 = self.mse(pred[0], ori)
        loss = self.alpha * loss1 + (1 - self.alpha) * loss2
        Y_pred = torch.argmax(pred[1], 1)
        return loss, Y_pred

    def __call__(self, pred, label, ori=None):
        return self.forward(pred, label, ori)
