import torch
import torch.nn as nn

__all__ = ["loss_CE"]


class loss_CE(nn.Module):
    def __init__(self):
        super(loss_CE, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, label, ori):
        loss = self.ce(pred, label)
        Y_pred = torch.argmax(pred, 1)
        return loss, Y_pred

    def __call__(self, pred, label, ori=None):
        return self.forward(pred, label, ori)

