import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import Sequential
import thop
'''
reference: https://github.com/jcz111/AvgNet/blob/79dfc5b75b18d19fe59c565abd532304227777d1/baseline/MCLDNN/mcldnn.py
'''

__all__ = ["CM_MCLDNN", "test"]


class CM_MCLDNN(nn.Module):

    def __init__(self, num_classes=25):
        super(CM_MCLDNN, self).__init__()
        self.fc1 = nn.Linear(1024 * 2, 80)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(80, 2)

        self.conv1 = nn.Conv1d(
            in_channels=2,
            out_channels=50,
            kernel_size=7,
            bias=False,
            padding=3,
        )
        self.conv2 = Sequential(
            nn.Conv1d(
            in_channels=2,
            out_channels=100,
            kernel_size=7,
            bias=False,
            padding=3,
            groups=2
        ),
            nn.ReLU(True),
            nn.Conv1d(
            in_channels=100,
            out_channels=50,
            kernel_size=7,
            bias=False,
            padding=3,
        ))
        self.conv3 = nn.Conv1d(
            in_channels=100,
            out_channels=100,
            kernel_size=5,
            bias=False
        )
        self.lstm1 = nn.LSTM(
            input_size=100,
            hidden_size=128,
            num_layers=1,
            bias=False,
        )
        self.lstm2 = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            bias=False,
            batch_first=True
        )
        self.fc = Sequential(
            nn.Linear(128, 128),
            nn.SELU(True),
            nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.SELU(True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        assert len(x.shape) == 3 and x.shape[1] == 2
        ####################  CM  start ############################
        x1 = self.tanh(self.fc1(x.reshape(x.shape[0], -1)))
        x2 = self.fc2(x1)
        w = x2[:, 0].reshape(x.shape[0], -1)
        phi = x2[:, 1].reshape(x.shape[0], -1)

        n = torch.arange(0, 1024).reshape(1, -1).to(x.device).float()
        R = x[:, 0]
        I = x[:, 1]

        nR = R * torch.cos(w * n + phi) + I * torch.sin(w * n + phi)
        nI = I * torch.cos(w * n + phi) - R * torch.sin(w * n + phi)
        x3 = torch.stack((nR, nI), dim=1)

        x = x3
        ####################  CM  end ############################

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = F.relu(torch.cat([x1,x2],dim=1))
        x3 = F.relu(self.conv3(x3))
        x3, _ = self.lstm1(x3.transpose(2,1))
        _, (x3, __) = self.lstm2(x3)
        x3 = self.fc(x3.squeeze())

        return x3


def test():
    x = torch.randn(1, 2, 1024)
    net = CM_MCLDNN(25)
    flops, params = thop.profile(net, inputs=(x,), verbose=False)
    flops, params = thop.clever_format([flops, params], "%.3f")
    print("flops & params", [flops, params])


if __name__ == "__main__":
    test()
