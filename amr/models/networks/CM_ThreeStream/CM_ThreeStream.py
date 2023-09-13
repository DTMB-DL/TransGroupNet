import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import thop
import math
__all__ = ["CM_ThreeStream", "test"]


class CM_ThreeStream(nn.Module):
    def __init__(self, classes):
        super(CM_ThreeStream, self).__init__()
        self.fc1 = nn.Linear(1024 * 3, 80)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(80, 2)

        self.signallen = 1024
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=2),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, 3, padding=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(self.signallen+3),
            nn.Conv1d(64, 64, 3, padding=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(self.signallen + 4),
            nn.Conv1d(64, 64, 3, padding=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(self.signallen + 5),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=2),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(64, 64, 3, padding=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(self.signallen+3),
            nn.Conv1d(64, 64, 3, padding=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(self.signallen + 4),
            nn.Conv1d(64, 64, 3, padding=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(self.signallen + 5),
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=2),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv1d(64, 64, 3, padding=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(self.signallen + 3),
            nn.Conv1d(64, 64, 3, padding=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(self.signallen + 4),
            nn.Conv1d(64, 64, 3, padding=2),
            nn.ReLU(),
        )

        self.conv7 = nn.Sequential(
            nn.Conv1d(128, 64, 3, padding=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(self.signallen + 6),
        )

        self.lstm1 = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, bias=False)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, bias=False, batch_first=True)

        self.fc3 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.fc4 = nn.Linear(64, classes)

    def forward(self, x):
        ####################  CM  start ############################
        x1 = self.tanh(self.fc1(x.reshape(x.shape[0], -1)))
        x2 = self.fc2(x1)
        w = x2[:, 0].reshape(x.shape[0], -1)
        phi = x2[:, 1].reshape(x.shape[0], -1)

        n = torch.arange(0, 1024).reshape(1, -1).to(x.device).float()
        A = x[:, 0]
        P = x[:, 1]
        F = x[:, 2]

        nA = A
        nP = P - (w * n - phi) / math.pi
        nP = (nP % 2) - 1
        nF = F
        x3 = torch.stack((nA, nP, nF), dim=1)

        x = x3
        ####################  CM  end ############################

        B = x.shape[0]
        x = x.float()
        A = x[:, 0].reshape(B, 1, -1)
        P = x[:, 1].reshape(B, 1, -1)
        F = x[:, 2].reshape(B, 1, -1)

        convA = self.conv2(self.conv1(A))
        convP = self.conv4(self.conv3(P))
        convF = self.conv6(self.conv5(F))

        AP = torch.cat((convA, convP), dim=1)
        convAP = self.conv7(AP)

        APF = torch.cat((convAP, convF), dim=1).reshape(B, 64, -1)

        lstmAPF1, _ = self.lstm1(APF.transpose(2, 1))
        _, (lstmAPF2, __) = self.lstm2(lstmAPF1)

        fc3 = self.fc3(lstmAPF2.squeeze())
        fc4 = self.fc4(fc3)

        return fc4


def test():
    x = torch.randn(1, 3, 1, 1024)
    net = CM_ThreeStream(25)
    flops, params = thop.profile(net, inputs=(x,), verbose=False)
    flops, params = thop.clever_format([flops, params], "%.3f")
    print("flops & params", [flops, params])


if __name__ == "__main__":
    test()
