import torch.nn as nn


class SimclrEncoder(nn.Module):
    def __init__(self, model):
        super(SimclrEncoder, self).__init__()
        self.feature_dim = 128
        self.encoder_q = model
        self.encoder_q.fc = nn.Sequential(
            nn.Linear(512, 2048, bias=False),
            nn.ReLU(),
            nn.Linear(2048, self.feature_dim, bias=False)
        )

    def forward(self, x):
        x = self.encoder_q(x)
        return x

