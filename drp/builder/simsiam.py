import torch
import torch.nn as nn


class SimSiam(nn.Module):
    """
    Build a SimSiam encoder_q.
    """
    def __init__(self, base_encoder, proj_output_dim, proj_hidden_dim, pred_hidden_dim):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder


        # build a 3-layer projector
        dim = base_encoder.fc.weight.shape[1]
        base_encoder.fc = nn.Sequential(nn.Linear(dim, proj_hidden_dim, bias=False),
                                          nn.BatchNorm1d(proj_hidden_dim),
                                          nn.ReLU(inplace=True),  # first layer
                                          nn.Linear(proj_hidden_dim, proj_hidden_dim, bias=False),
                                          nn.BatchNorm1d(proj_hidden_dim),
                                          nn.ReLU(inplace=True),  # second layer
                                          nn.Linear(proj_hidden_dim, proj_output_dim),
                                          nn.BatchNorm1d(proj_output_dim, affine=False))  # output layer
        self.encoder_q = base_encoder
        self.encoder_q.fc[6].bias.requires_grad = False  # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(proj_output_dim, pred_hidden_dim, bias=False),
                                       nn.BatchNorm1d(pred_hidden_dim),
                                       nn.ReLU(inplace=True),  # hidden layer
                                       nn.Linear(pred_hidden_dim, proj_output_dim))  # output layer

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder_q(x1)  # NxC
        z2 = self.encoder_q(x2)  # NxC

        p1 = self.predictor(z1)  # NxC
        p2 = self.predictor(z2)  # NxC

        return p1, p2, z1.detach(), z2.detach()

