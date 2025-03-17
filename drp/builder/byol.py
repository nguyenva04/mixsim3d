import torch
import torch.nn as nn


class BYOL(nn.Module):
    def __init__(self, base_encoder, base_encoder_momentum, m=0.999, proj_hidden_dim=2048, pred_hidden_dim=2048):
        """
        m: momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(BYOL, self).__init__()

        self.m = m

        # create the encoders
        self.encoder_q = base_encoder
        self.encoder_k = base_encoder_momentum

        dim_mlp = self.encoder_q.fc.weight.shape[1]
        out_dim = self.encoder_q.fc.weight.shape[0]

        self.encoder_q.fc = nn.Sequential(
            nn.Linear(dim_mlp, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim), nn.ReLU(),
            nn.Linear(proj_hidden_dim, out_dim)
        )
        self.encoder_k.fc = nn.Sequential(
            nn.Linear(dim_mlp, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim), nn.ReLU(),
            nn.Linear(proj_hidden_dim, out_dim)
        )

        self.prediction = nn.Sequential(
            nn.Linear(out_dim, pred_hidden_dim),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, out_dim),
        )
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
            param_q.requires_grad = True

    @torch.no_grad()
    def momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features

        q1 = self.prediction(self.encoder_q(im_q))
        q2 = self.prediction(self.encoder_q(im_k))

        with torch.no_grad():  # no gradient to keys
            # self._momentum_update_key_encoder()  # update the key encoder
            k1 = self.encoder_k(im_q)  # keys: NxC
            k2 = self.encoder_k(im_k)  # keys: NxC

        return q1, q2, k1, k2


