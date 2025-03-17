from torch import nn
from drp.train.basemethod import BaseMethod
from functools import partial
from drp.utils.config import Config
from torch.utils.data import DistributedSampler
from drp.handlers.checkpoint import CheckPoint
from drp.builder.VIT3d import MaskedAutoencoderViT
from drp.builder.transformer import SeqPool


class FineTunedMAE(nn.Module):
    def __init__(self, mae_model, num_classes=1, use_seq_pool=False):
        super().__init__()
        self.use_seq_pool = use_seq_pool
        self.encoder = mae_model

        self.layer_norm = nn.LayerNorm(mae_model.embed_dim)

        if self.use_seq_pool:  # Check if SeqPool should be added
            self.seq_pool = SeqPool(emb_dim=mae_model.embed_dim)

        self.classifier = nn.Linear(mae_model.embed_dim, num_classes)

    def forward(self, x):
        # Forward pass through the encoder
        latent, _, _ = self.encoder.forward_encoder(x, mask_ratio=0.0)  # mask_ratio=0.0 means no masking
        latent = self.layer_norm(latent)
        if self.use_seq_pool:
            x = latent[:, 1:, :]  # remove class token to perform Sequential Pooling
            x = self.seq_pool(x)
        else:
            x = latent[:, 0, :]  # Extract the class token
        output = self.classifier(x)
        return output


class FineTuneMAE(BaseMethod):
    def __init__(self,
                 config: Config,
                 rank: int = 0,
                 local_rank: int = 0,
                 sampler_train: DistributedSampler = None,
                 ) -> None:
        super().__init__(config, rank, local_rank, sampler_train)

        model = MaskedAutoencoderViT(img_size=self.config["dim"],
                                     in_chans=1,
                                     embed_dim=252,
                                     depth=4,
                                     num_heads=6,
                                     decoder_embed_dim=126,
                                     decoder_depth=2,
                                     decoder_num_heads=6,
                                     mlp_ratio=4.,
                                     norm_layer=partial(nn.LayerNorm, eps=1e-6))

        checkpoint_ = CheckPoint(run_id=self.config["ckp_runid"], best_ckp=False)
        checkpoint_.init_model(model)

        self.backbone = FineTunedMAE(
            mae_model=model,
            num_classes=self.config['num_classes'],
            use_seq_pool=self.config['use_seq_pool']
            )

        self.backbone = self._distribute_model(self.backbone)

        if not self.config["finetune"]:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True

