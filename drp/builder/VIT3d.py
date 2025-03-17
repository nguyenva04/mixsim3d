import torch
import torch.nn as nn
from drp.builder.vision_transformer import Block
from typing import Callable
from drp.utils.pos_embed import get_sinusoid_encoding_table
from drp.builder.transformer import SeqPool


# Define the to_3tuple function for PatchEmbed
def _ntuple(n):
    def parse(x):
        if hasattr(x, '__iter__') and not isinstance(x, (str, bytes)):
            return tuple(x)
        return tuple([x] * n)
    return parse


to_3tuple = _ntuple(3)


# Define the PatchEmbed class
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)
        num_patches = (img_size[2] // patch_size[2]) * (img_size[1] // patch_size[1]) * (img_size[0] // img_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.conv = nn.Conv3d(
            in_channels=in_chans, out_channels=embed_dim,
            kernel_size=(patch_size[0], patch_size[1], patch_size[2]),
            stride=(patch_size[0], patch_size[1], patch_size[2])
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        assert T == self.img_size[0] and H == self.img_size[1] and W == self.img_size[2], \
            f"Input image size ({T}*{H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ConvTokenizer(nn.Module):
    def __init__(
            self,
            channels: int = 1, emb_dim: int = 256,
            conv_kernel: int = 2, conv_stride: int = 2, conv_pad: int = 0,
            pool_kernel: int = 2, pool_stride: int = 2, pool_pad: int = 0,
            activation: Callable = nn.ReLU
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels=channels, out_channels=emb_dim,
            kernel_size=conv_kernel, stride=conv_stride,
            padding=conv_pad
        )
        self.act = activation(inplace=True)
        self.max_pool = nn.MaxPool3d(
            kernel_size=pool_kernel, stride=pool_stride,
            padding=pool_pad
        )

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.act(x)
        x = self.max_pool(x)

        return x.flatten(2).transpose(1, 2)


class EncoderViT(nn.Module):
    """ VisionTransformer backbone for 3D rock image to estimate permeability
    """

    def __init__(self, img_size=100,
                 in_chans=1,
                 embed_dim=256,
                 depth=24,
                 num_heads=8,
                 num_classes=1,
                 use_seq_pool=True,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 norm_pix_loss=False,
                 patch_embed=False):
        super().__init__()

        # --------------------------------------------------------------------------

        self.embed_dim = embed_dim
        self.img_size = img_size
        self.use_seq_pool= use_seq_pool

        if patch_embed:
            self.PatchEmbed = PatchEmbed(
                img_size=img_size,
                patch_size=10,
                in_chans=1,
                embed_dim=embed_dim
                )
        else:
            conv_kernel = 5
            conv_stride = 5
            conv_pad = 0
            pool_kernel = 2
            pool_stride = 2
            pool_pad = 0

            self.PatchEmbed = ConvTokenizer(
                channels=in_chans, emb_dim=self.embed_dim,
                conv_kernel=conv_kernel, conv_stride=conv_stride, conv_pad=conv_pad,
                pool_kernel=pool_kernel, pool_stride=pool_stride, pool_pad=pool_pad,
                activation=nn.ReLU
            )
        with torch.no_grad():
            x = torch.randn([1, in_chans, img_size, img_size, img_size])
            out = self.PatchEmbed(x)
            a, num_patches, embed_dim = out.shape

        print("a, num_patches, embed_dim", a, num_patches, embed_dim)

        self.number_patches = num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(
                [1, num_patches + 1, self.embed_dim]
            ), requires_grad=False)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True)
            for i in range(depth)])

        if self.use_seq_pool:
            self.seq_pool = SeqPool(emb_dim=embed_dim)

        self.classifier = nn.Linear(embed_dim, num_classes)

        self.norm = norm_layer(embed_dim)
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_sinusoid_encoding_table(n_position=self.number_patches,
                                                embed_dim=self.pos_embed.shape[-1],
                                                cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.PatchEmbed.conv.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):

        # embed patches
        x = self.PatchEmbed(x)  # Input shape after tokenization: [B, embed_dim, D, H, W]

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]

        cls_tokens = cls_token.expand(x.shape[0], -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        if self.use_seq_pool:
            x = x[:, 1:, :]  # remove class token to perform Sequential Pooling
            x = self.seq_pool(x)
        else:
            x = x[:, 0, :]  # Extract the class token
        output = self.classifier(x)

        return output
