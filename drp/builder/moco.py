import torch
import torch.nn as nn
from torch import Tensor
from drp.builder.vision_transformer import Block
from drp.builder.VIT3d import PatchEmbed, ConvTokenizer
from drp.utils.pos_embed import get_sinusoid_encoding_table


class QueueModule(nn.Module):
    def __init__(self, dim, K):
        """
        Initialize the queue module.

        Args:
            dim (int): Dimension of the embeddings.
            K (int): Number of features to store in the queue.
        """
        super(QueueModule, self).__init__()
        self.K = K
        # Create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        # Pointer to keep track of the position in the queue
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys: Tensor) -> None:
        """
        Dequeues old keys and enqueues new keys into the queue.

        Args:
            keys (Tensor): A batch of keys to be enqueued. Shape should be (batch_size, features).
        """

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, base_encoder_momentum, m=0.999, T=0.07, mlp=False, distributed=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        # self.K = K
        self.m = m
        self.T = T
        self.distributed = distributed

        # create the encoders
        self.encoder_q = base_encoder
        self.encoder_k = base_encoder_momentum
        fc = self.encoder_q.fc
        fc_momentum = self.encoder_k.fc

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), fc_momentum
            )
        for param_q, param_k in zip(
                self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
            param_q.requires_grad = True

        # create the queue
        # self.register_buffer("queue", torch.randn(dim, K))
        # self.queue = nn.functional.normalize(self.queue, dim=0)
        #
        # self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)
        if self.distributed:
            # compute key features
            with torch.no_grad():  # no gradient to keys
                # self.momentum_update_key_encoder()  # update the key encoder

                # shuffle for making use of BN
                # im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

                k = self.encoder_k(im_k)  # keys: NxC
                k = nn.functional.normalize(k, dim=1)

                # undo shuffle
                # k = self._batch_unshuffle_ddp(k, idx_unshuffle)
        else:
            # self.momentum_update_key_encoder()  # update the key encoder
            k = self.encoder_k(im_k)  # keys: NxC
        return q, k

    def update_queue(self, queue_state_dict):
        self.queue = queue_state_dict["queue"]
        self.queue_ptr = queue_state_dict["queue_ptr"]

    def get_queue(self):
        return {"queue": self.queue, "queue_ptr": self.queue_ptr}


class BaseEncoderMoCoV3(nn.Module):
    def __init__(self, embed_dim, num_heads: int, mlp_ratio=4., out_dim=128, depth=4, patch_embed=True, img_size=100):
        super(BaseEncoderMoCoV3, self).__init__()

        self.embed_dim = embed_dim
        self.img_size = img_size

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
                channels=1, emb_dim=self.embed_dim,
                conv_kernel=conv_kernel, conv_stride=conv_stride, conv_pad=conv_pad,
                pool_kernel=pool_kernel, pool_stride=pool_stride, pool_pad=pool_pad,
                activation=nn.ReLU
            )
        with torch.no_grad():
            x = torch.randn([1, 1, img_size, img_size, img_size])
            out = self.PatchEmbed(x)
            a, num_patches, embed_dim = out.shape
            print(a, num_patches, embed_dim)
        self.number_patches = num_patches

        self.encoder = nn.ModuleList([
            Block(dim=embed_dim, heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True)
            for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = _build_mlp(3, embed_dim, mlp_ratio, out_dim)
        self.pos_embed = nn.Parameter(
            torch.zeros(
                [1, num_patches + 1, self.embed_dim]
            ), requires_grad=False)  # from torchvision, which takes this from BERT

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_sinusoid_encoding_table(n_position=self.number_patches,
                                                embed_dim=self.pos_embed.shape[-1],
                                                cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.PatchEmbed(x)
        print(x.shape)
        print(self.pos_embed[:, 1:, :].shape)
        x = x + self.pos_embed[:, 1:, :]
        print(x.shape)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.encoder:
            x = blk(x)
        x = self.norm(x)
        x = self.head(x)
        print("x.shape  jgdfo", x.shape)
        return x


def _build_mlp(num_layers, input_dim, mlp_ratio, output_dim, last_bn=True):
    mlp = []
    mlp_dim = int(mlp_ratio)*input_dim
    for l in range(num_layers):
        dim1 = input_dim if l == 0 else mlp_dim
        dim2 = output_dim if l == num_layers - 1 else mlp_dim

        mlp.append(nn.Linear(dim1, dim2, bias=False))

        if l < num_layers - 1:
            mlp.append(nn.BatchNorm1d(dim2))
            mlp.append(nn.ReLU(inplace=True))
        elif last_bn:
            mlp.append(nn.BatchNorm1d(dim2, affine=False))

    return nn.Sequential(*mlp)


class MoCoV3(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, embed_dim, num_heads=6, mlp_ratio=4., out_dim=128, depth=6, T=1.0, m=0.999):
        """
        embed_dim: feature dimension (default: 256)
        num_heads: hidden dimension in MLPs (default: 4096)
        mlp_ratio :
        out_dim :
        depth :
        T: softmax temperature (default: 1.0)
        """
        super(MoCoV3, self).__init__()

        self.T = T
        self.m = m

        # build encoders
        self.base_encoder = BaseEncoderMoCoV3(embed_dim, num_heads, mlp_ratio, out_dim, depth)
        self.momentum_encoder = BaseEncoderMoCoV3(embed_dim, num_heads, mlp_ratio, out_dim, depth)

        self.predictor = _build_mlp(3, embed_dim, mlp_ratio, out_dim)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _update_momentum_encoder(self):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * self.m + param_b.data * (1. - self.m)

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """

        # compute features
        q1 = self.predictor(self.base_encoder(x1))
        q2 = self.predictor(self.base_encoder(x2))

        with torch.no_grad():  # no gradient
            # self._update_momentum_encoder(m)  # update the momentum encoder

            # compute momentum features as targets
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)

        return q1, q2, k1, k2


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)

    return output

