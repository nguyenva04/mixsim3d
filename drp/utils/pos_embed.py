import numpy as np
import torch


def get_sinusoid_encoding_table(n_position, embed_dim, cls_token=False):
    ''' Sinusoid position encoding table '''
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / embed_dim) for hid_j in range(embed_dim)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    # pos_embed = torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)
    if cls_token:
        cls_embed = np.zeros((1, embed_dim), dtype=np.float32)  # Positional embedding for class token
        sinusoid_table = np.concatenate([cls_embed, sinusoid_table], axis=0)

    return sinusoid_table


def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int or tuple (depth, height, width) for the grid dimensions
    return:
    pos_embed: [grid_size[0]*grid_size[1]*grid_size[2], embed_dim] or
               [1+grid_size[0]*grid_size[1]*grid_size[2], embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_d, grid_h, grid_w = grid_size, grid_size, grid_size
    else:
        grid_d, grid_h, grid_w = grid_size

    grid_d = np.arange(grid_d, dtype=np.float32)
    grid_h = np.arange(grid_h, dtype=np.float32)
    grid_w = np.arange(grid_w, dtype=np.float32)

    grid = np.meshgrid(grid_w, grid_h, grid_d)  # w, h, d
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, grid_size, grid_size, grid_size])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):

    assert embed_dim % 3 == 0, "embed_dim should be divisible by 3 for 3D positional encoding"

    # Divide the embedding dimensions by 3 for x, y, and z components
    emb_dim_per_axis = embed_dim // 3
    emb_d = get_1d_sincos_pos_embed_from_grid(emb_dim_per_axis, grid[2])  # (D*H*W, D/3)
    emb_h = get_1d_sincos_pos_embed_from_grid(emb_dim_per_axis, grid[1])  # (D*H*W, D/3)
    emb_w = get_1d_sincos_pos_embed_from_grid(emb_dim_per_axis, grid[0])  # (D*H*W, D/3)

    # Concatenate along the embedding dimension
    emb = np.concatenate([emb_d, emb_h, emb_w], axis=1)  # (D*H*W, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb




