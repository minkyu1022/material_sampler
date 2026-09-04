from typing import Tuple

from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper
import torch
from torch import Tensor, nn

from pairmixer.data import const
from pairmixer.model.layers.attention import AttentionPairBias
from pairmixer.model.layers.dropout import get_dropout_mask
from pairmixer.model.layers.transition import Transition
from pairmixer.model.layers.outer_product_mean import OuterProductMean
from pairmixer.model.layers.pair_averaging import PairWeightedAveraging
from pairmixer.model.layers.triangular_mult import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)

class PairmixerModule(nn.Module):
    # used in place of Pairformer in the trunk
    def __init__(
        self,
        token_z: int,
        num_blocks: int,
        dropout: float = 0.25,
        activation_checkpointing: bool = False,
        offload_to_cpu: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(num_blocks):
            layer = PairmixerLayer(token_z, dropout)
            if activation_checkpointing:
                layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
            self.layers.append(layer)

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor,
        pair_mask: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:

        for layer in self.layers:
            z = layer(z, pair_mask)

        return s, z

class PairmixerLayer(nn.Module):

    def __init__(self, token_z: int, dropout: float = 0.25) -> None:
        super().__init__()
        self.dropout = dropout
        self.tri_mul_out = TriangleMultiplicationOutgoing(token_z)
        self.tri_mul_in = TriangleMultiplicationIncoming(token_z)
        self.transition_z = Transition(token_z, token_z * 4)

    def forward(self, z: Tensor, pair_mask: Tensor) -> Tensor:

        dropout = get_dropout_mask(self.dropout, z, self.training)
        z = z + dropout * self.tri_mul_out(z, mask=pair_mask)

        dropout = get_dropout_mask(self.dropout, z, self.training)
        z = z + dropout * self.tri_mul_in(z, mask=pair_mask)

        z = z + self.transition_z(z)

        return z


class PairmixerWithSeqAttnModule(nn.Module):
    # used in place of Pairformer in the confidence module
    def __init__(
        self,
        token_s: int,
        token_z: int,
        num_blocks: int,
        num_heads: int = 16,
        dropout: float = 0.25,
        activation_checkpointing: bool = False,
        offload_to_cpu: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(num_blocks):
            layer = PairmixerWithSeqAttnLayer(token_s, token_z, num_heads, dropout)
            if activation_checkpointing:
                layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
            self.layers.append(layer)

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor,
        pair_mask: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:

        for layer in self.layers:
            s, z = layer(s, z, mask, pair_mask)

        return s, z

class PairmixerWithSeqAttnLayer(nn.Module):

    def __init__(
        self,
        token_s: int,
        token_z: int,
        num_heads: int = 16,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.dropout = dropout

        self.tri_mul_out = TriangleMultiplicationOutgoing(token_z)
        self.tri_mul_in = TriangleMultiplicationIncoming(token_z)
        self.transition_z = Transition(token_z, token_z * 4)

        self.attention = AttentionPairBias(token_s, token_z, num_heads)
        self.transition_s = Transition(token_s, token_s * 4)

    def forward(self, s: Tensor, z: Tensor, mask: Tensor, pair_mask: Tensor) -> Tensor:

        dropout = get_dropout_mask(self.dropout, z, self.training)
        z = z + dropout * self.tri_mul_out(z, mask=pair_mask)

        dropout = get_dropout_mask(self.dropout, z, self.training)
        z = z + dropout * self.tri_mul_in(z, mask=pair_mask)

        z = z + self.transition_z(z)

        s = s + self.attention(s, z, mask)
        s = s + self.transition_s(s)

        return s, z


class PairmixerMSAModule(nn.Module):
    # used in place of MSAModule in the trunk

    def __init__(
        self,
        msa_s: int,
        token_z: int,
        s_input_dim: int,
        msa_blocks: int,
        msa_dropout: float,
        z_dropout: float,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        activation_checkpointing: bool = False,
        use_paired_feature: bool = False,
        offload_to_cpu: bool = False,
        subsample_msa: bool = False,
        num_subsampled_msa: int = 1024,
        **kwargs,
    ) -> None:
        super().__init__()

        self.msa_blocks = msa_blocks
        self.msa_dropout = msa_dropout
        self.z_dropout = z_dropout
        self.use_paired_feature = use_paired_feature
        self.subsample_msa = subsample_msa
        self.num_subsampled_msa = num_subsampled_msa

        self.s_proj = nn.Linear(s_input_dim, msa_s, bias=False)
        self.msa_proj = nn.Linear(
            const.num_tokens + 2 + int(use_paired_feature),
            msa_s,
            bias=False,
        )
        self.layers = nn.ModuleList()
        for i in range(msa_blocks):
            layer = PairmixerMSALayer(
                msa_s,
                token_z,
                msa_dropout,
                z_dropout,
                pairwise_head_width,
                pairwise_num_heads,
            )
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
            self.layers.append(layer)

    def forward(
        self,
        z: Tensor,
        emb: Tensor,
        feats: dict[str, Tensor],
        use_kernels: bool = False,
    ) -> Tensor:
        # Set chunk sizes
        if not self.training:
            if z.shape[1] > const.chunk_size_threshold:
                chunk_heads_pwa = True
                chunk_size_transition_z = 64
                chunk_size_transition_msa = 32
                chunk_size_outer_product = 4
                chunk_size_tri_attn = 128
            else:
                chunk_heads_pwa = False
                chunk_size_transition_z = None
                chunk_size_transition_msa = None
                chunk_size_outer_product = None
                chunk_size_tri_attn = 512
        else:
            chunk_heads_pwa = False
            chunk_size_transition_z = None
            chunk_size_transition_msa = None
            chunk_size_outer_product = None
            chunk_size_tri_attn = None

        # Load relevant features
        msa = feats["msa"]
        has_deletion = feats["has_deletion"].unsqueeze(-1)
        deletion_value = feats["deletion_value"].unsqueeze(-1)
        is_paired = feats["msa_paired"].unsqueeze(-1)
        msa_mask = feats["msa_mask"]
        token_mask = feats["token_pad_mask"].float()
        token_mask = token_mask[:, :, None] * token_mask[:, None, :]

        # Compute MSA embeddings
        if self.use_paired_feature:
            m = torch.cat([msa, has_deletion, deletion_value, is_paired], dim=-1)
        else:
            m = torch.cat([msa, has_deletion, deletion_value], dim=-1)

        if self.subsample_msa:
            msa_indices = torch.randperm(m.shape[1])[: self.num_subsampled_msa]
            m = m[:, msa_indices]
            msa_mask = msa_mask[:, msa_indices]

        # Compute input projections
        m = self.msa_proj(m)
        m = m + self.s_proj(emb).unsqueeze(1)

        # Perform MSA blocks
        for i in range(self.msa_blocks):
            z, m = self.layers[i](
                z,
                m,
                token_mask,
                msa_mask,
                chunk_heads_pwa,
                chunk_size_transition_z,
                chunk_size_transition_msa,
                chunk_size_outer_product,
                chunk_size_tri_attn,
                use_kernels=use_kernels,
            )
        return z

class PairmixerMSALayer(nn.Module):
    def __init__(
        self,
        msa_s: int,
        token_z: int,
        msa_dropout: float,
        z_dropout: float,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.msa_dropout = msa_dropout
        self.z_dropout = z_dropout
        self.msa_transition = Transition(dim=msa_s, hidden=msa_s * 4)
        self.pair_weighted_averaging = PairWeightedAveraging(
            c_m=msa_s,
            c_z=token_z,
            c_h=32,
            num_heads=8,
        )

        self.tri_mul_out = TriangleMultiplicationOutgoing(token_z)
        self.tri_mul_in = TriangleMultiplicationIncoming(token_z)
        self.z_transition = Transition(
            dim=token_z,
            hidden=token_z * 4,
        )
        self.outer_product_mean = OuterProductMean(
            c_in=msa_s,
            c_hidden=32,
            c_out=token_z,
        )

    def forward(
        self,
        z: Tensor,
        m: Tensor,
        token_mask: Tensor,
        msa_mask: Tensor,
        chunk_heads_pwa: bool = False,
        chunk_size_transition_z: int = None,
        chunk_size_transition_msa: int = None,
        chunk_size_outer_product: int = None,
        chunk_size_tri_attn: int = None,
        use_kernels: bool = False,
    ) -> tuple[Tensor, Tensor]:
        # Communication to MSA stack
        msa_dropout = get_dropout_mask(self.msa_dropout, m, self.training)
        m = m + msa_dropout * self.pair_weighted_averaging(
            m, z, token_mask, chunk_heads_pwa
        )
        m = m + self.msa_transition(m, chunk_size_transition_msa)

        # Communication to pairwise stack
        z = z + self.outer_product_mean(m, msa_mask, chunk_size_outer_product)

        # Compute pairwise stack
        dropout = get_dropout_mask(self.z_dropout, z, self.training)
        z = z + dropout * self.tri_mul_out(z, mask=token_mask)

        dropout = get_dropout_mask(self.z_dropout, z, self.training)
        z = z + dropout * self.tri_mul_in(z, mask=token_mask)

        z = z + self.z_transition(z, chunk_size_transition_z)

        return z, m
