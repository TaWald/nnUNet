from typing import Tuple


from dynamic_network_architectures.building_blocks.patch_encode_decode import (
    LayerNormNd,
)
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from timm.layers import RotaryEmbeddingCat
from torch import nn
from dynamic_network_architectures.architectures.primus import Primus
from timm.layers import PatchDropout

from nnunetv2.architectures.deeper_patchembed import (PatchEmbed_deeper)





class Primusv2(Primus):
    def __init__(
            self,
            input_channels: int,
            embed_dim: int,
            patch_embed_size: Tuple[int, ...],
            num_classes: int,
            eva_depth: int = 24,
            eva_numheads: int = 16,
            input_shape: Tuple[int, ...] = None,
            decoder_norm=LayerNormNd,
            decoder_act=nn.GELU,
            num_register_tokens: int = 0,
            use_rot_pos_emb: bool = True,
            use_abs_pos_embed: bool = True,
            mlp_ratio=4 * 2 / 3,
            drop_path_rate=0,  # drops computations (multihead attention, mlp), Implementation of scaling might be useless here because this is not batch normed
            patch_drop_rate: float = 0.0,  # drops input patches, may be used for MAE style pretraining
            proj_drop_rate: float = 0.0,  # drops out things related to the projection. That is in the MLP and at the end of EVA attention
            attn_drop_rate: float = 0.0,  # drops attention, meaning connections between patches may bebroken up at random
            rope_impl=RotaryEmbeddingCat,
            rope_kwargs=None,
            init_values=None,
            scale_attn_inner=False,
            embed_base_features=32,
            embed_depth_per_level=(1,1,1),
            embed_block_type="basic",
            embed_proj_3x3x3=False,
            init_downproj=True,
            embed_block_style="residual",

    ):
        """
        consists of a UNet encoder, a EVA ViT bottleneck and a UNet decoder
        """
        assert input_shape is not None
        assert len(input_shape) == 3, "Currently on ly 3d is supported"
        assert all([j % i == 0 for i, j in zip(patch_embed_size, input_shape)])

        super().__init__(
            input_channels,
            embed_dim,
            patch_embed_size,
            num_classes,
            eva_depth,
            eva_numheads,
            input_shape,
            decoder_norm,
            decoder_act,
            num_register_tokens,
            use_rot_pos_emb,
            use_abs_pos_embed,
            mlp_ratio,
            drop_path_rate,
            patch_drop_rate,
            proj_drop_rate,
            attn_drop_rate,
            rope_impl,
            rope_kwargs,
            init_values,
            scale_attn_inner
        )

        self.down_projection = PatchEmbed_deeper(
            input_channels=input_channels,
            embed_dim=embed_dim,
            base_features=embed_base_features,
            depth_per_level=embed_depth_per_level,  # Defines patch embed size
            block_type=embed_block_type,
            embed_proj_3x3x3=embed_proj_3x3x3,
            embed_block_style=embed_block_style,
        )
        self.keys_to_in_proj = (
            "down_projection.stem",
            "down_projection.stages",
            "down_projection.final_proj",
        )
        if init_downproj:  # Previously missing
            self.down_projection.apply(InitWeights_He(1e-2))
