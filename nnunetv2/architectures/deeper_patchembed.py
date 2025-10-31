from torch import nn
import numpy as np
from typing import Tuple
from einops import rearrange
from einops.layers.torch import Rearrange
import torch
from dynamic_network_architectures.building_blocks.residual import StackedResidualBlocks, BottleneckD, BasicBlockD
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks, ConvDropoutNormReLU

# from nnunetv2.architectures.stacked_swin_blocks import PatchEmbed_Windowed3D, WindowedStage3D


class PatchEmbed_deeper(nn.Module):
    """ResNet-style patch embedding with progressive downsampling"""

    def __init__(
        self,
        input_channels: int = 3,
        embed_dim: int = 864,
        base_features: int = 32,
        depth_per_level: tuple[int, ...] = (3, 4, 6),  # number of residual blocks per level
        block_type: str = "bottleneck",  # "basic" or "bottleneck"
        embed_proj_3x3x3: bool = False,
        embed_block_style: str = "residual",
    ) -> None:
        super().__init__()

        block = BottleneckD if block_type == "bottleneck" else BasicBlockD
        norm_op = nn.InstanceNorm3d
        norm_op_kwargs = {"eps": 1e-5, "affine": True}
        nonlin = nn.LeakyReLU if block_type == "bottleneck" else nn.ReLU
        nonlin_kwargs = {"inplace": True}

        # Stem convolution (initial feature extraction)
        if block_type == "bottleneck":
            bottleneck_channels = base_features // 4
        else:
            bottleneck_channels = None

        if embed_block_style == "residual":
            self.stem = StackedResidualBlocks(
                1,
                nn.Conv3d,
                input_channels,
                base_features,
                [3, 3, 3],
                1,
                True,
                norm_op,
                norm_op_kwargs,
                None,
                None,
                nonlin,
                nonlin_kwargs,
                block=block,
            )
        elif embed_block_style == "conv":
            self.stem = StackedConvBlocks(
                1,
                nn.Conv3d,
                input_channels,
                base_features,
                [3, 3, 3],
                1,
                True,
                norm_op,
                norm_op_kwargs,
                None,
                None,
                nonlin,
                nonlin_kwargs,
            )
        elif embed_block_style == "shifted_window":
            self.stem = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)

        # Calculate total downsampling needed
        levels_needed = len(depth_per_level)

        # Build encoder stages
        self.stages = nn.ModuleList()
        input_channels = base_features

        for i in range(levels_needed):
            # First block in each stage handles downsampling and channel increase
            stride = 2
            output_channels = base_features * (2**i)
            if embed_block_style == "residual":
                if block_type == "bottleneck":
                    bottleneck_channels = output_channels // 4
                else:
                    bottleneck_channels = None
                stage = StackedResidualBlocks(
                    n_blocks=depth_per_level[i],
                    conv_op=nn.Conv3d,
                    input_channels=input_channels,
                    output_channels=output_channels,
                    kernel_size=3,
                    initial_stride=stride,
                    conv_bias=False,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    dropout_op=None,
                    dropout_op_kwargs=None,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                    block=block,
                    bottleneck_channels=bottleneck_channels,
                )
            elif embed_block_style == "conv":
                stage = StackedConvBlocks(
                    num_convs=depth_per_level[i],
                    conv_op=nn.Conv3d,
                    input_channels=input_channels,
                    output_channels=output_channels,
                    kernel_size=3,
                    initial_stride=stride,
                    conv_bias=False,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    dropout_op=None,
                    dropout_op_kwargs=None,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                )
            self.stages.append(stage)
            input_channels = output_channels

        # Global average pooling or final conv to get to embed_dim
        final_proj_kernel = [3, 3, 3] if embed_proj_3x3x3 else [1, 1, 1]
        final_pad = [1, 1, 1] if embed_proj_3x3x3 else [0, 0, 0]
        self.final_proj = nn.Conv3d(
            input_channels, embed_dim, kernel_size=final_proj_kernel, stride=[1, 1, 1], padding=final_pad
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.stem(x)

        # Progressive encoding through residual stages
        for stage in self.stages:
            x = stage(x)

        # Final projection to embedding tokens
        x = self.final_proj(x)

        return x


class PatchEmbed_convstyle(nn.Module):
    """ResNet-style patch embedding with progressive downsampling"""

    def __init__(
        self,
        input_size: Tuple[int, int, int],
        input_channels: int = 3,
        embed_dim: int = 864,
        patch_embed_size: Tuple[int, int, int] = (8, 8, 8),
        base_features: int = 64,
    ) -> None:
        super().__init__()

        dw, hw, ww = patch_embed_size
        ds, hs, ws = [in_size // windows_size for in_size, windows_size in zip(input_size, patch_embed_size)]
        self.stages = nn.ModuleList()
        self.in_rearrange = Rearrange("b c (ds dw) (hs hw) (ws ww) -> (b ds hs ws) c dw hw ww", dw=dw, hw=hw, ww=ww)
        self.out_rearrange = Rearrange("(b ds hs ws) c 1 1 1 -> b c ds hs ws", ds=ds, hs=hs, ws=ws)

        in_channels = input_channels
        out_channels = base_features
        cur_patch_embed_size = np.array(patch_embed_size)
        while not any([p <= 2 for p in cur_patch_embed_size]):
            cur_patch_embed_size = cur_patch_embed_size - 2
            self.stages.append(
                ConvDropoutNormReLU(
                    nn.Conv3d,
                    input_channels=in_channels,
                    output_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    norm_op=nn.InstanceNorm3d,
                    norm_op_kwargs={"eps": 1e-5, "affine": True},
                    nonlin=nn.LeakyReLU,
                    nonlin_kwargs={"inplace": True},
                )
            )
            self.stages[-1].all_modules[0].padding = (0, 0, 0)  # manually set padding to 0
            in_channels = out_channels
            out_channels = out_channels * 2
        self.projection = nn.Conv3d(
            in_channels, embed_dim, kernel_size=cur_patch_embed_size.tolist(), stride=(1, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Progressive encoding through residual stages
        x = self.in_rearrange(x)
        for stage in self.stages:
            x = stage(x)

        # Final projection to embedding tokens
        x = self.projection(x)
        x = self.out_rearrange(x)
        return x


class PatchEmbed_projections(nn.Module):
    """ResNet-style patch embedding with progressive downsampling"""

    def __init__(
        self,
        input_channels: int = 3,
        embed_dim: int = 864,
        stages: int = 3,  # 2**stages = number of projections
        base_features: int = 32,
        embed_proj_3x3x3: bool = False,
    ) -> None:
        super().__init__()

        self.stages = nn.ModuleList()
        in_channels = input_channels
        out_channels = base_features
        for i in range(stages):
            self.stages.add_module(
                f"{i}",
                ConvDropoutNormReLU(
                    nn.Conv3d,
                    input_channels=in_channels,
                    output_channels=out_channels,
                    kernel_size=2,
                    stride=2,
                    norm_op=nn.InstanceNorm3d,
                    norm_op_kwargs={"eps": 1e-5, "affine": True},
                    nonlin=nn.LeakyReLU,
                    nonlin_kwargs={"inplace": True},
                ),
            )
            in_channels = out_channels
            out_channels = out_channels * 2

        # Global average pooling or final conv to get to embed_dim
        final_proj_kernel = [3, 3, 3] if embed_proj_3x3x3 else [1, 1, 1]
        final_pad = [1, 1, 1] if embed_proj_3x3x3 else [0, 0, 0]
        self.final_proj = nn.Conv3d(
            in_channels, embed_dim, kernel_size=final_proj_kernel, stride=[1, 1, 1], padding=final_pad
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Progressive encoding through residual stages
        for stage in self.stages:
            x = stage(x)

        # Final projection to embedding tokens
        x = self.final_proj(x)

        return x


class PatchEmbed_projection_444_222(nn.Module):
    """ResNet-style patch embedding with progressive downsampling"""

    def __init__(
        self,
        input_channels: int = 3,
        embed_dim: int = 864,
        first_444: bool = True,  # 2**stages = number of projections
    ) -> None:
        super().__init__()

        self.stages = nn.ModuleList()
        in_channels = input_channels
        self.first_444 = first_444

        if first_444:
            conv = nn.Conv3d(in_channels, 64, kernel_size=4, stride=4, padding=0, bias=False)
            norm = nn.InstanceNorm3d(64, eps=1e-5, affine=True)
            nonlin = nn.LeakyReLU(inplace=True)
            self.stages.add_module("0", nn.Sequential(conv, norm, nonlin))
            in_channels = 64
            conv = nn.Conv3d(in_channels, embed_dim, kernel_size=2, stride=2, padding=0, bias=False)
            norm = nn.InstanceNorm3d(embed_dim, eps=1e-5, affine=True)
            nonlin = nn.LeakyReLU(inplace=True)
            self.stages.add_module("1", nn.Sequential(conv, norm, nonlin))
        else:
            conv = nn.Conv3d(in_channels, 64, kernel_size=2, stride=2, padding=0, bias=False)
            norm = nn.InstanceNorm3d(64, eps=1e-5, affine=True)
            nonlin = nn.LeakyReLU(inplace=True)
            stage_222 = nn.Sequential(conv, norm, nonlin)
            self.stages.add_module("0", stage_222)
            in_channels = 64
            conv = nn.Conv3d(in_channels, embed_dim, kernel_size=4, stride=4, padding=0, bias=False)
            norm = nn.InstanceNorm3d(embed_dim, eps=1e-5, affine=True)
            nonlin = nn.LeakyReLU(inplace=True)
            self.stages.add_module("1", nn.Sequential(conv, norm, nonlin))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Progressive encoding through residual stages
        for stage in self.stages:
            x = stage(x)

        return x


