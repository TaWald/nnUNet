from typing import Type
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder, PlainConvEncoder
from dynamic_network_architectures.building_blocks.simple_conv_blocks import ConvDropoutNormReLU
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import (
    maybe_convert_scalar_to_list,
    get_matching_pool_op,
)
import torch
from torch import nn

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd


class MixedStackedConvBlocks(nn.Module):
    def __init__(
        self,
        num_convs: int,
        conv_op: Type[_ConvNd],
        input_channels: int,
        output_channels: int | list[int] | tuple[int, ...],
        kernel_size: int | list[int] | tuple[int, ...],
        initial_stride: int | list[int] | tuple[int, ...],
        conv_bias: bool = False,
        norm_op: None | Type[nn.Module] = None,
        norm_op_kwargs: dict = None,
        dropout_op: None | Type[_DropoutNd] = None,
        dropout_op_kwargs: dict = None,
        nonlin: None | Type[torch.nn.Module] = None,
        nonlin_kwargs: dict = None,
        nonlin_first: bool = False,
    ):
        """

        :param conv_op:
        :param num_convs:
        :param input_channels:
        :param output_channels: can be int or a list/tuple of int. If list/tuple are provided, each entry is for
        one conv. The length of the list/tuple must then naturally be num_convs
        :param kernel_size:
        :param initial_stride:
        :param conv_bias:
        :param norm_op:
        :param norm_op_kwargs:
        :param dropout_op:
        :param dropout_op_kwargs:
        :param nonlin:
        :param nonlin_kwargs:
        """
        super().__init__()
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * num_convs

        self.convs = nn.Sequential(
            ConvDropoutNormReLU(
                conv_op,
                input_channels,
                output_channels[0],
                kernel_size,
                initial_stride,
                conv_bias,
                norm_op,
                norm_op_kwargs,
                dropout_op,
                dropout_op_kwargs,
                nonlin,
                nonlin_kwargs,
                nonlin_first,
            ),
            *[
                ConvDropoutNormReLU(
                    conv_op,
                    output_channels[i - 1],
                    output_channels[i],
                    kernel_size if (i % 2) == 0 else 1,
                    1,
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    dropout_op,
                    dropout_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                    nonlin_first,
                )
                for i in range(1, num_convs)
            ],
        )

        self.output_channels = output_channels[-1]
        self.initial_stride = maybe_convert_scalar_to_list(conv_op, initial_stride)

    def forward(self, x):
        return self.convs(x)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.initial_stride), (
            "just give the image size without color/feature channels or "
            "batch channel. Do not give input_size=(b, c, x, y(, z)). "
            "Give input_size=(x, y(, z))!"
        )
        output = self.convs[0].compute_conv_feature_map_size(input_size)
        size_after_stride = [i // j for i, j in zip(input_size, self.initial_stride)]
        for b in self.convs[1:]:
            output += b.compute_conv_feature_map_size(size_after_stride)
        return output


class FlatStackedConvBlocks(nn.Module):
    def __init__(
        self,
        num_convs: int,
        conv_op: Type[_ConvNd],
        input_channels: int,
        output_channels: int | list[int] | tuple[int, ...],
        kernel_size: int | list[int] | tuple[int, ...],
        initial_stride: int | list[int] | tuple[int, ...],
        conv_bias: bool = False,
        norm_op: None | Type[nn.Module] = None,
        norm_op_kwargs: dict = None,
        dropout_op: None | Type[_DropoutNd] = None,
        dropout_op_kwargs: dict = None,
        nonlin: None | Type[torch.nn.Module] = None,
        nonlin_kwargs: dict = None,
        nonlin_first: bool = False,
    ):
        """

        :param conv_op:
        :param num_convs:
        :param input_channels:
        :param output_channels: can be int or a list/tuple of int. If list/tuple are provided, each entry is for
        one conv. The length of the list/tuple must then naturally be num_convs
        :param kernel_size:
        :param initial_stride:
        :param conv_bias:
        :param norm_op:
        :param norm_op_kwargs:
        :param dropout_op:
        :param dropout_op_kwargs:
        :param nonlin:
        :param nonlin_kwargs:
        """
        super().__init__()
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * num_convs

        self.convs = nn.Sequential(
            ConvDropoutNormReLU(
                conv_op,
                input_channels,
                output_channels[0],
                1,
                initial_stride,
                conv_bias,
                norm_op,
                norm_op_kwargs,
                dropout_op,
                dropout_op_kwargs,
                nonlin,
                nonlin_kwargs,
                nonlin_first,
            ),
            *[
                ConvDropoutNormReLU(
                    conv_op,
                    output_channels[i - 1],
                    output_channels[i],
                    1,
                    1,
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    dropout_op,
                    dropout_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                    nonlin_first,
                )
                for i in range(1, num_convs)
            ],
        )

        self.output_channels = output_channels[-1]
        self.initial_stride = maybe_convert_scalar_to_list(conv_op, initial_stride)

    def forward(self, x):
        return self.convs(x)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.initial_stride), (
            "just give the image size without color/feature channels or "
            "batch channel. Do not give input_size=(b, c, x, y(, z)). "
            "Give input_size=(x, y(, z))!"
        )
        output = self.convs[0].compute_conv_feature_map_size(input_size)
        size_after_stride = [i // j for i, j in zip(input_size, self.initial_stride)]
        for b in self.convs[1:]:
            output += b.compute_conv_feature_map_size(size_after_stride)
        return output


block_types = {"BASIC": StackedConvBlocks, "MIXED": MixedStackedConvBlocks, "FLAT": FlatStackedConvBlocks}


class DynPlainConvEncoder(PlainConvEncoder):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: int | list[int] | tuple[int, ...],
        conv_op: Type[_ConvNd],
        kernel_sizes: int | list[int] | tuple[int, ...],
        strides: int | list[int] | tuple[int, ...],
        n_conv_per_stage: int | list[int] | tuple[int, ...],
        conv_bias: bool = False,
        norm_op: None | Type[nn.Module] = None,
        norm_op_kwargs: dict = None,
        dropout_op: None | Type[_DropoutNd] = None,
        dropout_op_kwargs: dict = None,
        nonlin: None | Type[torch.nn.Module] = None,
        nonlin_kwargs: dict = None,
        return_skips: bool = False,
        nonlin_first: bool = False,
        pool: str = "conv",
        blocks: None | list[str] = None,
    ):
        nn.Module.__init__(self)
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert (
            len(kernel_sizes) == n_stages
        ), "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert (
            len(n_conv_per_stage) == n_stages
        ), "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert (
            len(features_per_stage) == n_stages
        ), "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, (
            "strides must have as many entries as we have resolution stages (n_stages). "
            "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"
        )

        if blocks is None:
            blocks = ["BASIC" for _ in features_per_stage]

        stages = []
        for s, block in zip(range(n_stages), blocks):
            stage_modules = []
            if pool == "max" or pool == "avg":
                if (
                    (isinstance(strides[s], int) and strides[s] != 1)
                    or isinstance(strides[s], (tuple, list))
                    and any([i != 1 for i in strides[s]])
                ):
                    stage_modules.append(
                        get_matching_pool_op(conv_op, pool_type=pool)(kernel_size=strides[s], stride=strides[s])
                    )
                conv_stride = 1
            elif pool == "conv":
                conv_stride = strides[s]
            else:
                raise RuntimeError()
            stage_modules.append(
                block_types[block](
                    n_conv_per_stage[s],
                    conv_op,
                    input_channels,
                    features_per_stage[s],
                    kernel_sizes[s],
                    conv_stride,
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    dropout_op,
                    dropout_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                    nonlin_first,
                )
            )
            stages.append(nn.Sequential(*stage_modules))
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes


class DynSkipPlainDynConvUNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: int | list[int] | tuple[int, ...],
        conv_op: Type[_ConvNd],
        kernel_sizes: int | list[int] | tuple[int, ...],
        strides: int | list[int] | tuple[int, ...],
        n_conv_per_stage: int | list[int] | tuple[int, ...],
        num_classes: int,
        n_conv_per_stage_decoder: int | tuple[int, ...] | list[int],
        blocks: list[str],
        conv_bias: bool = False,
        norm_op: None | Type[nn.Module] = None,
        norm_op_kwargs: dict = None,
        dropout_op: None | Type[_DropoutNd] = None,
        dropout_op_kwargs: dict = None,
        nonlin: None | Type[torch.nn.Module] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        nonlin_first: bool = False,
        skips: None | list[bool] = None,
    ):
        super().__init__()
        if skips is None:
            self.skip = [True for _ in range(n_stages)]
        else:
            self.skip = skips

        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, (
            "n_conv_per_stage must have as many entries as we have "
            f"resolution stages. here: {n_stages}. "
            f"n_conv_per_stage: {n_conv_per_stage}"
        )
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), (
            "n_conv_per_stage_decoder must have one less entries "
            f"as we have resolution stages. here: {n_stages} "
            f"stages, so it should have {n_stages - 1} entries. "
            f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        )
        self.encoder = DynPlainConvEncoder(
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_conv_per_stage,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            return_skips=True,
            nonlin_first=nonlin_first,
            blocks=blocks,
        )
        self.decoder = UNetDecoder(
            self.encoder,
            num_classes,
            n_conv_per_stage_decoder,
            False,
            nonlin_first=nonlin_first,
        )

    def forward(self, x):
        skips = self.encoder(x)
        for cnt, skip in enumerate(self.skip):
            if skip:
                continue
            skips[cnt] = torch.zeros_like(skips[cnt])
        return self.decoder(skips)
