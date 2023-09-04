from copy import deepcopy
from dataclasses import field
import dataclasses
from typing import Sequence, Type
import warnings
import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np


def find_norms(network: nn.Module):
    """Find the name of the norms that are in the network"""
    all_norms = []
    for name, module in network.named_modules(remove_duplicate=False):
        if isinstance(
            module,
            (
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
                nn.InstanceNorm1d,
                nn.InstanceNorm2d,
                nn.InstanceNorm3d,
                nn.LazyBatchNorm1d,
                nn.LazyBatchNorm2d,
                nn.LazyBatchNorm3d,
                nn.LazyInstanceNorm1d,
                nn.LazyInstanceNorm2d,
                nn.LazyInstanceNorm3d,
            ),
        ):
            all_norms.append(name)
    return all_norms


def replace_norms(network: nn.Module) -> nn.Module:
    """Replace the norms with a simple nn.Identity() call."""
    replacement_names = find_norms(network)
    for rn in replacement_names:
        attr_name = rn.split(".")[-1]  # Get attribute name
        submodule_path = ".".join(rn.split(".")[:-1])  # Get Path to parent module
        sub_module = network.get_submodule(submodule_path)
        setattr(sub_module, attr_name, nn.Identity())
        # setattr(sub_module, attr_name, nn.Identity())
    return network


def set_conv_weights(network: nn.Module):
    """Sets the conv kernel weights to 1 and the bias to 0."""
    for name, module in network.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.LazyConv1d, nn.LazyConv2d, nn.LazyConv3d)):
            module.weight.data = torch.ones_like(module.weight.data)
            module.bias.data = torch.zeros_like(module.bias.data)  # Make sure bias doesn't skrew the propagated input
        elif isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            module.weight.data = torch.ones_like(module.weight.data)  # Make the upconvolution always work.
            module.bias.data = torch.zeros_like(module.bias.data)
        else:
            continue
            # warnings.warn(f"Module {name} is not accounted for in receptive field calculation. Skipping it.")
    return network


def sanity_check_model(network: nn.Module):
    """Assure nothing that globally changes values from 0 exists."""

    remaining_found_norms = find_norms(network)
    assert (
        len(remaining_found_norms) == 0
    ), f"Norm {remaining_found_norms} are still in the model. This should not be the case."


def init_architecture(network: nn.Module) -> nn.Module:
    """
    Removes norms, sets conv weights to 1 and bias to 0, so the single one can propagate without interference.
    """
    network = replace_norms(network)
    sanity_check_model(network)
    return set_conv_weights(network)


def first_nonzero(arr, axis, invalid_val=-1):
    """https://stackoverflow.com/questions/47269390/how-to-find-first-non-zero-value-in-every-column-of-a-numpy-array"""
    arr = arr.cpu().numpy()
    mask = arr != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def last_nonzero(arr, axis, invalid_val=-1):
    """https://stackoverflow.com/questions/47269390/how-to-find-first-non-zero-value-in-every-column-of-a-numpy-array"""
    arr = arr.cpu().numpy()
    mask = arr != 0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)


def _receptive_field_calc_hook(savings_dict: dict, name: str, input_size: Sequence[int]):
    """Calculates the minimum and maximum indices that are non-zero in the output of a module.
    Saves that with the name in the savings dict."""

    def hook(module, input, output):
        # Shape of the output is up to: [batch, channels, z, y, x]

        # ----- Calculate the minimum and maximum indices that are non-zero in the output of a module -----
        # Now collapse all the not interesting dimensions

        savings_dict[name] = {}

        for cnt, dim in enumerate(range(2, len(output.shape))):
            sum_dim = tuple(set(range(len(output.shape))) - {dim})

            summed_along_dim = torch.sum(output, dim=sum_dim)
            min = float(first_nonzero(summed_along_dim, axis=0))
            max = float(last_nonzero(summed_along_dim, axis=0))

            diff = max - min + 1

            rel_diff = diff / output.shape[dim]
            abs_fov = int(input_size[cnt] * rel_diff)
            if cnt == 0:
                savings_dict[name] = {
                    "Receptive Field": [abs_fov],
                    "pixel_diff_at_resolution": [diff],
                    "local_resolution": [output.shape[dim]],
                }
            else:
                savings_dict[name]["Receptive Field"].append(abs_fov)
                savings_dict[name]["pixel_diff_at_resolution"].append(diff)
                savings_dict[name]["local_resolution"].append(output.shape[dim])
        return

    return hook


def register_hooks(
    changed_model: nn.Module, input_size: Sequence[int]
) -> tuple[dict, list[torch.utils.hooks.RemovableHandle]]:
    handles = []
    module_size_dict: dict[str, dict] = {}
    for name, module in changed_model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.LazyConv1d, nn.LazyConv2d, nn.LazyConv3d)):
            handles.append(
                module.register_forward_hook(_receptive_field_calc_hook(module_size_dict, name, input_size))
            )
        # make a forward pass
    return module_size_dict, handles


def create_input(input_size, device) -> torch.Tensor:
    """Create the input to the model that is used for the receptive field calculation."""
    mid_id_0 = input_size[0] // 2
    mid_id_1 = input_size[1] // 2
    mid_id_2 = input_size[2] // 2
    x = torch.zeros(1, 1, *input_size, device=device)
    x[0, 0, mid_id_0, mid_id_1, mid_id_2] = 1
    return x


def pseudo_receptive_field(model: nn.Module, input_size, device="cuda"):
    """
    :parameter
    'input_size': tuple of (Channel, Height, Width)
    :return  OrderedDict of `Layername`->OrderedDict of receptive field stats {'j':,'r':,'start':,'conv_stage':,'output_shape':,}
    'j' for "jump" denotes how many pixels do the receptive fields of spatially neighboring units in the feature tensor
        do not overlap in one direction.
        i.e. shift one unit in this feature map == how many pixels shift in the input image in one direction.
    'r' for "receptive_field" is the spatial range of the receptive field in one direction.
    'start' denotes the center of the receptive field for the first unit (start) in on direction of the feature tensor.
        Convention is to use half a pixel as the center for a range. center for `slice(0,5)` is 2.5.
    """

    # register hook

    init_architecture(model)

    module_size_dict: dict[str, dict]
    handles: list[torch.utils.hooks.RemovableHandle]
    module_size_dict, handles = register_hooks(model, input_size)

    with torch.no_grad():
        x = create_input(input_size, device)
        model(x)

    for h in handles:
        h.remove()

    print(model)

    for name, mets in module_size_dict.items():
        print(name, mets["Receptive Field"])
    """
    print("------------------------------------------------------------------------------")
    line_new = "{:>20}  {:>10} {:>10} {:>10} {:>15} ".format(
        "Layer (type)", "map size", "start", "jump", "receptive_field"
    )
    print(line_new)
    print("==============================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in receptive_field:
        # input_shape, output_shape, trainable, nb_params
        assert "start" in receptive_field[layer], layer
        assert len(receptive_field[layer]["output_shape"]) == 4 or len(receptive_field[layer]["output_shape"]) == 5
        line_new = "{:7} {:12}  {:>10} {:>10} {:>10} {:>15} ".format(
            "",
            layer,
            str(receptive_field[layer]["output_shape"][2:]),
            str(receptive_field[layer]["start"]),
            str(receptive_field[layer]["j"]),
            format(str(receptive_field[layer]["r"])),
        )
        print(line_new)

    print("==============================================================================")
    # add input_shape
    receptive_field["input_size"] = input_size
    return receptive_field
    """
