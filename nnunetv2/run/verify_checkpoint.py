import argparse
import torch
from typing import Tuple, Union, List
from dynamic_network_architectures.architectures.abstract_arch import AbstractDynamicNetworkArchitectures
from nnunetv2.utilities.get_network_via_name import get_network_from_name
from torch import nn
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans


def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        input_patch_size: tuple[int, int, int],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
) -> AbstractDynamicNetworkArchitectures:
    """
    This is where you build the architecture according to the plans. There is no obligation to use
    get_network_from_plans, this is just a utility we use for the nnU-Net default architectures. You can do what
    you want. Even ignore the plans and just return something static (as long as it can process the requested
    patch size)
    but don't bug us with your bugs arising from fiddling with this :-P
    This is the function that is called in inference as well! This is needed so that all network architecture
    variants can be loaded at inference time (inference will use the same nnUNetTrainer that was used for
    training, so if you change the network architecture during training by deriving a new trainer class then
    inference will know about it).

    If you need to know how many segmentation outputs your custom architecture needs to have, use the following snippet:
    > label_manager = plans_manager.get_label_manager(dataset_json)
    > label_manager.num_segmentation_heads
    (why so complicated? -> We can have either classical training (classes) or regions. If we have regions,
    the number of outputs is != the number of classes. Also there is the ignore label for which no output
    should be generated. label_manager takes care of all that for you.)

    """
    if architecture_class_name in ["PlainConvUNet", "ResidualEncoderUNet"] or architecture_class_name.endswith(
            ("PlainConvUNet", "ResidualEncoderUNet")
    ):
        network = get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )
    elif architecture_class_name in ["PrimusS", "PrimusM", "PrimusL", "PrimusB", "ResEncL"]:
        network = get_network_from_name(
            architecture_class_name,
            input_channels=num_input_channels,
            output_channels=num_output_channels,
            input_patchsize=input_patch_size,
            allow_init=True,
            deep_supervision=False,
        )
    else:
        raise NotImplementedError("Unknown architecture class name: {}".format(architecture_class_name))
    return network


def load_pretrained_weights(
        network: AbstractDynamicNetworkArchitectures,
        pretrained_weights_path: str,
        downstream_input_channels: int,
        downstream_input_patchsize: int,
) -> tuple[nn.Module, bool]:
    """
    Load pretrained weights into the network.
    Per default we only load the encoder and the stem weights. The stem weights are adapted to the number of input channels through repeats.
    The decoder is initialized from scratch.

    :param network: The neural network to load weights into.
    :param pretrained_weights_path: Path to the pretrained weights file.
    :param pt_input_channels: Number of input channels used in the pretrained model.
    :param downstream_input_channels: Number of input channels used during adaptation (currently).
    :param pt_input_patchsize: Patch size used in the pretrained model.
    :param downstream_input_patchsize: Patch size used during adaptation (currently).
    :param pt_key_to_encoder: Key to the encoder in the pretrained model.
    :param pt_key_to_stem: Key to the stem in the pretrained model.

    :return: The network with loaded weights.
    """

    # --------------------------- Technical Description -------------------------- #
    # In this function we want to load the weights in a reliable manner.
    #   Hence we want to load the weights with `strict=False` to guarantee everything is loaded as expected.
    #   To do so, we grab the respective submodules and load the fitting weights into them.
    #   We can do this through `get_submodule` which is a nn.Module function.
    #   However we need to cut-away the prefix of the matching keys to correctly assign weights from both `state_dicts`!
    # Difficulties:
    # 1) Different stem dimensions: When pre-training had only a single input channel, we need to make the shapes fit!
    #    To do so, we utilize repeating the weights N times (N = number of input channels).
    #    Limitation currently we only support this for a single input channel used during pre-training.
    # 2) Different patch sizes: The learned positional embeddings LPe of `Transformer` (Primus) architectures are
    #    patch size dependent. To adapt the weights, we do trilinear interpolation of these weights back to shape.
    # 3) Stem and Encoder merging: Most architectures (Primus, ResidualEncoderUNet derivatives) have
    #    separate `stem` and `encoder` objects. Hence we can separate stem and encoder weight loading easily.
    #    However in the `PlainConvUNet` architecture the encoder contains the stem, so we must make sure
    #    to skip the stem weight loading in the encoder, and then separately load the (repeated) stem weights

    # The following code does this.

    key_to_encoder = network.key_to_encoder  # Key to the encoder in the current network
    key_to_stem = network.key_to_stem  # Key to the stem (beginning) in the current network

    random_init_statedict = network.state_dict()
    ckpt_file = torch.load(pretrained_weights_path, weights_only=True)
    pre_train_statedict = ckpt_file["network_weights"]
    adaptation_plan = ckpt_file['nnssl_adaptation_plan']
    pt_input_channels = adaptation_plan['pretrain_num_input_channels']
    pt_input_patchsize = adaptation_plan['pretrain_patch_size'] if 'pretrain_patch_size' in adaptation_plan.keys() else [160,160,160]
    pt_key_to_encoder = adaptation_plan["key_to_encoder"]
    pt_key_to_stem = adaptation_plan["key_to_stem"]
    pt_keys_to_in_proj = tuple(adaptation_plan["keys_to_in_proj"])
    pt_key_to_lpe = adaptation_plan["key_to_lpe"],
    stem_in_encoder = pt_key_to_stem in pre_train_statedict

    # Currently we don't have the logic for interpolating the positional embedding yet.
    pt_weight_in_ch_mismatch = False
    need_to_ignore_lpe = False  # I.e. Learnable positional embedding
    key_to_lpe = getattr(network, "key_to_lpe", None)
    # Check if the current module even uses a learnable positional embedding. If not ignore LPE logic.
    try:
        network.get_submodule(key_to_lpe)
    except AttributeError:
        key_to_lpe = None

    if key_to_lpe is not None:
        # Add interpolation logic for positional embeddings later
        lpe_in_encoder = key_to_lpe.startswith(key_to_encoder)
        lpe_in_stem = key_to_lpe.startswith(key_to_stem)
        if pt_input_patchsize != downstream_input_patchsize:
            need_to_ignore_lpe = True  # LPE shape won't fit -> replace with random init
            #  We actually tested impact of using interpolated LPE and it's basically identical.
            #  So we just ignore it at the moment.
        # Should the LPE be neither in the encoder nor the stem, we don't need to specifically ignore it.
        #   However when the patch sizes are identical, we need to explicitly load it.

    def strip_dot_prefix(s) -> str:
        """Mini func to strip the dot prefix from the keys"""
        if s.startswith("."):
            return s[1:]
        return s

    # ----- Match the keys of pretrained weights to the current architecture ----- #
    if stem_in_encoder:
        encoder_weights = {k: v for k, v in pre_train_statedict.items() if k.startswith(pt_key_to_encoder)}
        if downstream_input_channels > pt_input_channels:
            pt_weight_in_ch_mismatch = True
            k_proj = pt_keys_to_in_proj[0] + ".weight"  # Get the projection weights
            vals = (
                       encoder_weights[k_proj].repeat(1, downstream_input_channels, 1, 1)
                   ) / downstream_input_channels
            for k in pt_keys_to_in_proj:
                encoder_weights[k] = vals
        # Fix the path to the weights:
        new_encoder_weights = {
            strip_dot_prefix(k.replace(pt_key_to_encoder, "")): v for k, v in encoder_weights.items()
        }
        # --------------------------------- Adapt LPE -------------------------------- #
        if need_to_ignore_lpe:
            if lpe_in_encoder:
                new_encoder_weights[strip_dot_prefix(key_to_lpe.replace(pt_key_to_encoder, ""))] = (
                    random_init_statedict[key_to_lpe]
                )

        # ------------------------------- Load weights ------------------------------- #
        encoder_module = network.get_submodule(key_to_encoder)
        encoder_module.load_state_dict(new_encoder_weights)
    else:
        encoder_weights = {k: v for k, v in pre_train_statedict.items() if k.startswith(pt_key_to_encoder)}
        stem_weights = {k: v for k, v in pre_train_statedict.items() if k.startswith(pt_key_to_stem)}
        if downstream_input_channels > pt_input_channels:
            pt_weight_in_ch_mismatch = True
            k_proj = pt_keys_to_in_proj[0] + ".weight"  # Get the projection weights
            vals = (
                       stem_weights[k_proj].repeat(1, downstream_input_channels, 1, 1, 1)
                   ) / downstream_input_channels
            for k in pt_keys_to_in_proj:
                stem_weights[k + ".weight"] = vals
        new_encoder_weights = {
            strip_dot_prefix(k.replace(pt_key_to_encoder, "")): v for k, v in encoder_weights.items()
        }
        new_stem_weights = {strip_dot_prefix(k.replace(pt_key_to_stem, "")): v for k, v in stem_weights.items()}
        # --------------------------------- Adapt LPE -------------------------------- #
        if need_to_ignore_lpe:
            if lpe_in_stem:  # Since stem not in encoder we need to take care of lpe in it here
                new_stem_weights[strip_dot_prefix(key_to_lpe.replace(key_to_stem, ""))] = (
                    random_init_statedict[key_to_lpe]
                )
            elif lpe_in_encoder:
                new_encoder_weights[strip_dot_prefix(key_to_lpe.replace(key_to_encoder, ""))] = (
                    random_init_statedict[key_to_lpe]
                )
            else:
                pass

        # ------------------------------- Load weights ------------------------------- #
        encoder_module = network.get_submodule(key_to_encoder)
        encoder_module.load_state_dict(new_encoder_weights)
        stem_module = network.get_submodule(key_to_stem)
        stem_module.load_state_dict(new_stem_weights)

    if not need_to_ignore_lpe and key_to_lpe is not None:
        # Load the positional embedding weights
        lpe_weights = {k: v for k, v in pre_train_statedict.items() if k.startswith(pt_key_to_lpe)}
        assert (
                len(lpe_weights) == 1
        ), f"Found multiple lpe weights, but expect only a single tensor. Got {list(lpe_weights.keys())}"
        network.get_parameter(key_to_lpe).data = list(lpe_weights.values())[0]
        # ------------------------------- Load weights ------------------------------- #


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ckp", type=str, help="path to ckpt")
    parser.add_argument(
        "-arch",
        choices=["ResEncL", "PrimusM"],
        required=True,
        help="Must be either 'ResEncL' or 'PrimusM'"
    )
    args = parser.parse_args()
    network = build_network_architecture(args.arch,None, None, [160,160,160], 1, 1)
    load_pretrained_weights(network,args.ckp,1,[160,160,160])
    print('Pretrained weights of the encoder loaded successfully')
