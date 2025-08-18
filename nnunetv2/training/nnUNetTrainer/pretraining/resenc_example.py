from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from dynamic_network_architectures.architectures.abstract_arch import AbstractDynamicNetworkArchitectures
import torch
from torch import nn
import pydoc


def load_pretrained_weights(
        network: AbstractDynamicNetworkArchitectures,
        pre_train_statedict: dict[str, torch.Tensor],
        pt_input_channels: int,
        downstream_input_channels: int,
        pt_input_patchsize: int,
        downstream_input_patchsize: tuple[int,...],
        pt_key_to_encoder: str,
        pt_key_to_stem: str,
        pt_keys_to_in_proj: tuple[str, ...],
        pt_key_to_lpe: str,
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
    stem_in_encoder = pt_key_to_stem in pre_train_statedict

    # Currently we don't have the logic for interpolating the positional embedding yet.
    pt_weight_in_ch_mismatch = False
    need_to_adapt_lpe = False  # I.e. Learnable positional embedding

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


        # ------------------------------- Load weights ------------------------------- #
        encoder_module = network.get_submodule(key_to_encoder)
        encoder_module.load_state_dict(new_encoder_weights)
        stem_module = network.get_submodule(key_to_stem)
        stem_module.load_state_dict(new_stem_weights)

    # Theoretically we don't need to return the network, but we do it anyway.
    return network, pt_weight_in_ch_mismatch

if __name__ == '__main__':

    # Loading the checkpoint and extracting the network architecture parameters
    checkpoint =  torch.load('/home/c306h/rocket_share/mic_rocket/checkpoints/sharing/rocket_cnn_mae_width_m_depth_m_bs8.pth', map_location="cpu")
    arch = checkpoint['nnssl_adaptation_plan']['architecture_plans']['arch_kwargs']
    requires_import = checkpoint['nnssl_adaptation_plan']['architecture_plans']['arch_kwargs_requiring_import']
    pre_train_statedict = checkpoint["network_weights"]
    # print(checkpoint['nnssl_adaptation_plan'])


    ################ depend on your dataset #######################
    patch_size = [192,192,192] # thats what we pretrained with, but can be tested to change for your dataset
    input_channels = 1 # depends on your task
    num_classes = 1  # depends on your tasks
    batch_size = 2
    ##############################################################

    # Creating the test input tensor
    x = torch.rand([batch_size, input_channels, *patch_size], device="cpu", dtype=torch.float32)

    for ri in requires_import:
        if arch[ri] is not None:
            arch[ri] = pydoc.locate(arch[ri])
    # Initializing Primus with the extracted architecture parameters
    resenc_kwargs = dict(
        input_channels              = input_channels,
        n_stages                    = arch['n_stages'],
        features_per_stage          = arch['features_per_stage'],
        conv_op                     = arch['conv_op'],
        kernel_sizes                = arch['kernel_sizes'],
        strides                     = arch['strides'],
        n_blocks_per_stage          = arch['n_blocks_per_stage'],
        num_classes                 = num_classes,
        n_conv_per_stage_decoder    = arch['n_conv_per_stage_decoder'],
        conv_bias                   = arch['conv_bias'],
        norm_op                     = arch['norm_op'],
        norm_op_kwargs              = arch['norm_op_kwargs'],
        nonlin                      = arch['nonlin'],
        nonlin_kwargs               = arch['nonlin_kwargs'],
        deep_supervision            = True
    )
    model = ResidualEncoderUNet(**resenc_kwargs)


    # this function is quite complicated, but allows for loading different checkpoints, stems, adapting positional embedding, skipping cls token...
    # we only load the "encoder" of the MAE. The MAE Decoder weights are skipped and up_projection conv layers stay random initialized
    model, _ = load_pretrained_weights(
        network                     = model,
        pre_train_statedict         = pre_train_statedict,
        pt_input_channels           = checkpoint['nnssl_adaptation_plan']['pretrain_num_input_channels'],
        downstream_input_channels   = input_channels,
        pt_input_patchsize          = checkpoint['nnssl_adaptation_plan']['recommended_downstream_patchsize'],
        downstream_input_patchsize  = patch_size,
        pt_key_to_encoder           = checkpoint['nnssl_adaptation_plan']["key_to_encoder"],
        pt_key_to_stem              = checkpoint['nnssl_adaptation_plan']["key_to_stem"],
        pt_keys_to_in_proj          = tuple(checkpoint['nnssl_adaptation_plan']["keys_to_in_proj"]),
        pt_key_to_lpe               = checkpoint['nnssl_adaptation_plan']["key_to_lpe"],
    )

    with torch.no_grad():
        model.eval()
        output = model(x)




'''
Some notes for custom training pipeline

- Pretrained patch size [192,192,192]
- Zscore normalization
- We did not use a fixed the target spacing during pretraining -> can be selected depending on your dataset

torch.optim.SGD(
                params, self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True
            )
- initial_lr = 1e-3
-weight_decay = 3e-5

- Recommended LR scheduler:
    ~ 10% of training linear increasing lr
    then decreasing poly lr
'''