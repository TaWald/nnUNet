from dynamic_network_architectures.architectures.primus import Primus
from dynamic_network_architectures.architectures.abstract_arch import AbstractDynamicNetworkArchitectures
import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F

def handle_pos_embed_resize(pretrained_dict, model_dict, mode, input_shape=None, pretrained_input_patch_size=None, patch_embed_size=None):
    pretrained_pos_embed = pretrained_dict["pos_embed"]
    model_pos_embed = model_dict["pos_embed"]
    model_pos_embed_shape = model_pos_embed.shape

    # for key, value in pretrained_dict.items():
    #     print(f"{key}: {value.shape}")

    has_cls_token = "cls_token" in pretrained_dict



    if has_cls_token:
        cls_pos_embed = pretrained_pos_embed[:, :1, :]
        patch_pos_embed = pretrained_pos_embed[:, 1:, :]
    else:
        if  "cls_token" in model_dict.keys():
            cls_pos_embed = model_pos_embed[:, :1, :]
        patch_pos_embed = pretrained_pos_embed

    if mode == "interpolate":
        resized_patch_pos_embed = interpolate_patch_embed_1d(patch_pos_embed, target_len=model_pos_embed_shape[1] - int(has_cls_token))

    elif mode == "interpolate_trilinear":
        # Calculate input/output 3D shapes
        in_shape = dict(zip("xyz", [int(d / p) for d, p in zip(pretrained_input_patch_size, patch_embed_size)]))
        out_shape = dict(zip("xyz", [int(d / p) for d, p in zip(input_shape, patch_embed_size)]))
        resized_patch_pos_embed = interpolate_patch_embed_3d(patch_pos_embed, in_shape, out_shape)

    else:
        raise NotImplementedError(f"Unknown resize mode: {mode}")
    if "cls_token" in model_dict.keys():
        resized_pos_embed = torch.cat([cls_pos_embed, resized_patch_pos_embed], dim=1)
    else:
        resized_pos_embed = resized_patch_pos_embed
    pretrained_dict["pos_embed"] = resized_pos_embed

def interpolate_patch_embed_1d(patch_embed, target_len, mode="linear"):
    """Resizes patch embeddings using interpolation."""
    return F.interpolate(
        patch_embed.permute(0, 2, 1),  # [B, C, Tokens]
        size=target_len,
        mode=mode,
        align_corners=False,
    ).permute(0, 2, 1)  # [B, Tokens, C]

def interpolate_patch_embed_3d(patch_embed, in_shape, out_shape):
    """Resizes patch embeddings using 3D trilinear interpolation."""
    patch_embed = patch_embed.permute(0, 2, 1)
    patch_embed = rearrange(patch_embed, "B C (x y z) -> B C x y z", **in_shape)
    patch_embed = F.interpolate(patch_embed, size=list(out_shape.values()), mode="trilinear", align_corners=False)
    patch_embed = rearrange(patch_embed, "B C x y z -> B C (x y z)", **out_shape)
    return patch_embed.permute(0, 2, 1)

def filter_state_dict(state_dict, skip_strings):
    found_flag = False
    filtered_state_dict = {}

    for k, v in state_dict.items():
        if any(skip in k for skip in skip_strings):
            found_flag = True
            continue
        filtered_state_dict[k] = v

    return filtered_state_dict, found_flag

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
    key_to_lpe = getattr(network, "key_to_lpe", None)

    if key_to_lpe is not None:
        # Add interpolation logic for positional embeddings later
        lpe_in_encoder = key_to_lpe.startswith(key_to_encoder)
        lpe_in_stem = key_to_lpe.startswith(key_to_stem)
        if pt_input_patchsize != downstream_input_patchsize:
            need_to_adapt_lpe = True  # LPE shape won't fit ->  interpolate LPE

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
        if need_to_adapt_lpe:
            if lpe_in_encoder:
                handle_pos_embed_resize(new_encoder_weights,
                                        network.get_submodule(key_to_encoder).state_dict(),
                                        'interpolate_trilinear',
                                        downstream_input_patchsize,
                                        pt_input_patchsize,
                                        new_encoder_weights['eva.pos_embed'][2])
                print('did not believe it would work', new_encoder_weights['eva.pos_embed'][2])

        if "cls_token" in encoder_weights.keys():
            skip_strings_in_pretrained = ["cls_token"]
            new_encoder_weights, found_cls_token = filter_state_dict(encoder_weights, skip_strings_in_pretrained)

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
        if need_to_adapt_lpe:
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
        if "cls_token" in encoder_weights.keys():
            skip_strings_in_pretrained = ["cls_token"]
            new_encoder_weights, found_cls_token = filter_state_dict(encoder_weights, skip_strings_in_pretrained)

        # ------------------------------- Load weights ------------------------------- #
        encoder_module = network.get_submodule(key_to_encoder)
        encoder_module.load_state_dict(new_encoder_weights)
        stem_module = network.get_submodule(key_to_stem)
        stem_module.load_state_dict(new_stem_weights)


    if not need_to_adapt_lpe and key_to_lpe is not None:
        # Load the positional embedding weights
        lpe_weights = {k: v for k, v in pre_train_statedict.items() if k.startswith(pt_key_to_lpe)}
        assert (
                len(lpe_weights) == 1
        ), f"Found multiple lpe weights, but expect only a single tensor. Got {list(lpe_weights.keys())}"
        network.get_parameter(key_to_lpe).data = list(lpe_weights.values())[0]
        # ------------------------------- Load weights ------------------------------- #

    del pre_train_statedict, encoder_weights,  new_encoder_weights, new_stem_weights, stem_weights
    # Theoretically we don't need to return the network, but we do it anyway.
    return network, pt_weight_in_ch_mismatch

if __name__ == '__main__':

    # Loading the checkpoint and extracting the network architecture parameters
    checkpoint = torch.load("/home/c306h/rocket_share/mic_rocket/checkpoints/pretraining/MAE/Dataset804_Rocket_v3/BaseEvaMAETrainer_BS32_192ps_625ep_42_16_8_16_1056__nnsslPlans__noresample/fold_all/checkpoint_final.pth", map_location="cpu")
    arch = checkpoint['nnssl_adaptation_plan']['architecture_plans']['arch_kwargs']
    pre_train_statedict = checkpoint["network_weights"]
    print(checkpoint['nnssl_adaptation_plan'])


    ################ depend on your dataset #######################
    patch_size = [192,192,192] # thats what we pretrained with, but can be tested to change for your dataset
    input_channels = 1 # depends on your task
    num_classes = 1  # depends on your tasks
    batch_size = 2
    ##############################################################

    # Creating the test input tensor
    x = torch.rand([batch_size, input_channels, *patch_size], device="cpu", dtype=torch.float32)


    # Initializing Primus with the extracted architecture parameters
    primus_kwargs = dict(
        input_channels      = input_channels,
        embed_dim           = arch['embed_dim'],
        patch_embed_size    = tuple(arch['patch_embed_size']),
        num_classes         = num_classes,
        eva_depth           = arch['encoder_eva_depth'], # !!! depth of the encoder
        eva_numheads        = arch['encoder_eva_numheads'], # !!! number of heads in the encoder
        input_shape         = patch_size,
        drop_path_rate      = 0.2,
        scale_attn_inner    = True,
        init_values         = 0.1,
    )
    model = Primus(**primus_kwargs)


    # this function is quite complicated, but allows for loading different checkpoints, stems, adapting positional embedding, skipping cls token...
    # we only load the "encoder" of the MAE. The MAE Decoder weights are skipped and up_projection conv layers stay random initialized
    model, _ = load_pretrained_weights(
        network                     = model,
        pre_train_statedict         = pre_train_statedict,
        pt_input_channels           = checkpoint['nnssl_adaptation_plan']['architecture_plans']['arch_kwargs']["input_channels"],
        downstream_input_channels   = input_channels,
        pt_input_patchsize          = checkpoint['nnssl_adaptation_plan']['architecture_plans']['arch_kwargs']["input_shape"],
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

- optimizer = torch.optim.AdamW(
                    params,
                    self.initial_lr,
                    weight_decay=self.weight_decay,
                    amsgrad=False,
                    betas=(0.9, 0.98),
                    fused=True,
                )
- initial_lr = 1e-4
- weight_decay = 5e-2

- Recommended LR scheduler:
    ~ 10% of training linear increasing lr
    then decreasing poly lr
'''