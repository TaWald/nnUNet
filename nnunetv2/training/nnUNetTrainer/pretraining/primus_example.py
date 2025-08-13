from dynamic_network_architectures.architectures.primus import Primus
import torch
from torch import nn
from timm.layers import RotaryEmbeddingCat

from dynamic_network_architectures.building_blocks.patch_encode_decode import LayerNormNd


# Loading the checkpoint and extracting the network architecture parameters
checkpoint = torch.load("/home/AD/b030s/mounts/omics_groups_OE0441/e230-thrp-data/mic_rocket/checkpoints/pretraining/MAE/Dataset804_Rocket_v3/BaseEvaMAETrainer_BS32_192ps_625ep_42_16_8_16_1056__nnsslPlans__noresample/fold_all/checkpoint_final.pth", map_location="cpu")
arch = checkpoint['nnssl_adaptation_plan']['architecture_plans']['arch_kwargs']
print(arch)


# Creating the test input tensor
im_dim = arch['input_shape']
x = torch.rand([1, 1, *im_dim], device="cpu", dtype=torch.float32)


# Initializing Primus with the extracted architecture parameters
primus_kwargs = dict(
    input_channels      = arch['input_channels'],
    embed_dim           = arch['embed_dim'],
    patch_embed_size    = tuple(arch['patch_embed_size']),
    num_classes         = arch['output_channels'], # !!! num_classes = output_channels
    eva_depth           = arch['encoder_eva_depth'] if 'encoder_eva_depth' in arch else 24, # !!! depth of the encoder
    eva_numheads        = arch['encoder_eva_numheads'] if 'encoder_eva_numheads' in arch else 16, # !!! number of heads in the encoder
    input_shape         = tuple(arch['input_shape']),
    decoder_norm        = arch['decoder_norm'] if 'decoder_norm' in arch else LayerNormNd,
    decoder_act         = arch['decoder_act'] if 'decoder_act' in arch else nn.GELU,
    num_register_tokens = arch['num_register_tokens'] if 'num_register_tokens' in arch else 0,
    drop_path_rate      = arch['drop_path_rate'] if 'drop_path_rate' in arch else 0.0,
    patch_drop_rate     = arch['patch_drop_rate'] if 'patch_drop_rate' in arch else 0.0,
    proj_drop_rate      = arch['proj_drop_rate'] if 'proj_drop_rate' in arch else 0.0,
    attn_drop_rate      = arch['attn_drop_rate'] if 'attn_drop_rate' in arch else 0.0,
    rope_impl           = arch['rope_impl'] if 'rope_impl' in arch else RotaryEmbeddingCat,
)
model = Primus(**primus_kwargs)



# Forward pass
with torch.no_grad():
    model.eval()
    output = model(x)