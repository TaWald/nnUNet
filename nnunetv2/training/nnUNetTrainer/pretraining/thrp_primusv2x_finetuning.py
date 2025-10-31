from copy import deepcopy
from typing import Literal, Tuple, Union, List
import torch
from batchgenerators.utilities.file_and_folder_operations import isfile
from dynamic_network_architectures.architectures.abstract_arch import AbstractDynamicNetworkArchitectures
from torch._dynamo import OptimizedModule
import numpy as np
from nnunetv2.utilities.load_weights_utils import *
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.lr_scheduler.warmup import Lin_incr_LRScheduler, PolyLRScheduler_offset, Lin_incr_offset_LRScheduler
from nnunetv2.utilities.get_network_via_name import get_network_from_name
from torch import nn, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.training.nnUNetTrainer.pretraining.pretrainedTrainer import PretrainedTrainer_Primus
from nnunetv2.architectures.primusv2 import Primusv2

warmup_stages = Literal["warmup_all", "warmup_decoder", "train_all", "train_decoder"]

class PretrainedTrainer_Primusv2x(PretrainedTrainer_Primus):

    def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            use_pretrained_weights: bool = True,
            device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, use_pretrained_weights, device)
        # Can be overriden to train same architecture from scratch.
        self.initial_lr = 1e-4
        self.weight_decay = 5e-2
        self.enable_deep_supervision = False
        self.warmup_duration_whole_net = 50  # lin increase whole network
        self.use_pretrained_weights = use_pretrained_weights
        if not self.use_pretrained_weights:
            self.initial_lr = 3e-4
        self.adaptation_info = self.plans_manager.plans["pretrain_info"]

    def initialize(self):
        if not self.was_initialized:
            ## DDP batch size and oversampling can differ between workers and needs adaptation
            # we need to change the batch size in DDP because we don't use any of those distributed samplers
            self._set_batch_size_and_oversample()

            self.num_input_channels = determine_num_input_channels(
                self.plans_manager, self.configuration_manager, self.dataset_json
            )

            # During `nnUNetv2_preprocess_like_nnssl` we create a new plan that specifies the architecture already.
            #   This plan holds details on how the architecture is supposed to be built.

            self.network = self.build_network_architecture(
                architecture_class_name=self.configuration_manager.network_arch_class_name,
                arch_init_kwargs=self.configuration_manager.network_arch_init_kwargs,
                arch_init_kwargs_req_import=self.configuration_manager.network_arch_init_kwargs_req_import,
                input_patch_size=self.configuration_manager.patch_size,  # Set in plan to pt_recommended_patchsize
                num_input_channels=self.num_input_channels,
                num_output_channels=self.label_manager.num_segmentation_heads,
                enable_deep_supervision=False,
            ).to(self.device)

            # Load pretrained weights
            if self.use_pretrained_weights:
                assert (
                        "checkpoint_path" in self.adaptation_info
                ), "`checkpoint_path` not found in plans! Can't load weights"
                assert isfile(
                    self.adaptation_info["checkpoint_path"]
                ), f"Pretrained weights path {self.adaptation_info['checkpoint_path']} does not exist!"
                self.network,  self.pt_weight_in_ch_mismatch = self.load_pretrained_weights(
                    self.network,
                    pretrained_weights_path=self.adaptation_info["checkpoint_path"],
                    pt_input_channels=self.adaptation_info["pt_num_in_channels"],
                    downstream_input_channels=self.num_input_channels,
                    pt_input_patchsize=self.adaptation_info["pt_used_patchsize"],
                    downstream_input_patchsize=self.configuration_manager.patch_size,
                    pt_key_to_encoder=self.adaptation_info["key_to_encoder"],
                    pt_key_to_stem=self.adaptation_info["key_to_stem"],
                    pt_keys_to_in_proj=tuple(self.adaptation_info["keys_to_in_proj"]),
                    pt_key_to_lpe=self.adaptation_info["key_to_lpe"],
                )
                self.print_citations()
                self.print_to_log_file("Loaded Network from {}".format(self.adaptation_info["checkpoint_path"]))
            else:
                self.print_to_log_file("You are using a Trainer for fine-tuning but without loading weigts")
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file("Using torch.compile...")
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = self.network.to(self.device)
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()

            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

            # torch 2.2.2 crashes upon compiling CE loss
            # if self._do_i_compile():
            #     self.loss = torch.compile(self.loss)
            self.was_initialized = True
        else:
            raise RuntimeError(
                "You have called self.initialize even though the trainer was already initialized. "
                "That should not happen."
            )

    def build_network_architecture(
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        input_patch_size: tuple[int, int, int],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
        ) -> nn.Module:
        # this architecture will crash if the patch size is not divisible by 8!
        if 'init_values' in arch_init_kwargs:
            if isinstance(arch_init_kwargs['init_values'], list):
                arch_init_kwargs['init_values'] = arch_init_kwargs['init_values'][0]
        model = Primusv2(
            num_input_channels,
            arch_init_kwargs['embed_dim'],
            arch_init_kwargs['patch_embed_size'],
            num_output_channels,
            arch_init_kwargs['encoder_eva_depth'],
            arch_init_kwargs['encoder_eva_numheads'],
            input_patch_size,
            drop_path_rate=0.2,
            scale_attn_inner=arch_init_kwargs['scale_attn_inner'],
            init_values=arch_init_kwargs['init_values'],
        )
        return model

    @staticmethod
    def load_pretrained_weights(
            network: AbstractDynamicNetworkArchitectures,
            pretrained_weights_path: str,
            pt_input_channels: int,
            downstream_input_channels: int,
            pt_input_patchsize: int,
            downstream_input_patchsize: int,
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
        ckp =  torch.load(pretrained_weights_path, weights_only=True)
        pre_train_statedict: dict[str, torch.Tensor] =ckp["network_weights"]  # Get pre-trained state dict

        #take info from ckpt path (allows to overwrite plan specifications)
        if 'key_to_stem' in  ckp['nnssl_adaptation_plan'].keys():
            pt_key_to_stem =  ckp['nnssl_adaptation_plan']['key_to_stem']
        if 'key_to_encoder' in  ckp['nnssl_adaptation_plan'].keys():
            pt_key_to_encoder =  ckp['nnssl_adaptation_plan']['key_to_encoder']
        if 'keys_to_in_proj' in  ckp['nnssl_adaptation_plan'].keys():
            pt_keys_to_in_proj =  ckp['nnssl_adaptation_plan']['keys_to_in_proj']
        if 'key_to_lpe' in  ckp['nnssl_adaptation_plan'].keys():
            pt_key_to_lpe =  ckp['nnssl_adaptation_plan']['key_to_lpe']

        ####allows overwrites (e.g for voco needed)
        if 'nnssl_adaptation_plan' in ckp.keys():
            if 'pretrain_patch_size' in ckp['nnssl_adaptation_plan'].keys():
                pt_input_patchsize = ckp['nnssl_adaptation_plan']['pretrain_patch_size']



        stem_in_encoder = pt_key_to_stem in pre_train_statedict

        # Currently we don't have the logic for interpolating the positional embedding yet.
        pt_weight_in_ch_mismatch = False
        need_to_adapt_lpe = False  # I.e. Learnable positional embedding
        key_to_lpe = getattr(network, "key_to_lpe", None)
        lpe_in_stem = False

        # # Check if the current module even uses a learnable positional embedding. If not ignore LPE logic.
        # try:
        #     network.get_submodule(key_to_lpe)
        # except AttributeError:
        #     key_to_lpe = None

        if key_to_lpe is not None:
            lpe_in_encoder = key_to_lpe.startswith(key_to_encoder)
            lpe_in_stem = key_to_lpe.startswith(key_to_stem)
            if pt_input_patchsize != downstream_input_patchsize:
                if lpe_in_stem is not None or lpe_in_encoder is not None:
                    need_to_adapt_lpe = True # LPE shape won't fit -> resize it


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
                                            new_encoder_weights['down_projection.proj.weight'].shape[2:])
                    new_encoder_weights["pos_embed"].to(next(network.parameters()).device)
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
                    handle_pos_embed_resize(new_stem_weights,
                                            network.get_submodule(key_to_stem).state_dict(),
                                            'interpolate_trilinear',
                                            downstream_input_patchsize,
                                            pt_input_patchsize,
                                            new_stem_weights['proj.weight'].shape[2:])
                    new_stem_weights["pos_embed"].to(next(network.parameters()).device)
                elif lpe_in_encoder:
                    handle_pos_embed_resize(new_encoder_weights,
                                            network.get_submodule(key_to_encoder).state_dict(),
                                            'interpolate_trilinear',
                                            downstream_input_patchsize,
                                            pt_input_patchsize,
                                            new_stem_weights['proj.weight'].shape[2:])
                    new_encoder_weights["pos_embed"].to(next(network.parameters()).device)
                else:
                    pass
            if "cls_token" in encoder_weights.keys():
                skip_strings_in_pretrained = ["cls_token"]
                new_encoder_weights, found_cls_token = filter_state_dict(encoder_weights, skip_strings_in_pretrained)

            # ------------------------------- Load weights ------------------------------- #
            encoder_module = network.get_submodule(key_to_encoder)
            encoder_module.load_state_dict(new_encoder_weights)
            # stem_module = network.get_submodule(key_to_stem)
            # stem_module.load_state_dict(new_stem_weights)
            del  new_stem_weights, stem_weights


        if not need_to_adapt_lpe and key_to_lpe is not None:
            # Load the positional embedding weights
            lpe_weights = {k: v for k, v in pre_train_statedict.items() if k.startswith(pt_key_to_lpe)}
            assert (
                    len(lpe_weights) == 1
            ), f"Found multiple lpe weights, but expect only a single tensor. Got {list(lpe_weights.keys())}"
            network.get_parameter(key_to_lpe).data = list(lpe_weights.values())[0].to(next(network.parameters()).device)
            # ------------------------------- Load weights ------------------------------- #

        # Theoretically we don't need to return the network, but we do it anyway.
        del pre_train_statedict, encoder_weights,  new_encoder_weights
        return network, pt_weight_in_ch_mismatch

class PretrainedTrainer_Primusv2x_150ep(PretrainedTrainer_Primusv2x):

    def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            use_pretrained_weights: bool = True,
            device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, use_pretrained_weights, device)
        # Can be overriden to train same architecture from scratch.
        self.warmup_duration_whole_net = 15  # lin increase whole network
        self.num_epochs = 150 # lin increase whole network

class PretrainedTrainer_Primusv2x_150ep_small_debug(PretrainedTrainer_Primusv2x):

    def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            use_pretrained_weights: bool = True,
            device: torch.device = torch.device("cuda"),
    ):
        plans["configurations"][configuration]["patch_size"] = (48, 48, 48)
        plans["configurations"][configuration]["batch_size"] = 2
        super().__init__(plans, configuration, fold, dataset_json, use_pretrained_weights, device)
        # Can be overriden to train same architecture from scratch.
        self.warmup_duration_whole_net = 15  # lin increase whole network
        self.num_epochs = 150 # lin increase whole network

class PretrainedTrainer_Primusv2x_150ep_nomirroring(PretrainedTrainer_Primusv2x_150ep):
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes


class PretrainedTrainer_Primusv2x_150ep_warmup(PretrainedTrainer_Primusv2x):

    def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            use_pretrained_weights: bool = True,
            device: torch.device = torch.device("cuda"),
    ):
        plans["configurations"][configuration]["patch_size"] = (48, 48, 48)
        plans["configurations"][configuration]["batch_size"] = 1
        super().__init__(plans, configuration, fold, dataset_json, use_pretrained_weights, device)
        # Can be overriden to train same architecture from scratch.
        self.initial_lr = 1e-4
        self.warmup_lr_factor = 0.1 #during decoder warmup lr must be smaller otherwise training collaps
        self.weight_decay = 5e-2
        self.warmup_duration_decoder = 15
        self.warmup_duration_whole_net = 50  # lin increase whole network
        self.num_epochs = 150


    def on_train_epoch_start(self):
        if self.current_epoch == 0:
            self.optimizer, self.lr_scheduler = self.configure_optimizers('warmup_decoder')
        if self.current_epoch == int(self.warmup_duration_decoder//2):
            self.optimizer, self.lr_scheduler = self.configure_optimizers('train_decoder')
        elif self.current_epoch == self.warmup_duration_decoder:
            self.optimizer, self.lr_scheduler = self.configure_optimizers('warmup_all')
        elif self.current_epoch == self.warmup_duration_whole_net + self.warmup_duration_decoder:
            self.optimizer, self.lr_scheduler = self.configure_optimizers('train')
            self.network.train()
        self.network.train()
        self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)


    def get_stage(self):
        if self.current_epoch < self.warmup_duration_decoder//2:
            return 'warmup_decoder'
        elif self.current_epoch < self.warmup_duration_decoder and self.current_epoch >= self.warmup_duration_decoder//2:
            return 'train_decoder'
        elif self.current_epoch < self.warmup_duration_whole_net and self.current_epoch >= self.warmup_duration_decoder:
            return 'warmup_all'
        else:
            return 'train'

    def configure_optimizers(self, stage: str = 'warmup_all'):
        assert stage in ['warmup_all', 'train', 'warmup_decoder', 'train_decoder']

        if self.training_stage == stage:
            return self.optimizer, self.lr_scheduler

        if isinstance(self.network, DDP):
            params = self.network.module.parameters()
            heads = self.network.module.up_projection.parameters()
            in_proj_params = []
            for k in self.network.module.keys_to_in_proj:
                in_proj_params += list(self.network.module.get_submodule(k).parameters())
            rnd_param = list(heads) + list(in_proj_params)

        else:
            params = self.network.parameters()
            # print(self.network.state_dict().keys())
            heads = self.network.up_projection.parameters()
            in_proj_params = []
            for k in self.network.keys_to_in_proj:
                in_proj_params += list(self.network.get_submodule(k).parameters())
            rnd_param = list(heads) + list(in_proj_params)


        if stage == 'warmup_decoder':
            self.print_to_log_file("train decoder and stem and inprojection, lin warmup")
            optimizer =  torch.optim.AdamW(rnd_param, self.initial_lr, weight_decay=self.weight_decay,
                                           amsgrad=False, betas=(0.9, 0.98), fused=True)
            lr_scheduler = Lin_incr_LRScheduler(optimizer, self.initial_lr*self.warmup_lr_factor, int(self.warmup_duration_decoder//2))
            self.print_to_log_file(f"Initialized warmup only decoder, stem and inprojection optimizer and lr_scheduler at epoch {self.current_epoch}")
        elif stage == 'train_decoder':
            self.print_to_log_file("train decoder and stem and inprojection, poly lr")
            if self.training_stage == 'warmup_decoder':
                # we can keep the existing optimizer and don't need to create a new one. This will allow us to keep
                # the accumulated momentum terms which already point in a useful driection
                optimizer = self.optimizer
            else:
                optimizer = torch.optim.AdamW(heads, self.initial_lr, weight_decay=self.weight_decay,
                                              amsgrad=False, betas=(0.9, 0.98), fused=True)
            lr_scheduler = PolyLRScheduler_offset(optimizer, self.initial_lr*self.warmup_lr_factor, self.warmup_duration_decoder, int(self.warmup_duration_decoder//2))
            self.print_to_log_file(f"Initialized train only decoder optimizer and lr_scheduler at epoch {self.current_epoch}")

        elif stage == 'warmup_all':
            self.print_to_log_file("train whole net, warmup")
            optimizer = torch.optim.AdamW(params, self.initial_lr, weight_decay=self.weight_decay,
                                          amsgrad=False, betas=(0.9, 0.98), fused=True)
            lr_scheduler = Lin_incr_offset_LRScheduler(optimizer, self.initial_lr, self.warmup_duration_decoder + self.warmup_duration_whole_net,  self.warmup_duration_decoder)
            self.print_to_log_file(f"Initialized warmup_all optimizer and lr_scheduler at epoch {self.current_epoch}")
        elif stage == 'train':
            self.print_to_log_file("train whole net")
            if self.training_stage == 'warmup_all':
                self.print_to_log_file("train whole net, warmup")
                # we can keep the existing optimizer and don't need to create a new one. This will allow us to keep
                # the accumulated momentum terms which already point in a useful driection
                optimizer = self.optimizer
            else:
                self.print_to_log_file("train whole net, poly lr")
                optimizer = torch.optim.AdamW(params, self.initial_lr, weight_decay=self.weight_decay,
                                              amsgrad=False, betas=(0.9, 0.98), fused=True)
            lr_scheduler = PolyLRScheduler_offset(optimizer, self.initial_lr, self.num_epochs, self.warmup_duration_whole_net + self.warmup_duration_decoder)
            self.print_to_log_file(f"Initialized train optimizer and lr_scheduler at epoch {self.current_epoch}")
        self.training_stage = stage
        empty_cache(self.device)
        return optimizer, lr_scheduler


class PretrainedTrainer_Primusv2x_150ep_warmup_nomirroring(PretrainedTrainer_Primusv2x_150ep_warmup):
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes