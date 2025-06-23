import torch
from torch import nn
from nnunetv2.probing.probe_architectures import ProbeArchitecture
from nnunetv2.training.nnUNetTrainer.pretraining.pretrainedTrainer import (
    PretrainedTrainer_Primus_150ep,
    PretrainedTrainer_Primus,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from nnunetv2.training.lr_scheduler.warmup import Lin_incr_LRScheduler, PolyLRScheduler_offset
from nnunetv2.utilities.helpers import empty_cache
from dynamic_network_architectures.architectures.abstract_arch import AbstractDynamicNetworkArchitectures
from dynamic_network_architectures.architectures.primus import Primus
from einops.layers.torch import Rearrange


class LinearProbeTrainer_Primus(PretrainedTrainer_Primus_150ep):

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
        self.configuration_manager.configuration["batch_size"] = 4  # Should easily fit into the GPU memory
        self.initial_lr = 1e-2
        self.weight_decay = 3e-5
        self.enable_deep_supervision = False
        self.use_pretrained_weights = use_pretrained_weights
        self.adaptation_info = self.plans_manager.plans["pretrain_info"]
        self.probe_position: int | None = None  # -1 means last layer, -2 means second to last layer, etc.
        self.network: ProbeArchitecture

    def maybe_rewire_network(self, network: AbstractDynamicNetworkArchitectures):
        """
        We insert a linear probe head at `self.probe_position` in the network that predicts the segmentation labels.

        Args:
            network (_type_): The network with loaded weights.

        Returns:
            _type_: the network with a linear probe head inserted at `self.probe_position`.
        """
        assert isinstance(network, Primus), "This trainer is only compatible with Primus architectures."
        probe_locations = {cnt: f"eva.blocks.{cnt}" for cnt in range(len(network.get_submodule("eva.blocks")))}
        n_probe_locations = len(probe_locations)
        probe_location = probe_locations[n_probe_locations + self.probe_position]
        assert (
            self.probe_position < 0
        ), "The linear probe position must be negative, e.g. -1 for the last layer, -2 for the second to last layer, etc."
        embedding_dim = network.eva.embed_dim
        output_dim = self.label_manager.num_segmentation_heads
        output_size = self.configuration_manager.patch_size

        # Currently the tokens are 8x8x8 (always) so we need to divide the input size by 8 to get the output size
        D, H, W = [int(x) // 8 for x in output_size]

        linear_probe_module = nn.Sequential(
            Rearrange("B (D H W) Ch -> B Ch D H W", D=D, H=H, W=W),
            nn.Conv3d(
                in_channels=embedding_dim,
                out_channels=output_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Upsample(
                size=output_size,
                mode="trilinear",
            ),
            # nn.Softmax(dim=1),  # Softmax over the channels, i.e. the segmentation classes
        )

        return ProbeArchitecture(
            network_to_probe=network, probe_position=probe_location, probe_module=linear_probe_module
        )

    def configure_optimizers(self, stage: str = "warmup_all"):
        self.network: ProbeArchitecture
        if isinstance(self.network, DDP):
            params = self.network.module.probe_module.parameters()
        else:
            params = self.network.probe_module.parameters()

        optimizer = torch.optim.SGD(
            params, self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True
        )
        lr_scheduler = PolyLRScheduler_offset(
            optimizer, self.initial_lr, self.num_epochs, self.warmup_duration_whole_net
        )
        self.training_stage = stage
        return optimizer, lr_scheduler

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.network.detach_probes()


class LinearProbeTrainer_Primus_M1(LinearProbeTrainer_Primus):
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
        self.probe_position: int = -1


class LinearProbeTrainer_Primus_M4(LinearProbeTrainer_Primus):
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
        self.probe_position: int = -4


class LinearProbeTrainer_Primus_M7(LinearProbeTrainer_Primus):
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
        self.probe_position: int = -7


class LinearProbeTrainer_Primus_M11(LinearProbeTrainer_Primus):
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
        self.probe_position: int = -11
