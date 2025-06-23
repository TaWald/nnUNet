import torch
from torch import nn
from nnunetv2.probing.probe_architectures import ProbeArchitecture
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.training.lr_scheduler.warmup import PolyLRScheduler_offset
from nnunetv2.training.nnUNetTrainer.pretraining.linearprobeTrainer import LinearProbeTrainer_Primus
from nnunetv2.utilities.helpers import empty_cache
from dynamic_network_architectures.architectures.abstract_arch import AbstractDynamicNetworkArchitectures
from dynamic_network_architectures.architectures.primus import Primus
from einops.layers.torch import Rearrange
from dynamic_network_architectures.building_blocks.patch_encode_decode import PatchDecode


class LightweightDecoderProbeTrainer_Primus(LinearProbeTrainer_Primus):

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
            PatchDecode(
                patch_size=(8, 8, 8),  # <-- This refers to the token patch size
                embed_dim=embedding_dim,
                out_channels=output_dim,
            ),  # <--- Now does the iterative upsampling instead of a single conv layer.
            # nn.Softmax(dim=1),  # Softmax over the channels, i.e. the segmentation classes
        )

        return ProbeArchitecture(
            network_to_probe=network, probe_position=probe_location, probe_module=linear_probe_module
        )


class LightweightDecoderProbeTrainer_Primus_M1(LightweightDecoderProbeTrainer_Primus):
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


class LightweightDecoderProbeTrainer_Primus_M4(LightweightDecoderProbeTrainer_Primus):
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


class LightweightDecoderProbeTrainer_Primus_M7(LightweightDecoderProbeTrainer_Primus):
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


class LightweightDecoderProbeTrainer_Primus_M11(LightweightDecoderProbeTrainer_Primus):
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


class LightweightDecoderProbeTrainer_Primus_M1_likeFrozen_nowarmup(LightweightDecoderProbeTrainer_Primus):
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
        self.initial_lr = 3e-5
        self.weight_decay = 0
        self.num_epochs = 150
        self.probe_position: int = -1

    def configure_optimizers(self, stage: str = "warmup_decoder"):
        stage = "train_decoder"

        if self.training_stage == stage:
            return self.optimizer, self.lr_scheduler
        self.network: ProbeArchitecture
        if isinstance(self.network, DDP):
            probe = self.network.module.probe_module.parameters()
        else:
            probe = self.network.probe_module.parameters()

        self.print_to_log_file("train decoder, poly lr")
        optimizer = torch.optim.AdamW(
            probe,
            self.initial_lr,
            weight_decay=self.weight_decay,
            amsgrad=False,
            betas=(0.9, 0.98),
            fused=True,
        )
        lr_scheduler = PolyLRScheduler_offset(
                optimizer,
                self.initial_lr,
                self.num_epochs,
                start_step=0,
            )
        self.print_to_log_file(f"Initialized warmup_all optimizer and lr_scheduler at epoch {self.current_epoch}")

        self.training_stage = stage
        empty_cache(self.device)
        return optimizer, lr_scheduler
