from typing import List
from nnunetv2.training.lr_scheduler.warmup import (
    Lin_incr_LRScheduler,
    PolyLRScheduler_offset,
    Lin_incr_offset_LRScheduler,
)
from dynamic_network_architectures.architectures.abstract_arch import AbstractDynamicNetworkArchitectures
from nnunetv2.training.nnUNetTrainer.pretraining.pretrainedTrainer import PretrainedTrainer_Primus
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from nnunetv2.utilities.helpers import empty_cache
import numpy as np


class PretrainedTrainer_Primus_frozenEnc(PretrainedTrainer_Primus):
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
        self.warmup_duration_decoder = 15
        self.initial_lr = 1e-4
        self.weight_decay = 5e-2
        self.num_epochs = 150
        self.training_stage = None
        self.nan_threshold = 3
        self.nan_counter = 0

    def get_stage(self):
        if self.current_epoch < self.warmup_duration_decoder:
            return "warmup_decoder"
        else:
            return "train_decoder"

    def on_train_epoch_start(self):
        if self.current_epoch == 0:
            self.optimizer, self.lr_scheduler = self.configure_optimizers("warmup_decoder")
        elif self.current_epoch == int(self.warmup_duration_decoder):
            self.optimizer, self.lr_scheduler = self.configure_optimizers("train_decoder")

        self.network.train()
        self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file("")
        self.print_to_log_file(f"Epoch {self.current_epoch}")
        self.print_to_log_file(f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log("lrs", self.optimizer.param_groups[0]["lr"], self.current_epoch)

    def on_train_epoch_end(self, train_outputs: List[dict]):
        super().on_train_epoch_end(train_outputs)
        loss = self.logger.my_fantastic_logging["train_losses"][-1]
        if np.isnan(loss) or np.isinf(loss):
            self.nan_counter += 1
            self.print_to_log_file(f"Loss is NaN or Inf, nan_counter: {self.nan_counter}")
            if self.nan_counter >= self.nan_threshold:
                self.print_to_log_file("Stopping training due to NaN or Inf loss.")
                raise ValueError("Loss is NaN or Inf, stopping training.")

    def configure_optimizers(self, stage: str = "warmup_decoder"):
        assert stage in ["warmup_decoder", "train_decoder"]

        if self.training_stage == stage:
            return self.optimizer, self.lr_scheduler
        self.network: AbstractDynamicNetworkArchitectures
        if isinstance(self.network, DDP):
            heads = self.network.module.up_projection.parameters()
        else:
            heads = self.network.up_projection.parameters()

        if stage == "warmup_decoder":
            self.print_to_log_file("train decoder, lin warmup")
            optimizer = torch.optim.AdamW(
                heads, self.initial_lr, weight_decay=self.weight_decay, amsgrad=False, betas=(0.9, 0.98), fused=True
            )
            lr_scheduler = Lin_incr_LRScheduler(optimizer, self.initial_lr, int(self.warmup_duration_decoder))
        elif stage == "train_decoder":
            self.print_to_log_file("train decoder, poly lr")
            optimizer = torch.optim.AdamW(
                heads,
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
                int(self.warmup_duration_decoder),
            )
            self.print_to_log_file(f"Initialized warmup_all optimizer and lr_scheduler at epoch {self.current_epoch}")

        self.training_stage = stage
        empty_cache(self.device)
        return optimizer, lr_scheduler


class PretrainedTrainer_Primus_frozenEnc_lr_1e4_wd_5e3(PretrainedTrainer_Primus_frozenEnc):
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
        self.warmup_duration_decoder = 15
        self.initial_lr = 1e-4
        self.weight_decay = 5e-3
        self.num_epochs = 150
        self.training_stage = None


class PretrainedTrainer_Primus_frozenEnc_lr_1e4_wd_5e4(PretrainedTrainer_Primus_frozenEnc):
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
        self.warmup_duration_decoder = 15
        self.initial_lr = 1e-4
        self.weight_decay = 5e-4
        self.num_epochs = 150
        self.training_stage = None


class PretrainedTrainer_Primus_frozenEnc_lr_1e4_wd_5e5(PretrainedTrainer_Primus_frozenEnc):
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
        self.warmup_duration_decoder = 15
        self.initial_lr = 1e-4
        self.weight_decay = 5e-5
        self.num_epochs = 150
        self.training_stage = None


class PretrainedTrainer_Primus_frozenEnc_lr_1e4_wd_0(PretrainedTrainer_Primus_frozenEnc):
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
        self.warmup_duration_decoder = 15
        self.initial_lr = 1e-4
        self.weight_decay = 0
        self.num_epochs = 150
        self.training_stage = None


class PretrainedTrainer_Primus_frozenEnc_lr_3e5_wd_0(PretrainedTrainer_Primus_frozenEnc):
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
        self.warmup_duration_decoder = 15
        self.initial_lr = 3e-5
        self.weight_decay = 0
        self.num_epochs = 150
        self.training_stage = None


class PretrainedTrainer_Primus_frozenEnc_lr_7e5_wd_0(PretrainedTrainer_Primus_frozenEnc):
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
        self.warmup_duration_decoder = 15
        self.initial_lr = 7e-5
        self.weight_decay = 0
        self.num_epochs = 150
        self.training_stage = None


class PretrainedTrainer_Primus_frozenEnc_lr_7e5_wd_3e5(PretrainedTrainer_Primus_frozenEnc):
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
        self.warmup_duration_decoder = 15
        self.initial_lr = 7e-5
        self.weight_decay = 3e-5
        self.num_epochs = 150
        self.training_stage = None


class PretrainedTrainer_Primus_frozenEnc_noWarmUp_lr_3e5_wd_5e2(PretrainedTrainer_Primus_frozenEnc):
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
        self.warmup_duration_decoder = 0
        self.initial_lr = 3e-5
        self.weight_decay = 5e-2
        self.num_epochs = 150
        self.training_stage = None

    def configure_optimizers(self, stage: str = "warmup_decoder"):
        stage = "train_decoder"

        if self.training_stage == stage:
            return self.optimizer, self.lr_scheduler
        self.network: AbstractDynamicNetworkArchitectures
        if isinstance(self.network, DDP):
            heads = self.network.module.up_projection.parameters()
        else:
            heads = self.network.up_projection.parameters()

        self.print_to_log_file("train decoder, poly lr")
        optimizer = torch.optim.AdamW(
            heads,
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


class PretrainedTrainer_Primus_frozenEnc_noWarmUp_lr_3e5_wd_0(PretrainedTrainer_Primus_frozenEnc):
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
        self.warmup_duration_decoder = 0
        self.initial_lr = 3e-5
        self.weight_decay = 0
        self.num_epochs = 150
        self.training_stage = None

    def configure_optimizers(self, stage: str = "warmup_decoder"):
        stage = "train_decoder"

        if self.training_stage == stage:
            return self.optimizer, self.lr_scheduler
        self.network: AbstractDynamicNetworkArchitectures
        if isinstance(self.network, DDP):
            heads = self.network.module.up_projection.parameters()
        else:
            heads = self.network.up_projection.parameters()

        self.print_to_log_file("train decoder, poly lr")
        optimizer = torch.optim.AdamW(
            heads,
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
