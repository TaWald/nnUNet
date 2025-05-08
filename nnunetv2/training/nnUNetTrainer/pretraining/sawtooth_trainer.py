from nnunetv2.training.lr_scheduler.warmup import Lin_incr_LRScheduler, PolyLRScheduler_offset, \
    Lin_incr_offset_LRScheduler
from nnunetv2.training.nnUNetTrainer.pretraining.pretrainedTrainer import PretrainedTrainer_Primus, PretrainedTrainer
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from nnunetv2.utilities.helpers import empty_cache
import numpy as np

class PretrainedTrainer_sawtooth(PretrainedTrainer):
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
        self.warmup_duration_decoder = 50
        self.initial_lr = 1e-3
        self.warmup_duration_whole_net = 50  # lin increase whole network
        self.num_epochs = 1000
        self.training_stage = None

    def get_stage(self):
        if self.current_epoch < self.warmup_duration_decoder//2:
            return 'warmup_decoder'
        elif self.current_epoch < self.warmup_duration_decoder and self.current_epoch >= self.warmup_duration_decoder//2:
            return 'train_decoder'
        elif self.current_epoch < self.warmup_duration_whole_net and self.current_epoch >= self.warmup_duration_decoder:
            return 'warmup_all'
        else:
            return 'train'
   
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
        self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)

    def configure_optimizers(self, stage: str = 'warmup_all'):
        assert stage in ['warmup_all', 'train', 'warmup_decoder', 'train_decoder']

        if self.training_stage == stage:
            return self.optimizer, self.lr_scheduler

        if isinstance(self.network, DDP):
            params = self.network.module.parameters()
            heads = self.network.module.decoder.parameters()
        else:
            params = self.network.parameters()
            # print(self.network.state_dict().keys())
            heads = self.network.decoder.parameters()

        if stage == 'warmup_decoder':
            self.print_to_log_file("train decoder, lin warmup")
            optimizer = torch.optim.SGD(heads, self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
            lr_scheduler = Lin_incr_LRScheduler(optimizer, self.initial_lr, int(self.warmup_duration_decoder//2))
        elif stage == 'train_decoder':
            self.print_to_log_file("train decoder, poly lr")
            if self.training_stage == 'warmup_decoder':
                # we can keep the existing optimizer and don't need to create a new one. This will allow us to keep
                # the accumulated momentum terms which already point in a useful driection
                optimizer = self.optimizer
            else:
                optimizer = torch.optim.SGD(heads, self.initial_lr, weight_decay=self.weight_decay,
                                            momentum=0.99, nesterov=True)
            lr_scheduler = PolyLRScheduler_offset(optimizer, self.initial_lr, self.warmup_duration_decoder, int(self.warmup_duration_decoder//2))

        elif stage == 'warmup_all':
            self.print_to_log_file("train whole net, warmup")
            optimizer = torch.optim.SGD(params, self.initial_lr, weight_decay=self.weight_decay,
                                        momentum=0.99, nesterov=True)
            lr_scheduler = Lin_incr_offset_LRScheduler(optimizer, self.initial_lr, self.warmup_duration_decoder + self.warmup_duration_whole_net,  self.warmup_duration_decoder)
            self.print_to_log_file(f"Initialized warmup_all optimizer and lr_scheduler at epoch {self.current_epoch}")
        else:
            self.print_to_log_file("train whole net, default schedule")
            if self.training_stage == 'warmup_all':
                # we can keep the existing optimizer and don't need to create a new one. This will allow us to keep
                # the accumulated momentum terms which already point in a useful driection
                optimizer = self.optimizer
            else:
                optimizer = torch.optim.SGD(params, self.initial_lr, weight_decay=self.weight_decay,
                                            momentum=0.99, nesterov=True)
            lr_scheduler = PolyLRScheduler_offset(optimizer, self.initial_lr, self.num_epochs, self.warmup_duration_whole_net + self.warmup_duration_decoder)
            self.print_to_log_file(f"Initialized train optimizer and lr_scheduler at epoch {self.current_epoch}")
        self.training_stage = stage
        empty_cache(self.device)
        return optimizer, lr_scheduler

class PretrainedTrainer_sawtooth_150ep(PretrainedTrainer_sawtooth):
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
        self.initial_lr = 1e-3
        self.warmup_duration_whole_net = 15  # lin increase whole network
        self.num_epochs = 150
        self.training_stage = None

class PretrainedTrainer_Primus_sawtooth(PretrainedTrainer_Primus):

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
        self.initial_lr = 1e-4
        self.warmup_lr_factor = 0.01 #during decoder warmup lr must be smaller otherwise training collaps
        self.weight_decay = 5e-2
        self.warmup_duration_decoder = 50
        self.warmup_duration_whole_net = 50  # lin increase whole network
        self.num_epochs = 1000

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
        else:
            params = self.network.parameters()
            # print(self.network.state_dict().keys())
            heads = self.network.up_projection.parameters()

        if stage == 'warmup_decoder':
            self.print_to_log_file("train decoder, lin warmup")
            optimizer =  torch.optim.AdamW(heads, self.initial_lr, weight_decay=self.weight_decay,
                                           amsgrad=False, betas=(0.9, 0.98), fused=True)
            lr_scheduler = Lin_incr_LRScheduler(optimizer, self.initial_lr*self.warmup_lr_factor, int(self.warmup_duration_decoder//2))
            self.print_to_log_file(f"Initialized warmup  only decoder optimizer and lr_scheduler at epoch {self.current_epoch}")
        elif stage == 'train_decoder':
            self.print_to_log_file("train decoder, poly lr")
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


class PretrainedTrainer_Primus_sawtooth_150ep(PretrainedTrainer_Primus_sawtooth):

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
        self.initial_lr = 1e-4
        self.warmup_lr_factor = 0.01 #during decoder warmup lr must be smaller otherwise training collaps
        self.weight_decay = 5e-2
        self.warmup_duration_decoder = 15
        self.warmup_duration_whole_net = 50  # lin increase whole network
        self.num_epochs = 150