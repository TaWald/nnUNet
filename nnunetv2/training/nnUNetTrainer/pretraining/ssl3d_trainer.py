from nnunetv2.training.nnUNetTrainer.pretraining.pretrainedTrainer import PretrainedTrainer_Primus, PretrainedTrainer
import torch

class PretrainedTrainer_resencL_150ep(PretrainedTrainer):

    def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            use_pretrained_weights: bool = True,
            device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Can be overriden to train same architecture from scratch.
        self.initial_lr = 1e-3
        self.enable_deep_supervision = True
        self.warmup_duration_whole_net = 15  # lin increase whole network

class PretrainedTrainer_Primus_150ep(PretrainedTrainer_Primus):

    def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            use_pretrained_weights: bool = True,
            device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Can be overriden to train same architecture from scratch.
        self.initial_lr = 1e-4
        self.warmup_duration_whole_net = 15  # lin increase whole network
        self.num_epochs = 150 # lin increase whole network



