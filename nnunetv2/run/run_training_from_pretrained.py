import argparse
from typing import Union
import torch
import os
import multiprocessing as mp

from nnunetv2.run.run_training import find_free_network_port, get_trainer_from_args, maybe_load_checkpoint, run_ddp
from torch.backends import cudnn
from batchgenerators.utilities.file_and_folder_operations import join

from nnunetv2.training.nnUNetTrainer.pretraining.pretrainedTrainer import PretrainedTrainer


def train_pretrained(
    dataset_name_or_id: Union[str, int],
    configuration: str,
    fold: Union[int, str],
    trainer_class_name: str,
    plans_identifier: str,
    from_scratch: bool,
    num_gpus: int = 1,
    export_validation_probabilities: bool = False,
    continue_training: bool = False,
    only_run_validation: bool = False,
    disable_checkpointing: bool = False,
    val_with_best: bool = False,
    device: torch.device = torch.device("cuda"),
):
    if isinstance(fold, str):
        if fold != "all":
            try:
                fold = int(fold)
            except ValueError as e:
                print(
                    f'Unable to convert given value for fold to int: {fold}. fold must bei either "all" or an integer!'
                )
                raise e

    if val_with_best:
        assert not disable_checkpointing, "--val_best is not compatible with --disable_checkpointing"

    if num_gpus > 1:
        assert (
            device.type == "cuda"
        ), f"DDP training (triggered by num_gpus > 1) is only implemented for cuda devices. Your device: {device}"

        os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ.keys():
            port = str(find_free_network_port())
            print(f"using port {port}")
            os.environ["MASTER_PORT"] = port  # str(port)

        mp.spawn(
            run_ddp,
            args=(
                dataset_name_or_id,
                configuration,
                fold,
                trainer_class_name,
                plans_identifier,
                disable_checkpointing,
                continue_training,
                only_run_validation,
                None,
                export_validation_probabilities,
                val_with_best,
                num_gpus,
                from_scratch,
            ),
            nprocs=num_gpus,
            join=True,
        )
    else:
        # ToDo
        nnunet_trainer: PretrainedTrainer = get_trainer_from_args(
            dataset_name_or_id,
            configuration,
            fold,
            trainer_class_name,
            plans_identifier,
            device=device,
            pretrained_from_scratch=from_scratch,  # <-- Creates new plan name if true. Allows easy comparison Pretrained vs Non-Pretrained
        )

        nnunet_trainer.use_pretrained_weights = False if continue_training else True

        if disable_checkpointing:
            nnunet_trainer.disable_checkpointing = disable_checkpointing

        assert not (
            continue_training and only_run_validation
        ), f"Cannot set --c and --val flag at the same time. Dummy."

        # Still needed to allow continuation of incomplete (i.e. interrupted) trainings.
        # ToDo: Find a way to not pre-load the same checkpoints that get overriden by the --continue flag.
        maybe_load_checkpoint(nnunet_trainer, continue_training, only_run_validation, None)

        if torch.cuda.is_available():
            cudnn.deterministic = False
            cudnn.benchmark = True

        if not only_run_validation:
            nnunet_trainer.run_training()

        if val_with_best:
            nnunet_trainer.load_checkpoint(join(nnunet_trainer.output_folder, "checkpoint_best.pth"))
        nnunet_trainer.perform_actual_validation(export_validation_probabilities)


def train_pretrained_entrypoint():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name_or_id", type=str, help="Dataset name or ID to train with")
    parser.add_argument("configuration", type=str, help="Configuration that should be trained")
    parser.add_argument(
        "fold", type=str, help="Fold of the 5-fold cross-validation.", choices=["0", "1", "2", "3", "4", "all"]
    )
    parser.add_argument(
        "-tr",
        type=str,
        required=False,
        default="PretrainedTrainer",
        help="[OPTIONAL] Use this flag to specify a custom trainer. Default: nnUNetTrainer",
    )
    parser.add_argument(
        "-p",
        type=str,
        required=True,
        help="[REQUIRED] Use this to specify a custom plans identifier.",
    )
    parser.add_argument(
        "--from_scratch",
        required=False,
        action="store_true",
        help="[OPTIONAL] flag to train from scratch with same config as when loading checkpoints.",
    )
    parser.add_argument(
        "-num_gpus", type=int, default=1, required=False, help="Specify the number of GPUs to use for training"
    )
    parser.add_argument(
        "--npz",
        action="store_true",
        required=False,
        help="[OPTIONAL] Save softmax predictions from final validation as npz files (in addition to predicted "
        "segmentations). Needed for finding the best ensemble.",
    )
    parser.add_argument(
        "--c", action="store_true", required=False, help="[OPTIONAL] Continue training from latest checkpoint"
    )
    parser.add_argument(
        "--val",
        action="store_true",
        required=False,
        help="[OPTIONAL] Set this flag to only run the validation. Requires training to have finished.",
    )
    parser.add_argument(
        "--val_best",
        action="store_true",
        required=False,
        help="[OPTIONAL] If set, the validation will be performed with the checkpoint_best instead "
        "of checkpoint_final. NOT COMPATIBLE with --disable_checkpointing! "
        "WARNING: This will use the same 'validation' folder as the regular validation "
        "with no way of distinguishing the two!",
    )
    parser.add_argument(
        "--disable_checkpointing",
        action="store_true",
        required=False,
        help="[OPTIONAL] Set this flag to disable checkpointing. Ideal for testing things out and "
        "you dont want to flood your hard drive with checkpoints.",
    )
    parser.add_argument(
        "-device",
        type=str,
        default="cuda",
        required=False,
        help="Use this to set the device the training should run with. Available options are 'cuda' "
        "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
        "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_train [...] instead!",
    )
    args = parser.parse_args()

    assert args.device in [
        "cpu",
        "cuda",
        "mps",
    ], f"-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}."
    if args.device == "cpu":
        # let's allow torch to use hella threads
        import multiprocessing

        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device("cpu")
    elif args.device == "cuda":
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device("cuda")
    else:
        device = torch.device("mps")

    train_pretrained(
        args.dataset_name_or_id,
        args.configuration,
        args.fold,
        args.tr,
        args.p,
        args.from_scratch,
        args.num_gpus,
        args.npz,
        args.c,
        args.val,
        args.disable_checkpointing,
        args.val_best,
        device=device,
    )


if __name__ == "__main__":
    train_pretrained_entrypoint()
