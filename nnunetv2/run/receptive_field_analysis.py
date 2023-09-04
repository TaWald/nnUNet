import numpy as np
import nnunetv2
import torch.cuda
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.run.output_receptive_field import pseudo_receptive_field
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class


def get_trainer_from_args(
    dataset_name_or_id: int | str,
    configuration: str,
    fold: int,
    trainer_name: str = "nnUNetTrainer",
    plans_identifier: str = "nnUNetPlans",
    use_compressed: bool = False,
    device: torch.device = torch.device("cuda"),
) -> nnUNetTrainer:
    # load nnunet class and do sanity checks
    nnunet_trainer_class: type[nnUNetTrainer] | None = recursive_find_python_class(
        join(nnunetv2.__path__[0], "training", "nnUNetTrainer"), trainer_name, "nnunetv2.training.nnUNetTrainer"
    )
    if nnunet_trainer_class is None:
        raise RuntimeError(
            f"Could not find requested nnunet trainer {trainer_name} in "
            f"nnunetv2.training.nnUNetTrainer ("
            f'{join(nnunetv2.__path__[0], "training", "nnUNetTrainer")}). If it is located somewhere '
            f"else, please move it there."
        )
    assert issubclass(nnunet_trainer_class, nnUNetTrainer), (
        "The requested nnunet trainer class must inherit from " "nnUNetTrainer"
    )

    # initialize nnunet trainer
    preprocessed_dataset_folder_base = join(nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_name_or_id))
    plans_file = join(preprocessed_dataset_folder_base, plans_identifier + ".json")
    plans = load_json(plans_file)
    dataset_json = load_json(join(preprocessed_dataset_folder_base, "dataset.json"))
    nnunet_trainer = nnunet_trainer_class(
        plans=plans,
        configuration=configuration,
        fold=fold,
        dataset_json=dataset_json,
        unpack_dataset=not use_compressed,
        device=device,
    )
    return nnunet_trainer


def calc_receptive_field(
    dataset_name_or_id: str | int,
    configuration: str,
    plans_identifier: str = "nnUNetPlans",
    device: torch.device = torch.device("cuda"),
):
    fold = 0

    nnunet_trainer = get_trainer_from_args(
        dataset_name_or_id=dataset_name_or_id,
        configuration=configuration,
        fold=fold,
        plans_identifier=plans_identifier,
        device=device,
    )
    nnunet_trainer.initialize()
    network = nnunet_trainer.network
    assert network is not None, "Network must be initialized before calculating the receptive field!"
    patch_size = nnunet_trainer.configuration_manager.patch_size
    patch_size = patch_size

    receptive_result = pseudo_receptive_field(network, patch_size)
    print(receptive_result)


def receptive_field_analysis():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name_or_id", type=str, help="Dataset name or ID to train with")
    parser.add_argument("configuration", type=str, help="Configuration that should be trained")
    parser.add_argument(
        "-p",
        type=str,
        required=False,
        default="nnUNetPlans",
        help="[OPTIONAL] Use this flag to specify a custom plans identifier. Default: nnUNetPlans",
    )
    parser.add_argument(
        "-device",
        type=str,
        default="cuda",
        required=False,
        help="Use this to set the device the inference should run with. Available options are 'cuda' "
        "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
        "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_predict [...] instead!",
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

    calc_receptive_field(
        dataset_name_or_id=args.dataset_name_or_id,
        configuration=args.configuration,
        plans_identifier=args.p,
        device=device,
    )


if __name__ == "__main__":
    receptive_field_analysis()
