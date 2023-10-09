import numpy as np
import nnunetv2
import torch.cuda
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.run.receptive_field_analysis import (
    create_input,
    init_architecture,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from torch import nn


def register_shape_hooks(changed_model: nn.Module) -> tuple[dict, list[torch.utils.hooks.RemovableHandle]]:
    handles = []
    module_size_dict: dict[str, dict] = {}
    for name, module in changed_model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.LazyConv1d, nn.LazyConv2d, nn.LazyConv3d)):
            handles.append(module.register_forward_hook(_output_shape_hook(module_size_dict, name)))
        # make a forward pass
    return module_size_dict, handles


def _output_shape_hook(
    savings_dict: dict,
    name: str,
):
    """Calculates the minimum and maximum indices that are non-zero in the output of a module.
    Saves that with the name in the savings dict."""

    def hook(module, input, output):
        # Shape of the output is up to: [batch, channels, z, y, x]

        # ----- Calculate the minimum and maximum indices that are non-zero in the output of a module -----
        # Now collapse all the not interesting dimensions

        savings_dict[name] = tuple(output.shape)
        return

    return hook


def shape_analysis(model: nn.Module, input_size, device="cuda"):
    init_architecture(model)

    output_shape_dict: dict[str, dict]
    handles: list[torch.utils.hooks.RemovableHandle]

    output_shape_dict, handles = register_shape_hooks(model)
    with torch.no_grad():
        x = create_input(input_size, device)
        model(x)

    for h in handles:
        h.remove()

    print(model)

    for name, mets in output_shape_dict.items():
        print(name, mets)


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


def calc_output_shape(
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

    receptive_result = shape_analysis(network, patch_size)
    print(receptive_result)


def output_shape_anaylsis():
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

    calc_output_shape(
        dataset_name_or_id=args.dataset_name_or_id,
        configuration=args.configuration,
        plans_identifier=args.p,
        device=device,
    )


if __name__ == "__main__":
    output_shape_anaylsis()
