from copy import deepcopy
from typing import Literal, get_args
import torch
import numpy as np

# from numpy.core.multiarray

from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name
from batchgenerators.utilities.file_and_folder_operations import join, load_json, save_json, isfile
import argparse


ADAPTATION_MODES = Literal["fixed", "default_nnunet", "no_resample", "like_pretrained"]


def new_spacing_from_mode(
    adaptation_mode: ADAPTATION_MODES,
    original_spacing: tuple[float, float, float],
    pretrain_spacing: tuple[float, float, float],
    override_spacing: list[float] | None,
):
    """
    Determine the new spacing based on the `adaptation_mode` chosen by the user.
    This function adjusts the spacing of the data according to the specified adaptation mode.
    The adaptation mode dictates whether to use the original spacing, a fixed override spacing, or to skip resampling all-together, allowing differing spacings.
    :param adaptation_mode: The mode of adaptation to apply. Must be one of the values defined in `ADAPTATION_MODES`.
                            Options include:
                            - "default_nnunet": Use the original spacing without modification.
                            - "fixed": Use the spacing provided in `override_spacing`.
                            - "no_resample": Skip resampling and return `None`.
    :param original_spacing: A tuple of three floats representing the original spacing of the data (e.g., (x, y, z)).
    :param override_spacing: A list of three floats representing the new spacing to use when `adaptation_mode` is "fixed".
                             If `None`, this parameter is ignored unless the mode is "fixed".
    :return:
        - If `adaptation_mode` is "default_nnunet", returns the `original_spacing`.
        - If `adaptation_mode` is "fixed", returns the `override_spacing`.
        - If `adaptation_mode` is "no_resample", returns `None`.

    :raises ValueError: If an invalid `adaptation_mode` is provided, an exception is raised with a message listing valid options.
    """
    if adaptation_mode == "default_nnunet":
        return original_spacing
    elif adaptation_mode == "fixed":
        if override_spacing is None:
            raise ValueError("You need to provide a spacing when using the fixed adaptation mode.")
        if len(override_spacing) != 3:
            raise ValueError("The override spacing must be a list of three floats.")
        if not all(isinstance(i, (int, float)) for i in override_spacing):
            raise ValueError("The override spacing must be a list of three floats.")
        return override_spacing
    elif adaptation_mode == "no_resample":
        return [None, None, None]
    elif adaptation_mode == "like_pretrained":
        return pretrain_spacing
    else:
        raise ValueError(
            f"Unepexpected adaptation mode passed: {adaptation_mode}. Choose from {get_args(ADAPTATION_MODES)}"
        )


def get_new_normalization_format(original_config: ConfigurationManager) -> list[str]:
    """Sets the normalization scheme for all channels to ZScoreNormalization.
    Temporary Solution until there is a better way to do this.

    :param original_config: The original configuration manager."""
    norm_scheme = original_config.normalization_schemes
    # ToDo: Allow override for CT-Score normalization (e.g. override spacing mode & use default nnU-Net norm)
    return ["ZScoreNormalization" for _ in norm_scheme]


def preprocess_like_nnssl(
    dataset_id: int,
    pretrain_name: str,
    pt_ckpt: str,
    adaptation_mode: str,
    override_spacing: list[float] | None,
    num_processes: int,
    verbose: bool,
):
    """
    Preprocess a dataset relative to pre-training from nnssl.
    :param dataset: Dataset ID
    :param adaptation_mode: Adaptation mode for the downstream architecture
    :param override_spacing: Override spacing for the fixed adaptation mode
    :param num_processes: Number of processes to use
    """
    dataset_name = convert_id_to_dataset_name(dataset_id)
    print(f"Preprocessing dataset {dataset_name}")
    # Check if nnUNetPLans exists or ResEncL/M/S or whatever
    potential_plans = ["nnUNetPlans.json"] + [f"nnUNetResEncUNet{k}Plans.json" for k in "_M_L_XL".split("_")]
    existing_plans = [p for p in potential_plans if isfile(join(nnUNet_preprocessed, dataset_name, p))]
    assert len(existing_plans) > 0, (
        f"Could not find any plans file for dataset {dataset_name}."
        + "Please plan the dataset normally first to do preprocessing from pretrained."
        + f"FYI, we check for {potential_plans}"
    )
    downstream_plans_file = join(nnUNet_preprocessed, dataset_name, existing_plans[0])
    downstream_plans_manager = PlansManager(downstream_plans_file)
    downstream_config: ConfigurationManager = downstream_plans_manager.get_configuration("3d_fullres")
    fullres_spacing = downstream_config.spacing

    loaded_pretrain_ckpt = torch.load(pt_ckpt, weights_only=True)
    adaptation_plan = loaded_pretrain_ckpt["nnssl_adaptation_plan"]
    pretrain_config = list(adaptation_plan["pretrain_plan"]["configurations"].values())[0]
    pretrain_spacing = pretrain_config["spacing"]
    architecture_details: dict = adaptation_plan["architecture_plans"]
    architecture_details["network_class_name"] = architecture_details.pop("arch_class_name")
    architecture_details["_kw_requires_import"] = architecture_details.pop("arch_kwargs_requiring_import")

    # -------------------------------- Adapt plan -------------------------------- #
    adapted_plans = deepcopy(downstream_plans_manager)
    config = list(adaptation_plan["pretrain_plan"]["configurations"].keys())[0]
    pretrain_info = {
        "checkpoint_path": pt_ckpt,
        "checkpoint_name": pretrain_name,
        "key_to_encoder": adaptation_plan["key_to_encoder"],
        "key_to_stem": adaptation_plan["key_to_stem"],
        "keys_to_in_proj": adaptation_plan["keys_to_in_proj"],
        "key_to_lpe": adaptation_plan["key_to_lpe"],
        "pt_num_in_channels": adaptation_plan["pretrain_num_input_channels"],
        "pt_used_patchsize": adaptation_plan["pretrain_plan"]["configurations"][config]["patch_size"],
        "pt_recommended_downstream_patchsize": adaptation_plan["recommended_downstream_patchsize"],
        # ToDo (Maybe): Add pretrain patch size here instead of overwriting?
        #   Also might make sense to add info on the recommended patch size for downstream training.
        #   Also could help to give info if e.g. pos. Embedding should be re-initialized or not.
    }
    # Save important info for pretraining
    adapted_plans.plans["pretrain_info"] = pretrain_info
    new_spacing = new_spacing_from_mode(
        adaptation_mode,
        original_spacing=fullres_spacing,
        pretrain_spacing=pretrain_spacing,
        override_spacing=override_spacing,
    )
    # Set spacing
    adapted_config = adapted_plans.get_configuration("3d_fullres")
    adapted_config.configuration["spacing"] = new_spacing
    # Set normalization schemes
    new_normalization_schemes = get_new_normalization_format(adapted_config)
    adapted_config.configuration["normalization_schemes"] = new_normalization_schemes
    # Set Data Identifier
    spacing_format = "Spacing__{}_{}_{}".format(*[f"{x:.2f}" if x is not None else "None" for x in new_spacing])
    norm_scheme_identifier = f"Norm__" + "_".join(["Z" for _ in new_normalization_schemes])
    data_identifier = spacing_format + "___" + norm_scheme_identifier
    adapted_config.configuration["data_identifier"] = data_identifier  # Set the data identifier clearly.
    # Needs to be overriden to be found lower down to call the correct preprocessor.
    adapted_config.configuration["preprocessor_name"] = "DefaultPreprocessor"
    adapted_config.configuration["architecture"] = architecture_details
    adapted_config.configuration["patch_size"] = adaptation_plan[
        "recommended_downstream_patchsize"
    ]  # Overwrite patch size with pre-training patch size.
    adapted_plans.plans["configurations"] = {"3d_fullres": adapted_config.configuration}

    plans_name = f"ptPlans__{pretrain_name}____{data_identifier}"
    adapted_plans.plans["plans_name"] = plans_name
    save_json(adapted_plans.plans, join(nnUNet_preprocessed, dataset_name, plans_name + ".json"))

    preprocessor: DefaultPreprocessor = adapted_config.preprocessor_class(verbose=verbose)
    # We never overwrite existing currently, as multiple preprocessors can share the same data files.
    #     Otherwise stuff might break.
    preprocessor.run(dataset_id, "3d_fullres", plans_name, num_processes=num_processes, overwrite_existing=False)


def preprocess_like_nnssl_entrypoint():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        type=int,
        help="[REQUIRED] List of dataset IDs. Example: 2 4 5. This will run fingerprint extraction, experiment "
        "planning and preprocessing for these datasets. Can of course also be just one dataset",
    )
    parser.add_argument(
        "-n",
        "--pretraining_name",
        type=str,
        required=True,
        help="[REQUIRED] !Unique! name of the pretraining. Used to name the plans file."
        "Could look something like this 'CNN_MAE_XY_Dataset_Z'.",
    )
    parser.add_argument(
        "-pc",
        "--pretrained_checkpoint",
        type=str,
        help="[REQUIRED] Path to the pretrained ckpt`<CKPT>.pth` of a pre-training done with nnssl.",
        required=True,
    )
    parser.add_argument(
        "-am",
        "--adaptation_mode",
        default="default_nnunet",
        required=False,
        choices=get_args(ADAPTATION_MODES),
        help="[OPTIONAL] You can specify how to preprocess the downstream architecture given the pretrained config.",
    )
    parser.add_argument(
        "-os",
        "--override_spacing",
        required=False,
        nargs="+",
        help="[OPTIONAL] When choosing the fixed adaptation mode, you need to specify it and provide the target spacing of the data.",
    )
    parser.add_argument(
        "-np",
        type=int,
        default=4,
        required=False,
        help="[OPTIONAL] Use this to define how many processes are to be used. If this is just one number then "
        "this number of processes is used for all configurations specified with -c. If it's a "
        "list of numbers this list must have as many elements as there are configurations. We "
        "then iterate over zip(configs, num_processes) to determine then umber of processes "
        "used for each configuration. More processes is always faster (up to the number of "
        "threads your PC can support, so 8 for a 4 core CPU with hyperthreading. If you don't "
        "know what that is then dont touch it, or at least don't increase it!). DANGER: More "
        "often than not the number of processes that can be used is limited by the amount of "
        "RAM available. Image resampling takes up a lot of RAM. MONITOR RAM USAGE AND "
        "DECREASE -np IF YOUR RAM FILLS UP TOO MUCH!. Default: 8 processes for 2d, 4 "
        "for 3d_fullres, 8 for 3d_lowres and 4 for everything else",
    )
    parser.add_argument(
        "--verbose",
        required=False,
        action="store_true",
        help="Set this to print a lot of stuff. Useful for debugging. Will disable progress bar! "
        "Recommended for cluster environments",
    )
    args = parser.parse_args()

    dataset_ids: int = args.d
    pretrain_name: str = args.pretraining_name
    pretrained_checkpoint_path: str = args.pretrained_checkpoint
    adaptation_mode: str = args.adaptation_mode
    override_spacing: tuple[float, float, float] | None = args.override_spacing
    num_processes: int = args.np
    verbose: bool = args.verbose

    preprocess_like_nnssl(
        dataset_id=dataset_ids,
        pretrain_name=pretrain_name,
        pt_ckpt=pretrained_checkpoint_path,
        adaptation_mode=adaptation_mode,
        override_spacing=override_spacing,
        num_processes=num_processes,
        verbose=verbose,
    )


if __name__ == "__main__":
    preprocess_like_nnssl_entrypoint()
