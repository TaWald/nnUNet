import argparse

from batchgenerators.utilities.file_and_folder_operations import isfile, join, save_json

from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.preprocessing.preprocessors.default_preprocessor import (
    DefaultPreprocessor,
)
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


def plan_and_preprocess_scratch(
    dataset_id: int,
    network_name: str,
    override_spacing: tuple[float, float, float] | None,
    override_normalization: str | None,
    override_patchsize: tuple[int, int, int] | None,
    override_batchsize: int | None,
    no_pp: bool,
    num_processes: int,
    verbose: bool,
):
    dataset_name = convert_id_to_dataset_name(dataset_id)
    print(f"Planning scratch for: {dataset_name}")
    # Check if nnUNetPLans exists or ResEncL/M/S or whatever
    potential_plans = ["nnUNetPlans.json"] + [f"nnUNetResEncUNet{k}Plans.json" for k in "_M_L_XL".split("_")]
    existing_plans = [p for p in potential_plans if isfile(join(nnUNet_preprocessed, dataset_name, p))]
    assert len(existing_plans) > 0, f"Could not find any plans file for dataset {dataset_name}. Please plan the dataset normally first to do preprocessing from pretrained. FYI, we check for {potential_plans}"
    plans_file = join(nnUNet_preprocessed, dataset_name, existing_plans[0])
    print(f"Using auto-detected default plans file: {plans_file}")
    plans_manager = PlansManager(plans_file)
    config_manager = plans_manager.get_configuration("3d_fullres")  # Just focus on 3d_fullres

    # Override spacing
    if override_spacing is not None:
        config_manager.configuration["spacing"] = override_spacing

    # Override normalization schemes
    if override_normalization is not None:
        short_norm_name = "Z" if override_normalization == "ZScoreNormalization" else override_normalization.replace("Normalization", "")
        new_normalization_schemes = [override_normalization for _ in config_manager.configuration["normalization_schemes"]]
        config_manager.configuration["normalization_schemes"] = new_normalization_schemes

    # Override patch size
    if override_patchsize is not None:
        config_manager.configuration["patch_size"] = override_patchsize

    # Override batch size
    if override_batchsize is not None:
        config_manager.configuration["batch_size"] = override_batchsize

    # Set data identifier
    spacing_format = "Spacing_{}_{}_{}".format(*[f"{x:.2f}" if x is not None else "None" for x in config_manager.configuration["spacing"]])
    norm_scheme_format = "Norm_" + "_".join([short_norm_name for _ in new_normalization_schemes])
    data_identifier = spacing_format + "__" + norm_scheme_format  # Data identity only controlled by spacing and normalization
    config_manager.configuration["data_identifier"] = data_identifier

    # Set network architecture
    architecture = {}
    architecture["_kw_requires_import"] = None
    architecture["arch_kwargs"] = None
    architecture["network_class_name"] = network_name
    config_manager.configuration["architecture"] = architecture

    # Set an empty pretrain info to indicate training from scratch
    plans_manager.plans["pretrain_info"] = {}

    # Needs to be overriden (see below the code before preprocessor.run())
    config_manager.configuration["preprocessor_name"] = "DefaultPreprocessor"

    plans_manager.plans["configurations"] = {"3d_fullres": config_manager.configuration}

    # Save plans json
    patch_size_format = "PS_{}_{}_{}".format(*[f"{x}" if x is not None else "None" for x in config_manager.configuration["patch_size"]])
    batch_size_format = "BS_{}".format(config_manager.configuration["batch_size"])
    plans_name = f"scratchPlans__{network_name}__{data_identifier}__{patch_size_format}__{batch_size_format}"
    plans_manager.plans["plans_name"] = plans_name
    save_json(plans_manager.plans, join(nnUNet_preprocessed, dataset_name, plans_name + ".json"))

    if not no_pp:
        print(f"Preprocessing scratch for: {dataset_name}")
        preprocessor: DefaultPreprocessor = config_manager.preprocessor_class(verbose=verbose)
        preprocessor.run(dataset_id, "3d_fullres", plans_name, num_processes=num_processes, overwrite_existing=False)


def plan_and_preprocess_scratch_entrypoint():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset_id",
        type=int,
        required=True,
        help="[REQUIRED] Dataset ID.",
    )
    parser.add_argument(
        "-nn",
        "--network_name",
        type=str,
        required=True,
        help="[REQUIRED] Name of the network architecture, e.g., ResEncL, PrimusM.",
    )
    parser.add_argument(
        "-os",
        "--override_spacing",
        nargs="+",
        type=float,
        required=False,
        help="[OPTIONAL] Override spacing, e.g., 1.0 1.0 1.0",
    )
    parser.add_argument(
        "-on",
        "--override_normalization",
        type=str,
        required=False,
        help="[OPTIONAL] Override normalization scheme, e.g., ZScoreNormalization",
    )
    parser.add_argument(
        "-op",
        "--override_patchsize",
        nargs="+",
        type=int,
        required=False,
        help="[OPTIONAL] Override patch size, e.g., 128 128 128",
    )
    parser.add_argument(
        "-ob",
        "--override_batchsize",
        type=int,
        required=False,
        help="[OPTIONAL] Override batch size, e.g., 2",
    )
    parser.add_argument(
        "--no_pp",
        action="store_true",
        required=False,
        help="[OPTIONAL] Only do planning but no preprocessing.",
    )
    parser.add_argument(
        "-np",
        "--num_processes",
        default=4,
        type=int,
        required=False,
        help="[OPTIONAL] Number of processes to use for preprocessing. Default: 4",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        required=False,
        help="[OPTIONAL] Set this to print a lot of stuff. Useful for debugging. Will disable progress bar!",
    )
    args = parser.parse_args()

    dataset_id: int = args.dataset_id
    network_name: str = args.network_name
    override_spacing: tuple[float, float, float] | None = args.override_spacing
    override_normalization: str | None = args.override_normalization
    override_patchsize: tuple[int, int, int] | None = args.override_patchsize
    override_batchsize: int | None = args.override_batchsize
    no_pp: bool = args.no_pp
    num_processes: int = args.num_processes
    verbose: bool = args.verbose

    plan_and_preprocess_scratch(
        dataset_id=dataset_id,
        network_name=network_name,
        override_spacing=override_spacing,
        override_normalization=override_normalization,
        override_patchsize=override_patchsize,
        override_batchsize=override_batchsize,
        no_pp=no_pp,
        num_processes=num_processes,
        verbose=verbose,
    )


if __name__ == "__main__":
    plan_and_preprocess_scratch_entrypoint()
