import argparse
import os
from copy import deepcopy
from pathlib import Path

import torch
from batchgenerators.utilities.file_and_folder_operations import join, save_json
from huggingface_hub import hf_hub_download

from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


def plan_nnssl(
    dataset_id: int,
    pt_name: str,
    pt_ckpt_path: str,
    scratch_identifier: str | None,
):
    dataset_name = convert_id_to_dataset_name(dataset_id)
    print(f"Planning nnssl for: {dataset_name}")
    # Adapt from the previous train-from-scratch plans
    if scratch_identifier is None:
        # Automatically search for the first scratchPlans__xxxx.json
        existing_scratch_plans_file = [f for f in os.listdir(join(nnUNet_preprocessed, dataset_name)) if f.startswith("scratchPlans__") and f.endswith(".json")]
        scratch_identifier = existing_scratch_plans_file[0].replace(".json", "")
        scratch_plans_file = join(nnUNet_preprocessed, dataset_name, scratch_identifier + ".json")
        print(f"Using auto-detected scratch plans file: {scratch_plans_file}")
    else:
        scratch_plans_file = join(nnUNet_preprocessed, dataset_name, scratch_identifier + ".json")
    scratch_plans_manager = PlansManager(scratch_plans_file)

    # -------------------------------- Adapt plan -------------------------------- #
    loaded_pt_ckpt = torch.load(pt_ckpt_path, weights_only=True, map_location="cpu")
    loaded_adapt_plan = loaded_pt_ckpt["nnssl_adaptation_plan"]
    adapted_plans_manager = deepcopy(scratch_plans_manager)
    adapted_config_manager = adapted_plans_manager.get_configuration("3d_fullres")
    _key = list(loaded_adapt_plan["pretrain_plan"]["configurations"].keys())[0]  # For pt_used_patchsize
    pretrain_info = {
        "checkpoint_path": pt_ckpt_path,
        "checkpoint_name": pt_name,
        "key_to_encoder": loaded_adapt_plan["key_to_encoder"],
        "key_to_stem": loaded_adapt_plan["key_to_stem"],
        "keys_to_in_proj": loaded_adapt_plan["keys_to_in_proj"],
        "key_to_lpe": loaded_adapt_plan["key_to_lpe"],
        "pt_num_in_channels": loaded_adapt_plan["pretrain_num_input_channels"],
        "pt_used_patchsize": loaded_adapt_plan["pretrain_plan"]["configurations"][_key]["patch_size"],
        "citations": loaded_pt_ckpt["citations"] if "citations" in loaded_pt_ckpt else [],
    }
    # Save important info for pretraining
    adapted_plans_manager.plans["pretrain_info"] = pretrain_info

    # Set network architecture
    architecture: dict = loaded_adapt_plan["architecture_plans"]
    architecture["_kw_requires_import"] = architecture.pop("arch_kwargs_requiring_import")
    architecture["network_class_name"] = architecture.pop("arch_class_name")
    adapted_config_manager.configuration["architecture"] = architecture

    adapted_plans_manager.plans["configurations"] = {"3d_fullres": adapted_config_manager.configuration}

    # Save plans json
    plans_name = f"ptPlans__{pt_name}__{scratch_identifier.replace("scratchPlans__", "")}"
    adapted_plans_manager.plans["plans_name"] = plans_name
    save_json(adapted_plans_manager.plans, join(nnUNet_preprocessed, dataset_name, plans_name + ".json"))


def maybe_download_pretrained_weights(pretrained_checkpoint_path: str):
    """
    Check if the pretrained checkpoint path points to a hugging face directory.
    If it does, check the repository has an adaptation_plan.json file indicating compatibility with nnssl.
    If it exists, download the checkpoint and store it. Then replace the local path with the URL and continue as usual.
    :param pretrained_checkpoint_path: Path or URL to the pretrained checkpoint.
    """

    # Check if the path is a Hugging Face repository URL
    if pretrained_checkpoint_path.startswith("https://huggingface.co/"):
        assert "nnssl_pretrained_models" in os.environ, "To allow auto-downloading weights you need to set the environment variable 'nnssl_pretrained_models' to the path where you want to store the pretrained models."
        local_dir = os.environ["nnssl_pretrained_models"]
        repo_id = pretrained_checkpoint_path.split("https://huggingface.co/")[-1].strip("/")

        final_path = os.path.join(local_dir, repo_id.replace("/", "_"))
        if not os.path.exists(final_path):
            os.makedirs(final_path, exist_ok=True)
        # Check if the repository contains the adaptation_plan.json file
        try:
            _ = hf_hub_download(repo_id=repo_id, filename="adaptation_plan.json", local_dir=final_path)
        except Exception as e:
            raise ValueError(f"The repository {repo_id} does not contain an adaptation_plan.json file.\n" "This indicates that the checkpoint is not compatible with this fine-tuning workflow." f"Error: {e}")

        # Download the checkpoint file
        try:
            checkpoint_path = hf_hub_download(repo_id=repo_id, filename="checkpoint_final.pth", local_dir=final_path)
            # For download tracking purposes
            _ = hf_hub_download(repo_id=repo_id, filename="config.json", local_dir=final_path)
        except Exception as e:
            raise ValueError(f"Failed to download the checkpoint file from the repository {repo_id}. " f"Error: {e}")

        # Replace the local path with the downloaded checkpoint path
        pretrained_checkpoint_path = checkpoint_path

    # Verify the local path exists
    if not Path(pretrained_checkpoint_path).is_file():
        raise FileNotFoundError(f"The pretrained checkpoint path {pretrained_checkpoint_path} does not exist.")
    return pretrained_checkpoint_path


def plan_nnssl_entrypoint():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset_id",
        type=int,
        required=True,
        help="[REQUIRED] Dataset ID.",
    )
    parser.add_argument(
        "-ptn",
        "--pretrain_name",
        type=str,
        required=True,
        help="[REQUIRED] !Unique! name of the pretraining. Used to name the plans file." "Could look something like this 'CNN_MAE_XY_Dataset_Z'.",
    )
    parser.add_argument(
        "-pc",
        "--pretrained_checkpoint_path",
        type=str,
        required=True,
        help="[REQUIRED] Path to the pretrained ckpt`<CKPT>.pth` of a pre-training done with nnssl.",
    )
    parser.add_argument(
        "-si",
        "--scratch_identifier",
        type=str,
        required=False,
        help="[OPTIONAL] Base name of the train-from-scratch plans json. If not provided, will automatically search for the first scratchPlans__xxxx.json",
    )
    args = parser.parse_args()

    dataset_id: int = args.dataset_id
    pretrain_name: str = args.pretrain_name
    pretrained_checkpoint_path: str = args.pretrained_checkpoint_path
    scratch_identifier: str | None = args.scratch_identifier

    pretrained_checkpoint_path: str = maybe_download_pretrained_weights(pretrained_checkpoint_path)

    plan_nnssl(
        dataset_id=dataset_id,
        pt_name=pretrain_name,
        pt_ckpt_path=pretrained_checkpoint_path,
        scratch_identifier=scratch_identifier,
    )


if __name__ == "__main__":
    plan_nnssl_entrypoint()
