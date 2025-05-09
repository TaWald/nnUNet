# Welcome to the nnssl pretraining segmentation adaptation framework!

>Authored by: Tassilo Wald & Constantin Ulrich

This is a temporary development branch that is intended to later be integrated into nnU-Net fully once proven stable enough. 
It is heavily dependent on the [nnssl](https://github.com/MIC-DKFZ/nnssl) framework and allows (hopefully) really simple adaptation of the pre-trained checkpoints from it.

## Why finetune pre-trained models?
Pre-training is a great way to improve performance of a model on a downstream dataset and can cut-down training time significantly. While the nnssl framework provides easy-to-use methods to pre-train models on a large dataset this repository provides the easy-to-use segmentation adaptation of these checkpoints.

## Getting Started:

<details>
<summary> <h3>Installation:</h3></summary>

To install this repository you need to follow these steps:

```bash
# Recommended: Create an environment with python 3.12
conda create -n nnunet_adaptation python=3.12
conda activate nnunet_adaptation
# Install this repository
git clone https://github.com/TaWald/nnUNet.git nnunet_adaptation
cd nnunet_adaptation
# Checkout this branch
git checkout nnssl_finetuning_inclusion
# Install the dependencies
pip install -e .
```

To check if this repository installed correctly you can run the following command:
```bash
nnUNetv2_preprocess_like_nnssl --help
nnUNetv2_train_pretrained --help
```
Both of these commands should exist and provide you with a help message in the command line interface.
</details>


### Pre-requisits:

To use this repository you need to:
1. Set-Up your environment variables [(See Setting up Paths)](documentation/setting_up_paths.md)
2. Have a raw dataset in nnU-Net format lying around somewhere [(See Dataset Format)](documentation/dataset_format.md)
3. Have a pre-trained nnssl checkpoint. [(See nnssl documentation)](https://github.com/MIC-DKFZ/nnssl)

## How to use fine-tune a pre-trained model?
Given that you have a raw dataset in nnU-Net format and a pre-trained nnssl checkpoint, you need to do two things:
1. Preprocess the dataset using the `nnUNetv2_preprocess_like_nnssl` command.
2. Train the model using the `nnUNetv2_train_pretrained` command.
```bash
# Do fingerprint extraction and planning as nnU-net would
nnUNetv2_plan_and_preprocess -d <dataset_identifier> --no_pp
# Preprocess the dataset like nnssl would
nnUNetv2_preprocess_like_nnssl -d <dataset_identifier> -n <UniqueNameOfTraining> -pc <Path/to/the/pretrained/checkpoint.pth> -am "like_pretrained"
```
*To easily fine-tune a model you can use any [nnssl pre-trained model from hugging face](https://huggingface.co/collections/MIC-DKFZ/openmind-models-6819c21c7fe6f0aaaab7dadf) e.g. the MAE pre-trained ResEnc-L model. Simply pass a Hugging Face URL to `-pc` https://huggingface.co/AnonRes/ResEncL-OpenMind-MAE to automatically download the model and use it for fine-tuning.*

This command first creates the fingerprint and default nnU-Net plans for the dataset. 
Then the new `nnUNetv2_preprocess_like_nnssl` uses the pre-existing plan as well as your pre-trained checkpoint to automatically pre-process the dataset in the same way as the pre-training did. 
The specific preprocessing can be different through the `-am`/`--adaptation_mode` argument, which changes how the spacing of the downstream dataset is set. We mostly recommend `like_pretrained` which is mostly 1x1x1 mm isotropic, however this may be a bad choice for datasets with much smaller spacings, for which we recommend using `adaptation_mode` which sets it to the datasets median spacing. 

When running this command you will see a progress bar for the preprocessing allowing you to observe the progress of the preprocessing.

Once this concluded you should see a new dataset folder in your `nnUNet_preprocessed` folder and a new preprocessed plan that start with `ptPlans__<UniqueNameOfTraining>...`. This plan contains all the information about which pre-trained weights to use and which architecture to initialize and how to load the pre-trained weights into the downstream architecture, so you can forget about the checkpoint path here!

Now you can start fine-tuning your pre-trained model using the `nnUNetv2_train_pretrained` command:
```bash
nnUNetv2_train_pretrained -d <dataset_identifier> -c "3d_fullres" -f <Pick_fold_0-4> -p <NameOfYourNewlyCreatedPlans>
```
This should start the training of your model with the default `PretrainedTrainer` which follows a simple warm-up schedule.


# Acknowledgements
<img src="documentation/assets/HI_Logo.png" height="100px" />

<img src="documentation/assets/dkfz_logo.png" height="100px" />

This nnssl adaptation branch is developed and maintained by the [Division of Medical Image Computing](https://www.dkfz.de/en/mic/index.php) at the [German Cancer Research Center (DKFZ)](https://www.dkfz.de/en/index.html) and is part of the Helmholtz Foundation Model Initiative (HFMI) project of The Human Radiome Project [THRP](https://hfmi.helmholtz.de/pilot-projects/thrp/)
