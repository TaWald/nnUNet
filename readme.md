# Introducing Primus: A pure Transformer-based 3D semantic segmentation architecture
Primus is a pure Transformer-based 3D semantic segmentation architecture. It is the first of its kind and is designed to be a strong baseline for 3D semantic segmentation tasks. Primus is built on top of the nnU-Net framework and is fully compatible with it. It is designed to be easy to use and can be trained on any 3D dataset with minimal effort. Primus is a part of the nnU-Net framework and is available as a new configuration in the nnU-Net.


## Primus Trainers

Current Primus trainers are implemented and are named.
```
nnUNet_Primus_S_Trainer
nnUNet_Primus_B_Trainer
nnUNet_Primus_M_Trainer
nnUNet_Primus_L_Trainer
```
To run a training simply call

```bash
nnUNetv2_train <dataset_id> <config> <fold>`
```
Be sure to plan_and_preprocess as you would do for the ResEnc U-Net in nnU-Net.
