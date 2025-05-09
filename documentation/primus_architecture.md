# Introducing Primus: A pure Transformer-based 3D semantic segmentation architecture

[Primus](https://arxiv.org/abs/2503.01835) is a pure Transformer-based 3D semantic segmentation architecture. As opposed to previous 'transformer architectures' it is the first transformer architecture that actually depends on the attention paradigm to do 3D semantic segmentation. Primus is built on top of the nnU-Net framework and is fully compatible with it. It is designed to be easy to use and can be trained on any 3D dataset with minimal effort. Primus is a part of the nnU-Net framework and is available as a new configuration in the nnU-Net.

![Primus architecture](documentation/assets/transformed_primus.png)

## Primus Trainers

Current Primus trainers are implemented and are named.
```
nnUNet_Primus_S_Trainer
nnUNet_Primus_B_Trainer
nnUNet_Primus_M_Trainer
nnUNet_Primus_L_Trainer
```
To run a training of Primus M plan and preprocess as you would do for the ResEnc U-Net in nnU-Net.
Then call:
```bash
nnUNetv2_train <dataset_id> <config> <fold> -tr nnUNet_Primus_M_Trainer
```

----
*This is the nnU-Net inclusion of the original Primus architecture as introduced in the paper.
Please cite the orinal paper if you use this architecture in your work:*

    Wald, T., Roy, S., Isensee, F., Ulrich, C., Ziegler, S., Trofimova, D., ... & Maier-Hein, K. (2025). Primus: Enforcing attention usage for 3d medical image segmentation. arXiv preprint arXiv:2503.01835.

