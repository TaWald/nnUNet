from torch import nn
import torch
from torch.utils.hooks import RemovableHandle


def register_feature_extraction_hook(
    module: nn.Module,
    module_name: str,
) -> tuple[RemovableHandle, list[torch.Tensor]]:
    """
    A hook function to extract features from a specific module during the forward pass.

    Args:
        module (nn.Module): The module from which to extract features.
        module_name (str): The name of the module.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing the extracted features.
    """
    features = []
    wanted_module: torch.nn.Module = module.get_submodule(module_name)
    handle = wanted_module.register_forward_hook(lambda m, i, o: features.append(o))

    return handle, features


class ProbeArchitecture(nn.Module):

    def __init__(self, network_to_probe: nn.Module, probe_position: str, probe_module: nn.Module):
        """
        Architecture that probes a given network at the specified positions with a given module.
        The `probe_module` receives whatever the probes return, so it needs to be able to handle the output of the probes.

        Args:
            network_to_probe (nn.Module): Module which will have `probe` registered at `probe_positions`.
            probe_positions (list[str]): Keys to the submodule where `probe` will be registered.
        """
        super().__init__()
        self.network_to_probe = network_to_probe
        # for p in self.network_to_probe.parameters():
        #     p.requires_grad = False
        self.probe_position = probe_position
        self.probe_module = probe_module
        self.hook_handles = []
        self.hook_outputs = []

        self.probes_are_attached = False

    def attach_probes(self):
        """
        Attach the probes to the network at the specified positions.
        """

        features: list[torch.Tensor]
        handle: RemovableHandle
        handle, features = register_feature_extraction_hook(self.network_to_probe, self.probe_position)
        self.hook_handles.append(handle)
        self.hook_outputs = features

    def detach_probes(self):
        """
        Detach the probes from the network.
        """
        for handle in self.hook_handles:
            handle.remove()
        self.probes_are_attached = False
        self.hook_handles = []
        self.hook_outputs = []

    def train(self, mode=True):
        super().train(mode)  # Recursibely set s
        self.network_to_probe.train(False)  # This module should never be changed!
        return self

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass through the network and return the outputs of the probes.

        Args:
            x (torch.Tensor): Input tensor to the network.
        Returns:
            list[torch.Tensor]: List of outputs from the probes.
        """
        if not self.probes_are_attached:
            self.probes_are_attached = True
            self.attach_probes()
        with torch.no_grad():
            self.network_to_probe(x)  # We forward the module to trigger the hooks, but don't use the output.
        probe_outputs = self.hook_outputs[0]  # We grab the outputs -- this is a torch.tensor
        # Now forward through our probe module and return the outputs.
        predictions = self.probe_module(probe_outputs)
        self.hook_outputs.clear()  # Clear the outputs to assure we don't keep the old tensors in the list and memory.
        return predictions
