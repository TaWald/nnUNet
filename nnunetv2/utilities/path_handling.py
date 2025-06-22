import os
from copy import deepcopy


def maybe_resolve_relative_path(path: str) -> str:
    """
    Resolve the relative path `$nnssl_results/` to an absolute path with the help of the `nnssl_results` environment variable.

    This function checks if the provided path starts with `$nnssl_results/`. If it does, it retrieves the value of the `nnssl_results` environment variable,
    removes any trailing slashes, and replaces the prefix with the actual path. If the environment variable is not set, it raises a ValueError.

    Args:
        path (str): The path to resolve.

    Raises:
        ValueError: If we encounter a path that starts with `$nnssl_results/` but the `nnssl_results` environment variable is not set.

    Returns:
        str: The resolved absolute path if the input path starts with `$nnssl_results/`, otherwise the original path.
    """
    # If the path is not a URL, we assume it's a local path. But it might be a relative path from `nnssl_results`.
    # If so, we try to resolve this relative path to an absolute path.
    if path.startswith("$nnssl_results/"):
        # Resolve the path relative to the nnssl_results directory
        nnssl_results_dir = os.environ.get("nnssl_results", None)
        if nnssl_results_dir is None:
            raise ValueError(
                "The environment variable 'nnssl_results' is not set but was given in the pretrained checkpoint path."
                "Please set it to the path so the `nnssl_results` environment variable so it can be resolved."
            )
        if nnssl_results_dir.endswith("/"):  # Ensure no trailing slash
            nnssl_results_dir = nnssl_results_dir[:-1]
        # Replace the prefix with the actual path, and stitching the paths together
        path = os.path.join(nnssl_results_dir, deepcopy(path).replace("$nnssl_results/", ""))
    return path


def maybe_absolute_to_relative_path(path: str) -> str:
    """
    Convert an absolute path to a relative path by replacing the `nnssl_results` directory with `$nnssl_results/`.

    This function checks if the provided path starts with the `nnssl_results` directory. If it does, it replaces that part of the path with `$nnssl_results/`.
    If the path does not start with `nnssl_results`, it returns the original path.

    Args:
        path (str): The absolute path to convert.
    Returns:
        str: The relative path with the `nnssl_results` directory replaced by `$nnssl_results/`, or the original path if it does not start with `nnssl_results`.
    """
    nnssl_results_dir = os.environ.get("nnssl_results", None)
    if nnssl_results_dir is None:
        return path  # If the environment variable is not set, we can't replace an absolute path with it
    if path.startswith(nnssl_results_dir):
        # Replace the absolute path with the relative path
        path_from_nnssl_results = deepcopy(path).replace(nnssl_results_dir, "")
        if path_from_nnssl_results.startswith("/"):
            path_from_nnssl_results = path_from_nnssl_results[1:]  # Remove leading slash if present
        path = os.path.join("$nnssl_results", path_from_nnssl_results)
    return path
