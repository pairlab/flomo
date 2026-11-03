import torch


def get_device(prefer_device: str = None):
    """
    Utility function that can be used from anywhere in the codebase to
    grab the current torch device.
    Args:
        prefer_device: Allows the user to specify a torch device they
        want to use.

    Returns: The preferred torch device if requested, otherwise cuda, or mps, or cpu,
    in that order of preference.

    """
    if prefer_device:
        return torch.device(prefer_device)

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
