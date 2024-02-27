import torch.nn.functional as F
def reinforce_loss(output, target, reward):
    """
    Args:
    output (torch.Tensor): The output from the model.
    target (torch.Tensor): The ground truth labels.
    reward (torch.Tensor): The reward signal to adjust the loss.

    Returns:
    torch.Tensor: The adjusted loss.
    """
    loss = F.cross_entropy(output, target)
    adjusted_loss = loss * reward
    return adjusted_loss
