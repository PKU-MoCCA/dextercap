import torch


def interval_loss(x, a, b):
    """
    Calculate the loss of the interval [a, b]

    Parameters:
    - x (torch.Tensor, shape=(1, num_dof)): The value to be clipped
    - a (torch.Tensor, shape=(num_dof,)): The lower bound of the interval
    - b (torch.Tensor, shape=(num_dof,)): The upper bound of the interval

    Returns:
    - torch.Tensor: The loss of the interval [a, b]
    """
    # 计算 max(a - x, 0)
    left_loss = torch.clamp(a - x, min=0)
    if torch.isnan(left_loss).any():
        print(f"Interval loss for left_loss: {left_loss.item()}")
        exit()
    # 计算 max(x - b, 0)
    right_loss = torch.clamp(x - b, min=0)
    if torch.isnan(right_loss).any():
        print(f"Interval loss for right_loss: {right_loss.item()}")
        exit()
    # 返回左区间和右区间的损失之和的平方
    return (left_loss + right_loss) ** 2


def loss_regularization_dof(hand_dof):
    """
    Calculate regularization loss for hand DOF parameters.

    It's very simple, because we use DOF to optimize the hand pose, so the loss is the sum of the value out of the normal range.
    """
    from HandReconstruction.Utility.utils_hand import dof_limit

    regularization_matrix = interval_loss(hand_dof, dof_limit[:, 0], dof_limit[:, 1])
    regularization_loss = torch.mean(regularization_matrix)
    return regularization_loss
