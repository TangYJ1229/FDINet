import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.

    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)
    return result


def swap_mask(mask):
    swap = torch.where(mask == 1, torch.tensor(2).cuda(), torch.where(mask == 2, torch.tensor(1).cuda(), mask))
    return swap


def dice_loss(pred, label, eps=1e-7):

    pred = F.softmax(pred, dim=1)
    label = label.type(pred.type())
    dims = (0,) + tuple(range(2, label.ndimension()))
    intersection = torch.sum(pred * label, dims)
    cardinality = torch.sum(pred + label, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)


def change_dice_loss(pred1, pred2):
    output_pred1 = F.softmax(pred1, dim=1)
    output_pred2 = F.softmax(pred2, dim=1)
    _, label1 = torch.max(output_pred1.data, dim=1)
    # _, label2 = torch.max(output_pred2.data, dim=1)

    label1 = swap_mask(label1)
    # label2 = swap_mask(label2)
    label1_one_hot = make_one_hot(label1.unsqueeze(1), 3).squeeze(0)
    # label2_one_hot = make_one_hot(label2.unsqueeze(1), 3).squeeze(0)

    # dl1 = dice_loss(output_pred1, label2_one_hot)
    dl2 = dice_loss(output_pred2, label1_one_hot)

    # dl = (dl1 + dl2) / 2
    return dl2


# def contrastive_loss(pred1, pred2, label, mean=False):
#     bg_mask = label[:, 0, :, :]
#     fg1_mask = label[:, 1, :, :]
#     fg2_mask = label[:, 2, :, :]
#     output_pred1 = F.softmax(pred1, dim=1)
#     output_pred2 = F.softmax(pred2, dim=1)
#
#     loss_bg = F.mse_loss(output_pred1[:, 0, :, :] * bg_mask, output_pred2[:, 0, :, :] * bg_mask)
#     loss_fg1 = F.mse_loss(output_pred1[:, 1, :, :] * fg1_mask, output_pred2[:, 2, :, :] * fg1_mask)
#     loss_fg2 = F.mse_loss(output_pred1[:, 2, :, :] * fg2_mask, output_pred2[:, 1, :, :] * fg2_mask)
#
#     loss_ct = loss_bg + loss_fg1 + loss_fg2
#     if mean:
#         loss_ct = loss_ct.mean()
#
#     return loss_ct


# def change_constraction_loss(skip1, skip2, label):
#     label = make_one_hot(label.unsqueeze(1), 3)
#     bg_mask = label[:, 0, :, :].unsqueeze(1)
#     fg1_mask = label[:, 1, :, :].unsqueeze(1)
#     fg2_mask = label[:, 2, :, :].unsqueeze(1)
#
#     bg1_mean, bg1_std = calc_mean_var(skip1, bg_mask.cuda())
#     bg2_mean, bg2_std = calc_mean_var(skip2, bg_mask.cuda())
#
#     fg11_mean, fg11_std = calc_mean_var(skip1, fg1_mask.cuda())
#     fg12_mean, fg12_std = calc_mean_var(skip1, fg2_mask.cuda())
#
#     fg21_mean, fg21_std = calc_mean_var(skip2, fg1_mask.cuda())
#     fg22_mean, fg22_std = calc_mean_var(skip2, fg2_mask.cuda())
#
#     loss = nn.MSELoss()(bg1_mean, bg2_mean) + nn.MSELoss()(fg11_mean, fg22_mean) + nn.MSELoss()(fg12_mean, fg21_mean)
#     return loss
