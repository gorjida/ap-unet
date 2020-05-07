import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn

from dice_loss import dice_coeff


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            if net.n_classes > 1:
                tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred, true_masks).item()
            pbar.update()

    return tot / n_val

def eval_net_AP_Power(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    totAP = 0
    totPower = 0
    mseLoss = nn.MSELoss()

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks, true_powers = \
                batch['image'], batch['mask'], batch['power']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)
            true_powers = true_powers.to(device=device, dtype=torch.long)
            with torch.no_grad():
                mask_pred = net(imgs)

            if "AP" in mask_pred:

                if net.n_classes > 1:
                    totAP += F.cross_entropy(mask_pred["AP"],
                                             true_masks).item()
                else:
                    pred = torch.sigmoid(mask_pred["AP"])
                    pred = (pred > 0.5).float()
                    totAP += dice_coeff(pred, true_masks).item()
            if "power" in mask_pred:
                totPower += mseLoss(mask_pred["power"],true_powers)
            pbar.update()

    return totAP / n_val, totPower/ n_val

