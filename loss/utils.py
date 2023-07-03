import torch
import torch.nn as nn

from .loss_clf import BinaryCrossEntropy, SoftTargetCrossEntropy, LabelSmoothingCrossEntropy


def load_loss(task, *args, **kws):
    if task == 'clf':
        if kws['active_mixup']:
            # Label smoothing has been applied upstream by mixup target transform.
            # In this case, targets could be sparse or soft ones.
            if kws['bce']:
                loss_fn = BinaryCrossEntropy(kws['smoothing'], target_threshold=kws['bce_target_thresh'])
            else:
                loss_fn = SoftTargetCrossEntropy(kws['smoothing'])
        else:
            # In this case, targets could be sparse or soft ones.
            loss_fn = SoftTargetCrossEntropy(kws['smoothing'])
        return loss_fn
    else:
        pass
        return None

def loss_reg_l1(coef):
    coef = .0 if coef is None else coef
    def func(model_params):
        if coef <= 1e-8:
            return 0.0
        else:
            return coef * sum([torch.abs(W).sum() for W in model_params])
    return func
