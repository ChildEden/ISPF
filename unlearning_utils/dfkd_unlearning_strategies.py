import torch

def block_unlearn_classes(t_out, unlearn_classes):
    """
    Block samples from unlearn_classes to be classified by teacher.
    """
    # t_out = teacher(inputs)
    preds = t_out.max(1)[1]
    mask = torch.ones_like(preds, dtype=torch.bool)
    for c in unlearn_classes:
        mask = mask & (preds != c)

    return mask

def filter_unlearn_classes(t_out, unlearn_classes, threshold=0.01):
    """
    Filter samples from unlearn_classes to be classified by teacher.
    """
    probs = torch.softmax(t_out, dim=1)
    mask = torch.ones(t_out.size(0), dtype=torch.bool).to(t_out.device)
    for c in unlearn_classes:
        mask = mask & (probs[:, c] < threshold)


    return mask


def filter_low_confidence(t_out, threshold=0.6):
    """
    Filter samples from unlearn_classes to be classified by teacher.
    """
    probs = torch.softmax(t_out, dim=1)
    mask = torch.ones(t_out.size(0), dtype=torch.bool).to(t_out.device)
    
    max_probs, _ = probs.max(dim=1)
    mask = mask & (max_probs > threshold)

    return mask
