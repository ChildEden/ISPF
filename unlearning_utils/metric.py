import torch
import torch.nn.functional as F

import numpy as np

class AggregateScalar(object):
    """
    Computes and stores the average and std of stream.
    Mostly used to average losses and accuracies.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0001  # DIV/0!
        self.sum = 0

    def update(self, val, w=1):
        """
        :param val: new running value
        :param w: weight, e.g batch size
        """
        self.sum += w * (val)
        self.count += w

    def avg(self):
        return self.sum / self.count
    
class AggregateVector(object):
    """
    Computes and stores the average and std of stream.
    Mostly used to average losses and accuracies.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.container = []

    def update(self, val):
        """
        :param val: new running value
        """
        self.container.append(val)

    def avg(self):
        return np.mean(self.container, axis=0)
    
    def std(self):
        return np.std(self.container, axis=0)
    
def top_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if len(target.shape) > 1:
        target = torch.argmax(target, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1 / batch_size))
    return res


def accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

def precision(y_pred, y_true, unlearn_classes_set):
    res = np.zeros(len(unlearn_classes_set))
    for idx, cls in enumerate(unlearn_classes_set):
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        if tp == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)
        res[idx] = precision
    return res

def recall(y_pred, y_true, unlearn_classes_set):
    res = np.zeros(len(unlearn_classes_set))
    for idx, cls in enumerate(unlearn_classes_set):
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        if tp == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)
        res[idx] = recall
    return res

def f1_score(y_pred, y_true, unlearn_classes_set):
    res = np.zeros(len(unlearn_classes_set))
    for idx, cls in enumerate(unlearn_classes_set):
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        if tp == 0:
            f1 = 0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)
        res[idx] = f1
    return res


@torch.no_grad()
def plain_test(net, test_loader, device='cuda'):
    running_test_acc = AggregateScalar()
    net.eval()
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits = net(x)
        acc = top_accuracy(logits, y, topk=(1,))[0]
        running_test_acc.update(float(acc), len(x))
    return running_test_acc.avg()

@torch.no_grad()
def test_all_in_one(net, test_loader, unlearn_classes, device='cuda'):
    if isinstance(unlearn_classes, int):
        unlearn_classes = [unlearn_classes]
    unlearn_classes = np.array(unlearn_classes)

    running_test_unlearn_acc = AggregateScalar()
    running_test_retain_acc = AggregateScalar()
    running_test_overall_acc = AggregateScalar()
    
    net.eval()
    
    predictions = []
    targets = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = net(x)
            predictions.append(logits)
            targets.append(y)
        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)

        unlearn_item_idxs = torch.zeros_like(targets, dtype=torch.bool)
        for c in unlearn_classes:
            unlearn_item_idxs = unlearn_item_idxs | (targets == c)
        retain_item_idxs = ~unlearn_item_idxs
        unlearn_predictions = predictions[unlearn_item_idxs]
        unlearn_targets = targets[unlearn_item_idxs]
        retain_predictions = predictions[retain_item_idxs]
        retain_targets = targets[retain_item_idxs]

        unlearn_acc = top_accuracy(unlearn_predictions, unlearn_targets, topk=(1,))[0]
        retain_acc = top_accuracy(retain_predictions, retain_targets, topk=(1,))[0]
        overall_acc = top_accuracy(predictions, targets, topk=(1,))[0]
        running_test_unlearn_acc.update(float(unlearn_acc), len(unlearn_predictions))
        running_test_retain_acc.update(float(retain_acc), len(retain_predictions))
        running_test_overall_acc.update(float(overall_acc), len(predictions))

    return {
        'unlearn_acc': running_test_unlearn_acc.avg(),
        'retain_acc': running_test_retain_acc.avg(),
        'overall_acc': running_test_overall_acc.avg()
    }

@torch.no_grad()
def test_all_in_one_(net, test_loader, unlearn_class, device='cuda'):
    running_test_unlearn_acc = AggregateScalar()
    running_test_retain_acc = AggregateScalar()
    running_test_overall_acc = AggregateScalar()
    
    net.eval()
    
    predictions = []
    targets = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = net(x)
            predictions.append(logits)
            targets.append(y)
        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)

        unlearn_item_idxs = targets == unlearn_class
        retain_item_idxs = ~unlearn_item_idxs
        unlearn_predictions = predictions[unlearn_item_idxs]
        unlearn_targets = targets[unlearn_item_idxs]
        retain_predictions = predictions[retain_item_idxs]
        retain_targets = targets[retain_item_idxs]

        unlearn_acc = top_accuracy(unlearn_predictions, unlearn_targets, topk=(1,))[0]
        retain_acc = top_accuracy(retain_predictions, retain_targets, topk=(1,))[0]
        overall_acc = top_accuracy(predictions, targets, topk=(1,))[0]
        running_test_unlearn_acc.update(float(unlearn_acc), len(unlearn_predictions))
        running_test_retain_acc.update(float(retain_acc), len(retain_predictions))
        running_test_overall_acc.update(float(overall_acc), len(predictions))

    return {
        'unlearn_acc': running_test_unlearn_acc.avg(),
        'retain_acc': running_test_retain_acc.avg(),
        'overall_acc': running_test_overall_acc.avg()
    }


@torch.no_grad()
def test(net, loader, idxs, class_count, device):
    net.eval()
    pred = np.array([])
    label = np.array([])
    record_loss = []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = torch.tensor(x.clone().detach().numpy(), dtype=torch.float32).to(device)
            y = y.to(device)
            # y = y.cpu().numpy()
            label = np.concatenate((label, y.cpu().numpy()))

            y_pred = net(x)

            loss = F.cross_entropy(y_pred, y)
            record_loss.append(loss.item())
            _, predicted = torch.max(y_pred.data, 1)
            predicted = predicted.cpu().numpy()
            pred = np.concatenate((pred, predicted))

    pred = pred[idxs]
    label = label[idxs]
    
    # accuracy
    acc = accuracy(pred, label)

    # precision
    p = precision(pred, label, class_count)

    # recall
    r = recall(pred, label, class_count)

    # f1 score
    f1 = f1_score(pred, label, class_count)
    # save pred
    # np.save('pred.npy', pred)
    # np.save('label.npy', label)
    
    return acc, p, r, f1, np.mean(record_loss)
        


