import torch
import torch.nn.functional as F

class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss, accuracy, etc.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0       # current value
        self.avg = 0       # average
        self.sum = 0       # sum of all values
        self.count = 0     # number of updates

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


def accuracy(logits, targets, topk=(1,)):
    """
    Computes the top-k accuracy for the specified values of k.
    
    Args:
        logits: Tensor of shape [B, C]
        targets: Tensor of shape [B]
        topk: tuple of int

    Returns:
        List[float]: accuracy for each k in topk
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)  # [B, maxk]
        pred = pred.t()  # [maxk, B]
        correct = pred.eq(targets.view(1, -1).expand_as(pred))  # [maxk, B]

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            acc = (correct_k / batch_size) * 100.0
            results.append(acc.item())

        return results[0] if len(results) == 1 else results
    
def Cosine_classifier(support, query, temperature=1):
    """Cosine classifier"""
    # normalize for cosine distance
    # l = cosine_similarity(query.unsqueeze(1), support.unsqueeze(0), -1)
    proto = F.normalize(support, dim=-1)
    query = F.normalize(query, dim=-1)
    logits = torch.mm(query, proto.permute([1, 0])) / temperature
    predict = torch.argmax(logits, dim=1)
    return logits, predict
