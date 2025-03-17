import torch
from typing import Sequence


class TopKAccuracy:
    def __init__(self,
                 num_classes,
                 top_k: Sequence[int] = (1, 5)) -> None:
        self.output_size = num_classes

        self.predicted = torch.zeros(size=(0, self.output_size))
        self.target = torch.zeros(size=(0, 1))

        self.top_k = top_k

    def to(self, device):
        self.predicted = self.predicted.to(device)
        self.target = self.target.to(device)

        return self

    def update(self, predicted, target):
        if target.dim() == 1:
            target = target.unsqueeze(1)

        self.predicted = torch.cat((self.predicted, predicted), dim=0)
        self.target = torch.cat((self.target, target), dim=0)

    def compute(self, train=True):

        with torch.no_grad():
            maxk = max(self.top_k)

            batch_size = self.target.size(0)

            _, pred = self.predicted.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(self.target.view(1, -1).expand_as(pred))

            res = []
            for k in self.top_k:
                correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return {
            f"top_{c}_accuracy_{'Train' if train else 'Validation'}": res[i].item() for i, c in enumerate(self.top_k)
        }

    def reset(self):
        device = self.predicted.device

        self.predicted = torch.zeros(size=(0, self.output_size))
        self.target = torch.zeros(size=(0, 1))

        self.to(device)


class TopKAccuracyTest:
    def __init__(self,
                 num_classes,
                 top_k: Sequence[int] = (1, 5)) -> None:
        self.output_size = num_classes

        self.predicted = torch.zeros(size=(0, self.output_size))
        self.target = torch.zeros(size=(0, 1))

        self.top_k = top_k

    def to(self, device):
        self.predicted = self.predicted.to(device)
        self.target = self.target.to(device)

        return self

    def update(self, predicted, target):
        if target.dim() == 1:
            target = target.unsqueeze(1)

        self.predicted = torch.cat((self.predicted, predicted), dim=0)
        self.target = torch.cat((self.target, target), dim=0)

    def compute(self, train=True):
        with torch.no_grad():
            maxk = max(self.top_k)
            batch_size = self.target.size(0)

            _, pred = self.predicted.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(self.target.view(1, -1).expand_as(pred))

            # Top-k Accuracy
            res = []
            print(batch_size)
            for k in self.top_k:
                correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))

            # Flatten predicted and target for metrics calculation
            predicted_classes = pred[0]  # Taking the top-1 prediction
            target_classes = self.target.squeeze(1)

            # Calculating Precision, Recall, F1
            true_positive = torch.zeros(self.output_size)
            false_positive = torch.zeros(self.output_size)
            false_negative = torch.zeros(self.output_size)

            for c in range(self.output_size):
                true_positive[c] = ((predicted_classes == c) & (target_classes == c)).sum().item()
                false_positive[c] = ((predicted_classes == c) & (target_classes != c)).sum().item()
                false_negative[c] = ((predicted_classes != c) & (target_classes == c)).sum().item()

            precision = (true_positive / (true_positive + false_positive + 1e-10)).mean().item()
            recall = (true_positive / (true_positive + false_negative + 1e-10)).mean().item()
            f1 = (2 * precision * recall / (precision + recall + 1e-10))

            return {
                **{f"top_{c}_accuracy_{'Train' if train else 'Validation'}": res[i].item() for i, c in enumerate(self.top_k)},
                f"precision_{'Train' if train else 'Validation'}": precision,
                f"recall_{'Train' if train else 'Validation'}": recall,
                f"f1_{'Train' if train else 'Validation'}": f1
            }

    def reset(self):
        device = self.predicted.device

        self.predicted = torch.zeros(size=(0, self.output_size))
        self.target = torch.zeros(size=(0, 1))

        self.to(device)
