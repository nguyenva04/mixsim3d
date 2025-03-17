import torch
from typing import Any
# from torchmetrics import R2Score
from drp.data.dataset import Drp3dBaseDataset


# class DrpR2Metric(R2Score):
#     def __init__(self, **kwargs: Any) -> None:
#         super().__init__(
#             num_outputs=Drp3dBaseDataset.output_size(),
#             multioutput="raw_values",
#             **kwargs,
#         )

#     def compute(self):
#         r2_score = super().compute()
#         return {
#             f"r2_{c}": r2_score[i].item() for i, c in enumerate(Drp3dBaseDataset.labels)
#         }
class DrpR2Metric:
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.output_size = Drp3dBaseDataset.output_size()

        self.predicted = torch.zeros(size=(0, self.output_size))
        self.target = torch.zeros(size=(0, self.output_size))

    def to(self, device):
        self.predicted = self.predicted.to(device)
        self.target = self.target.to(device)

        return self

    def update(self, predicted, target):
        self.predicted = torch.cat((self.predicted, predicted), dim=0)
        self.target = torch.cat((self.target, target), dim=0)

    def compute(self, train_flag="Train"):
        # Compute the mean of the target values
        self.predicted = mD_to_volxel(self.predicted)
        target_mean = torch.mean(self.target, dim=0)

        # Compute the total sum of squares (SS_tot)
        ss_tot = torch.sum((self.target - target_mean) ** 2, dim=0)

        # Compute the residual sum of squares (SS_res)
        ss_res = torch.sum((self.target - self.predicted) ** 2, dim=0)

        # Compute the R² score
        r2_score = 1 - ss_res / ss_tot

        print("r2_score", r2_score)

        # Return the mean R² score across all output dimensions
        return {
            f"r2_{train_flag}": r2_score[i].item()
            for i, _ in enumerate(Drp3dBaseDataset.labels)
        }

    def reset(self):
        device = self.predicted.device

        self.predicted = torch.zeros(size=(0, self.output_size))
        self.target = torch.zeros(size=(0, self.output_size))

        self.to(device)


def mD_to_volxel(k_voxel):
    resolution = 0.18283724910288787
    return k_voxel * (resolution * resolution * 1013.25)

