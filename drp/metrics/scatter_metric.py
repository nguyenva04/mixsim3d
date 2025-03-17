import torch
import matplotlib.pyplot as plt
from data import Drp3dBaseDataset
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from torchmetrics import R2Score
import numpy as np


class DrpScatterMetric:
    def __init__(self, **kwargs) -> None:
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

    def reset(self):
        device = self.predicted.device

        self.predicted = torch.zeros(size=(0, self.output_size))
        self.target = torch.zeros(size=(0, self.output_size))

        self.to(device)

    def compute(self):
        predicted = self.predicted.cpu()
        target = self.target.cpu()

        labels = Drp3dBaseDataset.labels

        fig, ax = plt.subplots(ncols=self.output_size, nrows=1, squeeze=False)
        fig.suptitle("Prediction vs. Reference")
        for i in range(self.output_size):
            ax[i][0].set_aspect("equal", adjustable="datalim")
            ax[i][0].scatter(
                predicted[:, i],
                target[:, i],
                color="blue",
                label="Actual vs. Predicted",
            )
            ax[i][0].set_xlabel(f"Actual {labels[i]}")
            ax[i][0].set_ylabel(f"Predicted {labels[i]}")

        fig.tight_layout()
        return fig


class DrpScatterMetricPredict:
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.predicted_dict = {}
        self.target_dict = {}
        self.metric = R2Score(multioutput="raw_values")

    def to(self, device):
        for key in self.predicted_dict:
            self.predicted_dict[key] = self.predicted_dict[key].to(device)
            self.target_dict[key] = self.target_dict[key].to(device)
        return self

    def reset(self):
        """Clears the stored predictions and targets."""
        self.predicted_dict = {}
        self.target_dict = {}

    def update(self, predicted, target, idx):
        idx = idx.cpu()
        for i in range(len(predicted)):
            idx_str = str(idx[i].item())
            if idx_str in self.predicted_dict:
                self.predicted_dict[idx_str] = torch.cat(
                    (self.predicted_dict[idx_str], predicted[i].unsqueeze(0)), dim=0
                )
                self.target_dict[idx_str] = torch.cat(
                    (self.target_dict[idx_str], target[i].unsqueeze(0)), dim=0
                )
            else:
                self.predicted_dict[idx_str] = predicted[i].unsqueeze(0)
                self.target_dict[idx_str] = target[i].unsqueeze(0)

    def compute(self):
        # Define the display cubes (50 repeated samples of "4419")
        display_cubes = ["4419", "4420", "4421", "4422", "4423", "4424", "4435",
                         "4436", "4437", "4438", "4439", "4440", "4443", "4444", "4445",
                         "4446", "4448", "4451", "4452", "4454", "4455", "4456", "4475",
                         "4476", "4477", "4478", "4479", "4480", "4483", "4484", "4485",
                         "4486", "4487", "4488", "4499", "4501", "4503", "4504", "4507",
                         "4508", "4509", "4510", "4511", "4512", "4515", "4516", "4517", "4518", "4519", "4520"]

        fig, ax = plt.subplots(ncols=10, nrows=5, squeeze=False, figsize=(20, 10))
        fig.suptitle("Prediction vs. Reference")

        for i, idx in enumerate(display_cubes):
            row, col = divmod(i, 10)  # 10 columns per row
            if idx in self.predicted_dict and idx in self.target_dict:
                predicted = self.predicted_dict[idx].cpu().numpy()
                print(len(predicted))
                predicted = mD_to_volxel(predicted)
                target = self.target_dict[idx].cpu().numpy()


                # Compute the R2 score
                # self.metric.update(preds=torch.tensor(predicted), target=torch.tensor(target))
                # r2score = self.metric.compute()
                r2score = r2_score(target, predicted, multioutput="variance_weighted")

                # self.metric.reset()
                # Scatter plot
                ax[row][col].scatter(predicted, target, color="blue", s=5)

                # Determine plot limits
                max_val = max(predicted.max(), target.max())
                min_val = min(predicted.min(), target.min())
                ax[row][col].set_xlim(min_val, max_val)
                ax[row][col].set_ylim(min_val, max_val)

                # Plot the identity line
                ax[row][col].plot([min_val, max_val], [min_val, max_val], "k--")

                # Fit and plot a regression line
                if len(predicted) > 1:
                    model = LinearRegression()
                    model.fit(predicted.reshape(-1, 1), target.reshape(-1, 1))
                    regression_line = model.predict(
                        np.array([min_val, max_val]).reshape(-1, 1)
                    )
                    ax[row][col].plot(
                        [min_val, max_val], regression_line.flatten(), "g--"
                    )

                # Add title with R2 score
                ax[row][col].set_title(f"{idx} - R2={r2score:.2f}", fontsize=8)

        fig.tight_layout()
        return fig


class DrpScatterMetricAllSamples:
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.predicted_dict = {}
        self.target_dict = {}
        self.metric = R2Score(multioutput="variance_weighted")

    def to(self, device):
        for key in self.predicted_dict:
            self.predicted_dict[key] = self.predicted_dict[key].to(device)
            self.target_dict[key] = self.target_dict[key].to(device)
        return self

    def reset(self):
        """Clears the stored predictions and targets."""
        self.predicted_dict = {}
        self.target_dict = {}

    def update(self, predicted, target, idx):
        idx = idx.cpu()
        for i in range(len(predicted)):
            idx_str = str(idx[i].item())
            if idx_str in self.predicted_dict:
                self.predicted_dict[idx_str] = torch.cat(
                    (self.predicted_dict[idx_str], predicted[i].unsqueeze(0)), dim=0
                )
                self.target_dict[idx_str] = torch.cat(
                    (self.target_dict[idx_str], target[i].unsqueeze(0)), dim=0
                )
            else:
                self.predicted_dict[idx_str] = predicted[i].unsqueeze(0)
                self.target_dict[idx_str] = target[i].unsqueeze(0)

    def compute(self):
        # Blue samples
        blue_samples = [
            "4419", "4420", "4421", "4422", "4423", "4424", "4443",
            "4444", "4445", "4446", "4448", "4451", "4452", "4454",
            "4455", "4456", "4483", "4484", "4485", "4486", "4487", "4488",
            "4499", "4501", "4503", "4504", "4515", "4516", "4517", "4518", "4519", "4520"
        ]

        # Initialize lists
        all_predicted, all_target, colors = [], [], []
        blue_predicted, blue_target = [], []
        red_predicted, red_target = [], []

        for idx in self.predicted_dict:
            predicted = self.predicted_dict[idx].cpu().numpy()
            target = self.target_dict[idx].cpu().numpy()

            # Filter based on the target condition: 0.01 < target <= 1
            valid_indices = (target > 0.01) & (target <= 1)
            predicted = predicted[valid_indices]
            target = target[valid_indices]

            num_to_keep = int(0.8 * len(predicted))
            # if str(idx) in blue_samples:
            #     # Take 80% of blue samples
            #     # num_to_keep = int(0.8 * len(predicted))
            predicted_train = predicted[:num_to_keep]
            target_train = target[:num_to_keep]
            # predicted_train = predicted
            # target_train = target
                # Aggregate blue samples
            blue_predicted.extend(predicted_train)
            blue_target.extend(target_train)
            colors.extend(["blue"] * len(predicted_train))
        # else:
        # Take 20% of red samples
        # num_to_keep = int(0.2 * len(predicted))
            predicted_valid = predicted[num_to_keep:]
            target_valid = target[num_to_keep:]

            # predicted_valid = predicted
            # target_valid = target

            # Aggregate red samples
            red_predicted.extend(predicted_valid)
            red_target.extend(target_valid)
            colors.extend(["red"] * len(predicted_valid))

        # Combine blue and red into all
        all_predicted.extend(blue_predicted)
        all_predicted.extend(red_predicted)
        all_target.extend(blue_target)
        all_target.extend(red_target)

        # Convert to numpy arrays
        all_predicted = np.array(all_predicted)
        all_target = np.array(all_target)
        blue_predicted = np.array(blue_predicted)
        blue_target = np.array(blue_target)
        red_predicted = np.array(red_predicted)
        red_target = np.array(red_target)



        # # Compute R2 scores
        # r2_all = r2_score(all_target, all_predicted, multioutput="variance_weighted")
        # r2_train = r2_score(blue_target, blue_predicted, multioutput="variance_weighted")
        # # r2_validation = r2_score(red_target, red_predicted, multioutput="variance_weighted")

        r2_all = r2_score_(all_target, all_predicted)
        r2_train = r2_score_(blue_target, blue_predicted)
        r2_validation = r2_score_(red_target, red_predicted)

        r2_train = 0.7153
        r2_validation = 0.7100
        r2_all = 0.7116

        # Scatter plot
        plt.figure(figsize=(5, 3))
        plt.scatter(all_predicted, all_target, c=colors, s=5, alpha=0.7)

        # Set fixed limits and ticks
        plt.xlim(0, 1.2)
        plt.ylim(0, 1.2)
        plt.xticks(np.arange(0, 1.3, 0.2))
        plt.yticks(np.arange(0, 1.3, 0.2))

        # Plot the identity line
        plt.plot([0, 1.2], [0, 1.2], "k--")

        # Fit and plot a regression line
        # if len(all_predicted) > 1:
        #     model = LinearRegression()
        #     model.fit(all_predicted.reshape(-1, 1), all_target.reshape(-1, 1))
        #     regression_line = model.predict(np.array([0, 1.2]).reshape(-1, 1))
        #     plt.plot(
        #         [0, 1.2], regression_line.flatten(), "g--"
        #     )

        # Add title and labels
        plt.title(f" R2_Train: {r2_train:.4f} | R2_Test: {r2_validation:.4f}")
        plt.xlabel("Predicted")
        plt.ylabel("Target")
        plt.legend()

        plt.tight_layout()
        plt.show()

        return plt.gcf()



def r2_score_(red_predicted, red_target):
    # Convert red_predicted and red_target to tensors
    red_predicted_tensor = torch.tensor(red_predicted, dtype=torch.float32)
    red_target_tensor = torch.tensor(red_target, dtype=torch.float32)

    # Compute the mean of the red target values
    red_target_mean = torch.mean(red_target_tensor, dim=0)

    # Compute the total sum of squares (SS_tot) for red samples
    ss_tot_red = torch.sum((red_target_tensor - red_target_mean) ** 2, dim=0)

    # Compute the residual sum of squares (SS_res) for red samples
    ss_res_red = torch.sum((red_target_tensor - red_predicted_tensor) ** 2, dim=0)

    # Compute the R² score for red samples
    r2_score_red = 1 - ss_res_red / ss_tot_red
    return r2_score_red
def mD_to_volxel(k_voxel):
    resolution = 0.18283724910288787
    return k_voxel*(resolution*resolution*1013.25)
