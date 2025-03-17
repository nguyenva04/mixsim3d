import torch
import matplotlib.pyplot as plt
from data import Drp3dBaseDataset
from sklearn.manifold import TSNE
import numpy as np


class TSNEMetric:
    def __init__(self, dim=256, **kwargs) -> None:
        super().__init__(**kwargs)
        self.output_size = Drp3dBaseDataset.output_size()
        self.dim = dim
        self.predicted = torch.zeros(size=(0, dim))
        self.target = torch.zeros(size=(0, 1))
        self.tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=5)

    def to(self, device):
        self.predicted = self.predicted.to(device)
        self.target = self.target.to(device)

        return self

    def update(self, predicted, target):
        self.predicted = torch.cat((self.predicted, predicted), dim=0)
        self.target = torch.cat((self.target, target), dim=0)

    def reset(self):
        device = self.predicted.device

        self.predicted = torch.zeros(size=(0, self.dim))
        self.target = torch.zeros(size=(0, 1))

        self.to(device)

    def compute(self):
        predicted = self.predicted.detach().cpu().numpy()
        target = self.target.detach().cpu().numpy().flatten()  # Flatten to make sure target is 1D

        fisher_criterion = self.fisher_compute(predicted, target)

        # Perform t-SNE dimensionality reduction
        tsne_result = self.tsne.fit_transform(predicted)

        # Define target names and corresponding colors
        target_names = ["RT1", "RT2", "RT3", "RT4", "RT5", "RT6", "RT7", "RT8", "RT9"]
        colors = plt.cm.get_cmap("tab10", len(target_names)).colors

        # Darken the colors by multiplying RGB values by a factor less than 1
        darkening_factor = 1  # You can adjust this factor to make the colors as dark as you like
        darker_colors = [tuple([darkening_factor * c for c in color]) for color in colors]

        # Create a mapping from target labels to darker colors
        target_to_color = {i: darker_colors[i] for i in range(len(target_names))}

        # Map target labels to their corresponding darker colors
        target_colors = [target_to_color[int(t)] for t in target]

        # Create the plot
        fig, ax = plt.subplots(ncols=1, nrows=1, squeeze=False)

        ax[0][0].set_aspect("equal", adjustable="datalim")
        scatter = ax[0][0].scatter(tsne_result[:, 0], tsne_result[:, 1], c=target_colors, s=15)

        # Add Fisher criterion to the title
        title = f"Fisher Criterion: {fisher_criterion:.2f}"
        ax[0][0].set_title(title)

        # Create a legend
        handles = []
        for i, target_name in enumerate(target_names):
            handles.append(plt.Line2D([0], [0], marker='o', color=darker_colors[i], linestyle='None', markersize=5,
                                      label=target_name))

        ax[0][0].legend(handles=handles, title="Classes")
        fig.tight_layout()

        return fig

    @staticmethod
    def fisher_compute(predicted, target):

        # Step 1: Separate predicted data into classes
        class_means = []
        class_variances = []

        unique_classes = np.unique(target)
        target = np.squeeze(target)
        for cls in unique_classes:
            class_data = predicted[target == cls]
            class_mean = np.mean(class_data, axis=0)
            class_variance = np.var(class_data, axis=0)
            class_means.append(class_mean)
            class_variances.append(class_variance)

        # Convert lists to arrays for easier manipulation
        class_means = np.stack(class_means)
        class_variances = np.stack(class_variances)

        # Step 2: Calculate intra-class variance
        intra_class_variance = np.mean(class_variances)

        # Step 3: Calculate extra-class variance
        overall_mean = np.mean(predicted, axis=0)
        extra_class_variance = np.mean((class_means - overall_mean) ** 2)

        # Step 4: Compute Fisher criterion
        fisher_criterion = extra_class_variance / intra_class_variance
        return fisher_criterion
