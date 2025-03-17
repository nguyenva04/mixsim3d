import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from scipy.ndimage import gaussian_filter
from torchvision.transforms import transforms
from torchvision import transforms
from drp.utils.config import Config


class Rotate3D:
    def __init__(self, angles=(90,), axes=(1, 2), p=0.5):
        """
        Initialize the Rotate3D class.

        Args:
            angles (tuple): A tuple of angles (in degrees) by which to rotate the image.
            axes (tuple): A tuple of axes along which to rotate the image.
            p (float): Probability of applying the rotation.
        """
        self.angles = angles
        self.axes = axes
        self.probability = p

    def __call__(self, image):
        """
        Apply the rotation to the image with the given probability.

        Args:
            image (np.ndarray): The input image array.

        Returns:
            np.ndarray: The rotated image array.
        """
        if np.random.rand() < self.probability:
            angle = np.random.choice(self.angles)
            image = self._rotate_image(torch.tensor(image), angle, self.axes)
        return image

    def _rotate_image(self, image, angle, axes):
        """
        Rotate the image by the given angle along the specified axes.

        Args:
            image (np.ndarray): The input image array.
            angle (int): The angle by which to rotate the image.
            axes (tuple): The axes along which to rotate the image.

        Returns:
            np.ndarray: The rotated image array.
        """
        k = angle // 90  # Convert angle to number of 90 degree rotations
        return np.array(torch.rot90(image, k, dims=axes))


class GaussianNoise:
    def __init__(self, p=0.5, mean=0.0, std=1.0):
        """
        Initialization of the class.

        Parameters:
        - p: The probability of applying Gaussian noise.
        - mean: The mean of the Gaussian noise distribution.
        - std: The standard deviation of the Gaussian noise distribution.
        """
        self.mean = mean
        self.std = std
        self.probability = p

    def __call__(self, x):
        """
        Applies Gaussian noise to the input array.

        Parameters:
        - x: A 3D array (or a batch of 3D arrays) representing the image(s).

        Returns:
        - The input array with Gaussian noise added, if the random condition is met.
        """
        if np.random.rand(1) < self.probability:
            noise = np.random.normal(self.mean, self.std, x.shape)
            return (x + noise).astype(np.float32)
        else:
            return x


class Cutout3D:
    def __init__(self, p=0.5, num_patches=1, patch_size=(10, 10, 10)):
        self.probability = p
        self.num_patches = num_patches
        self.patch_size = patch_size

    def __call__(self, image):
        if torch.rand(1) < self.probability:
            for _ in range(self.num_patches):
                image = self._hide_patch(image)
        return image

    def _hide_patch(self, image):
        shape = image.shape
        patch_start = [
            np.random.randint(0, s - p) for s, p in zip(shape, self.patch_size)
        ]
        patch_end = [
            min(start + p, s)
            for start, p, s in zip(patch_start, self.patch_size, shape)
        ]

        image[
            patch_start[0]: patch_end[0],
            patch_start[1]: patch_end[1],
            patch_start[2]: patch_end[2],
        ] = 1  # Set the patch to 0 (or any other value to hide)
        return image


class GaussianBlur:
    def __init__(self, p=1):
        self.p = p

    def __call__(self, image):
        if torch.rand(1) < self.p:
            return gaussian_filter(image, sigma=1)
        else:
            return image


class SobelFilter3D:
    def __init__(self, p=0.5):
        self.probability = p
        self.sobel_kernels = self._create_sobel_kernels()

    def __call__(self, image):
        if torch.rand(1) < self.probability:
            image = self._apply_sobel_filter(image)
        return image

    @staticmethod
    def _create_sobel_kernels():
        sobel_x = torch.tensor(
            [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
             [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
             [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]
        ).float().unsqueeze(0).unsqueeze(0)

        sobel_y = torch.tensor(
            [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
             [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
             [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]
        ).float().unsqueeze(0).unsqueeze(0)

        sobel_z = torch.tensor(
            [[[-1, -1, -1], [-2, -2, -2], [-1, -1, -1]],
             [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
             [[1, 1, 1], [2, 2, 2], [1, 1, 1]]]
        ).float().unsqueeze(0).unsqueeze(0)

        return sobel_x, sobel_y, sobel_z

    def _apply_sobel_filter(self, image):
        sobel_x, sobel_y, sobel_z = self.sobel_kernels

        image = torch.tensor(image).float().unsqueeze(0).unsqueeze(0)

        grad_x = F.conv3d(image, sobel_x, padding=1)
        grad_y = F.conv3d(image, sobel_y, padding=1)
        grad_z = F.conv3d(image, sobel_z, padding=1)

        gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2)

        return gradient_magnitude.squeeze().numpy()


class MinMaxNormalize:
    def __init__(self, p=1):
        """
        Args:
            p (float, optional): The probability of applying the normalization. Defaults to 1.
        """
        self.p = p

    def __call__(self, image):
        if torch.rand(1) < self.p:
            min_val = torch.min(image)
            max_val = torch.max(image)
            normalized_image = (image - min_val) / (max_val - min_val)
            return normalized_image
        else:
            return image


class PairTransform:
    def __init__(self, config: Config):

        self._config = config

        self.type_transform = self.config["type_transform"]

        custom_transformations = {
            "Rotation": Rotate3D(p=0, angles=(90,), axes=(0, 1)),
            "Cutout": Cutout3D(p=1, num_patches=100, patch_size=(10, 10, 10)),
            "GaussianBlur": GaussianBlur(p=1),
            "SobelFilter3D": SobelFilter3D(p=1),
            "GaussianNoise": GaussianNoise(p=1, mean=0, std=6935.238201002077*10e-2)
        }

        if isinstance(self.type_transform, str):
            self.type_transform = [self.type_transform]
        elif not isinstance(self.type_transform, list):
            raise ValueError("type_transforms should be a list of strings or a string.")

        selected_transforms = []
        self.type_transform = self.type_transform[0].split(', ')
        for transform_name in self.type_transform:
            if not isinstance(transform_name, str):
                raise ValueError("Each element in type_transforms should be a string.")
            if transform_name not in custom_transformations:
                raise ValueError(f"Invalid transform name: {transform_name}")
            selected_transforms.append(custom_transformations[transform_name])

        self.transform = transforms.Compose(
            [
                *selected_transforms,
                transforms.ToTensor(),
                transforms.Normalize(config.mean, config.std),
                MinMaxNormalize()
            ]
        )

        self.totensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(config.mean, config.std),
                MinMaxNormalize()
            ]
        )

    def __call__(self, sample):
        xi = self.totensor(sample)
        xj = self.transform(sample)
        return [xi, xj]

    @property
    def config(self):
        return self._config.__dict__
