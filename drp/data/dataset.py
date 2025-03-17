import os
import random
import pandas as pd
import numpy as np
import tiledb

from torch.utils.data import Dataset
from drp.utils.memmap import DataModality, load_cube
from drp.utils.config import Config
from torchvision.transforms import Normalize, Compose, ToTensor


def find_class(permeability):

    a = 0.0219018985028516
    b = 0.0420434923086744
    c = 0.0853263918362798
    d = 0.158618981690047
    if permeability < a:
        return 0
    elif a <= permeability < b:
        return 1
    elif b <= permeability < c:
        return 2
    elif c <= permeability < d:
        return 3
    else:
        return 4


def find_position(groups, id_cube):
    for i, values in enumerate(groups.values()):
        if id_cube in values:
            return i


def get_indices(size, seed=1):
    random.seed(seed)
    indices = list(range(size))
    random.shuffle(indices)

    return indices


class Drp3dBaseDataset(Dataset):
    labels = ["labels"]

    def __init__(self, config: Config, train_flag="Train", finetune=False):
        self.root = config.root_path
        self.train_flag = train_flag
        self.train_test_ratio = config.train_test_ratio
        self.dim = config.dim
        self.transform = Compose([
            ToTensor(),
            Normalize(config.mean, config.std)
        ])

        self.volumes = config.volumes
        self.dataset = None
        self.csv_pattern = config.csv_pattern
        if self.train_flag == "Test":
            csv_pattern_test = "volume_{volume}_[100, 100, 100]_random.csv"
            volumes_test = [4419, 4420, 4421, 4422, 4423, 4424, 4435, 4436, 4437, 4438, 4439, 4440, 4443, 4444, 4445,
                            4446, 4448, 4451, 4452, 4454, 4455, 4456, 4475, 4476, 4477, 4478, 4479, 4480, 4483, 4484,
                            4485, 4486, 4487, 4488, 4499, 4501, 4503, 4504, 4507, 4508, 4509, 4510, 4511, 4512, 4515,
                            4516, 4517, 4518, 4519, 4520]
            for v in volumes_test:
                df = pd.read_csv(os.path.join(self.root, str(v), csv_pattern_test.format(volume=v)))
                if self.dataset is None:
                    self.dataset = df
                else:
                    self.dataset = pd.concat([self.dataset, df]).reset_index(drop=True)
            self.train_test_ratio = 1
        else:
            for v in self.volumes:
                df = pd.read_csv(os.path.join(self.root, str(v), self.csv_pattern.format(volume=v)))
                if self.dataset is None:
                    self.dataset = df
                else:
                    self.dataset = pd.concat([self.dataset, df]).reset_index(drop=True)

        indices = get_indices(self.dataset.shape[0], seed=config.seed)
        train_len = int(self.train_test_ratio * len(indices))

        if finetune:
            if train_flag == "Train":
                self.keys = indices[:train_len]
            elif train_flag == "Valid":
                self.keys = indices[int(0.8 * len(indices)):]

        else:
            if train_flag == "Train":
                self.keys = indices[:int(train_len)]
            elif train_flag == "Valid":
                self.keys = indices[:int(train_len)]
            else:
                self.keys = indices[:int(train_len)]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        pass

    @staticmethod
    def output_size():
        return len(Drp3dBaseDataset.labels)


class Drp3dMMapDataset(Drp3dBaseDataset):
    def __init__(self, config: Config, train_flag, finetune=False, type_model="idx"):
        super().__init__(config, train_flag, finetune)
        self.type_model = type_model

    def __getitem__(self, idx):
        row = self.dataset.loc[self.keys[idx]]
        cube_id = int(row["cube_id"])
        offset = (int(row["x"]), int(row["y"]), int(row["z"]))

        minicube = load_cube(
            os.path.join(self.root, str(cube_id)),
            DataModality.GLV,
            offset=offset,
            subshape=[self.dim, self.dim, self.dim],
        )

        minicube = self.transform(minicube.astype(np.float32))
        labels = np.array([row[l] for l in Drp3dBaseDataset.labels]).astype(np.float32)
        class_range = find_class(labels)
        if self.type_model == "idx":
            return minicube.unsqueeze(0), labels, cube_id
        elif self.type_model == "class_range":
            return minicube.unsqueeze(0), labels, class_range
        elif self.type_model == "class_idx":
            groups = [4419, 4420, 4421, 4422, 4423, 4424, 4443, 4444, 4445, 4446, 4448, 4475, 4476, 4477, 4478,
                      4479, 4480, 4483, 4484, 4485, 4486, 4487, 4488, 4499, 4501, 4503, 4504, 4515, 4516, 4517,
                      4518, 4519, 4520]
            return minicube.unsqueeze(0), labels, groups.index(cube_id)
        else:
            groups = {
                "RTX1": [4419, 4420, 4421, 4422, 4423, 4424],
                #"RTX2": [4435, 4436, 4437, 4438, 4439, 4440],
                "RTX3": [4443, 4444, 4445, 4446, 4448],
                #"RTX4": [4451, 4452, 4454, 4455, 4456],
                "RTX5": [4475, 4476, 4477, 4478, 4479, 4480],
                "RTX6": [4483, 4484, 4485, 4486, 4487, 4488],
                "RTX7": [4499, 4501, 4503, 4504],
                #"RTX8": [4507, 4508, 4509, 4510, 4511, 4512],
                "RTX9": [4515, 4516, 4517, 4518, 4519, 4520]
             }
            idx = find_position(groups, cube_id)
            return minicube.unsqueeze(0), labels, idx


class Drp3dMMapDatasetSSL(Drp3dBaseDataset):
    def __init__(self, config: Config, train_flag="Train", transform=None, finetune=True):
        super().__init__(config, train_flag, finetune)
        self.transform = transform

    def __getitem__(self, idx):
        row = self.dataset.loc[self.keys[idx]]
        cube_id = int(row["cube_id"])
        offset = (int(row["x"]), int(row["y"]), int(row["z"]))

        minicube = load_cube(
            os.path.join(self.root, str(cube_id)),
            DataModality.GLV,
            offset=offset,
            subshape=[self.dim, self.dim, self.dim],
        )

        minicube = self.transform(minicube.astype(np.float32))
        labels = np.array([row[l] for l in Drp3dBaseDataset.labels]).astype(np.float32)

        return [minicube[0].unsqueeze(0), minicube[1].unsqueeze(0)], labels, cube_id
