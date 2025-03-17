import os

import numpy as np

from drp.utils.memmap import DataModality, load_cube
import pandas as pd
from ast import literal_eval
from drp.utils.generate_offset import generate_offset


def to_csv(root, uniform=True, nb_point=None):

    for idx_cube in [x for x in os.listdir(root)]:

        path = os.path.join(root, idx_cube)
        k_heatmap = load_cube(
                path, modality=DataModality.K_HEATMAP, offset=None, subshape=None
            )
        phi_heatmap = load_cube(
            path, modality=DataModality.PHI_HEATMAP, offset=None, subshape=None
        )
        df = pd.DataFrame(columns=["x", "y", "z", "porosity", "permeability", "labels"])

        if uniform:
            for x in range(0, 2700, 100):
                for y in range(0, 1000, 100):
                    for z in range(0, 1000, 100):
                        k_mD = k_heatmap[x, y, z]
                        phi = phi_heatmap[x, y, z]
                        k_volxel = mD_to_volxel(k_mD)
                        row = pd.DataFrame({
                            "x": [x],
                            "y": [y],
                            "z": [z],
                            "porosity": [phi],
                            "permeability": [k_mD],
                            "labels": [k_volxel]
                        })
                        df = pd.concat([df, row], axis=0)
        else:
            for i in range(nb_point):
                roi_file = next(
                    filter(
                        lambda path: str("K_HEATMAP").lower() in path.lower()
                                     and path.lower().endswith(".dat"),
                        os.listdir(path),
                    ),
                    None,
                )
                assert (
                        roi_file is not None
                ), f"Memmap data not found for modality K_HEATMAP and root {path}"

                cube_params = roi_file.split("-")
                cube_shape = literal_eval(cube_params[2].split(".")[0])
                offset = generate_offset(fullshape=cube_shape, subshape=(100, 100, 100))

                k_mD = k_heatmap[offset[0], offset[1], offset[2]]
                phi = phi_heatmap[offset[0], offset[1], offset[2]]
                k_volxel = mD_to_volxel(k_mD)
                row = pd.DataFrame({
                    "x": [offset[0]],
                    "y": [offset[1]],
                    "z": [offset[2]],
                    "porosity": [phi],
                    "permeability": [k_mD],
                    "labels": [k_volxel]
                })

                df = pd.concat([df, row], axis=0, ignore_index=True)

        df.to_csv(path + f"/volume_{idx_cube}_[100, 100, 100]_random_100_2024-01-23.csv", index=False)
        print(f"File volume_{idx_cube}_[100, 100, 100]_random_100_2024-01-23.csv is created. It includes {df.shape[0]} rows")


import os
import pandas as pd


# Define groups of rock samples
groups = {
    "RTX1": [4419, 4420, 4421, 4422, 4423, 4424],
    "RTX2": [4435, 4436, 4437, 4438, 4439, 4440],
    "RTX3": [4443, 4444, 4445, 4446, 4448],
    "RTX4": [4451, 4452, 4454, 4455, 4456],
    "RTX5": [4475, 4476, 4477, 4478, 4479, 4480],
    "RTX6": [4483, 4484, 4485, 4486, 4487, 4488],
    "RTX7": [4499, 4501, 4503, 4504],
    "RTX8": [4507, 4508, 4509, 4510, 4511, 4512],
    "RTX9": [4515, 4516, 4517, 4518, 4519, 4520]
}


# Function to randomly sample 10% of data points from a CSV file
def sample_csv(input_filepath, output_filepath, num_samples=3000):
    # Load the CSV file
    data = pd.read_csv(input_filepath)
    # Randomly sample 3000 points
    sampled_data = data.sample(n=num_samples, random_state=42)
    # Save the sampled data to a new CSV file
    sampled_data.to_csv(output_filepath, index=False)





def mD_to_volxel(k_mD):
    resolution = 0.18283724910288787
    return k_mD/(resolution*resolution*1013.25)


if __name__ =="__main__":
    # to_csv("C:/Users/nguyenva/Downloads/data_DRP", uniform=False, nb_point=1000)
    # Base directory where the files are stored and output will be saved
    input_directory = "C:/Users/nguyenva/Documents/data_DRP"
    output_directory = "C:/Users/nguyenva/Documents/data_DRP"
    os.makedirs(output_directory, exist_ok=True)

    # Process each volume in each group
    for group, volumes in groups.items():
        for volume in volumes:
            input_filename = f"volume_{volume}_[100, 100, 100]_random.csv"
            output_filename = f"volume_{volume}_[100, 100, 100]_random_1k.csv"

            input_filepath = os.path.join(input_directory, input_filename)
            output_filepath = os.path.join(output_directory, output_filename)

            # Ensure the input file exists before sampling
            if os.path.exists(input_filepath):
                sample_csv(input_filepath, output_filepath)
            else:
                print(f"Input file not found: {input_filepath}")

