from .sim import Costmap
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import List, Dict
from torch.utils.data.dataset import Dataset


class CostmapDataset(Dataset):
    def __init__(
        self,
        n_samples=1000000,
        H=128,
        W=128,
        max_num_obstacles=4,
        min_total_obstacles=3,
        min_num_obstacles=2,
    ):
        self._cache = {}
        self.H = H
        self.W = W
        self.cost = np.zeros((H, W), dtype=np.float32)
        self.obstacles = []
        self.obstacles_byclass = {}
        self.robot = [H, W]
        self.goal = np.array([10, 10])
        self.n_samples = n_samples
        self.min_num_obstacles = min_num_obstacles
        self.max_num_obstacles = max_num_obstacles
        self.obstacle_classes: List[str] = None

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):

        if idx in self._cache:
            return self._cache[idx]

        obstacle_classes = self.obstacle_classes
        assert self.obstacle_classes is not None

        obstacles_by_class: Dict[str, List[Dict[str, int]]] = {}

        for obs in obstacle_classes:
            obstacles_by_class[obs] = [
                {
                    "pos": np.random.randint(low=10, high=self.W - 7, size=2, dtype=int),
                    "rad": int(np.random.randint(low=1, high=3)),
                    "angle": float(np.random.uniform(0, 2 * np.pi)),
                }
                for _ in range(
                    np.random.randint(
                        low=self.min_num_obstacles,
                        high=self.max_num_obstacles,
                        dtype=int,
                    )
                )
            ]

        self.goal = np.array(
            [
                np.random.randint(low=5, high=self.H-5, dtype=int),
                np.random.randint(low=5, high=self.W-5, dtype=int),
            ]
        )

        cm = Costmap(H=self.H, W=self.W)
        cm.goal = self.goal

        trajectory_map, binary_occupancy_map, radii_maps, sin_angle_maps, cos_angle_maps = (
            cm.calculateCost(obstacles_by_class)
        )

        # Goal channel
        goal_map = np.zeros((self.H, self.W), dtype=np.float32)
        goal_map[self.goal[0], self.goal[1]] = 1.0
        goal_t = torch.from_numpy(goal_map).float().unsqueeze(0)

        keys = list(binary_occupancy_map.keys())
        keys_to_i = {k: i for i, k in enumerate(keys)}

        bin_stack = torch.stack(
            [torch.from_numpy(binary_occupancy_map[k]).float() for k in keys], dim=0
        )
        rad_stack = torch.stack(
            [torch.from_numpy(radii_maps[k]).float() for k in keys], dim=0
        )
        sin_stack = torch.stack(
            [torch.from_numpy(sin_angle_maps[k]).float() for k in keys], dim=0
        )
        cos_stack = torch.stack(
            [torch.from_numpy(cos_angle_maps[k]).float() for k in keys], dim=0
        )

        features = {}

        for key in keys:
            i = keys_to_i[key]

            curr_bin = bin_stack[i : i + 1]
            curr_rad = rad_stack[i : i + 1]
            curr_sin = sin_stack[i : i + 1]
            curr_cos = cos_stack[i : i + 1]

            other_bin = torch.cat([bin_stack[:i], bin_stack[i + 1 :]], dim=0)
            other_rad = torch.cat([rad_stack[:i], rad_stack[i + 1 :]], dim=0)
            other_sin = torch.cat([sin_stack[:i], sin_stack[i + 1 :]], dim=0)
            other_cos = torch.cat([cos_stack[:i], cos_stack[i + 1 :]], dim=0)

            # Channels: curr_bin, curr_rad, curr_sin, curr_cos,
            #           other_bin, other_rad, other_sin, other_cos, goal
            x = torch.cat(
                [
                    curr_bin, curr_rad, curr_sin, curr_cos,
                    other_bin, other_rad, other_sin, other_cos,
                    goal_t,
                ],
                dim=0,
            )
            features[key] = x

        # Single target: trajectory heatmap [1, H, W]
        target = torch.from_numpy(trajectory_map).float().unsqueeze(0)

        # In dataset.py, after creating the target
        target = torch.from_numpy(trajectory_map).float().unsqueeze(0)
        target = target * 2.0 - 1.0  # [0,1] -> [-1,1]
        positions = {}
        radii = {}
        angles = {}
        for cls, obstacles in obstacles_by_class.items():
            positions[cls] = [tuple(obs["pos"]) for obs in obstacles]
            radii[cls] = [obs["rad"] for obs in obstacles]
            angles[cls] = [obs["angle"] for obs in obstacles]


        result = (features, target, positions, radii, self.goal, angles)
        if len(self._cache) < 200:  # only cache small datasets
            self._cache[idx] = result
        return result
     #   return features, target, positions, radii, self.goal, angles
    #


if __name__ == "__main__":
    cm_data = CostmapDataset()
    cm_data.obstacle_classes = ["chair", "table"]
    features, target, positions, radii, goal, angles = cm_data.__getitem__(1)
    print("Target shape:", target.shape)
    print("Feature keys:", list(features.keys()))
    for k, v in features.items():
        print(f"  {k}: {v.shape}")
