
from .sim import Costmap
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import List
from torch.utils.data.dataset import Dataset
from typing import List, Dict

class CostmapDataset(Dataset):

    def __init__(self, n_samples = 1000000, H=128, W=128, max_num_obstacles=3, min_total_obstacles=3, min_num_obstacles=0):
        self.H = H
        self.W = W
        self.cost = np.zeros((H, W), dtype=np.float32)
        self.obstacles = []
        self.obstacles_byclass = {}
        self.robot = [H, W]
        self.goal = np.array([10, 10])
        self.n_samples = n_samples
        self.min_num_obstacles = min_num_obstacles #min num of obstacles per class
        self.max_num_obstacles = max_num_obstacles
        self.obstacle_classes: List[str] = None

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        
        obstacle_classes = self.obstacle_classes
        assert self.obstacle_classes is not None

        obstacles_by_class:Dict[str, List[Dict[str, int]]] = {} # obstacle_class -> List of all obstacle of that type, in that list is a sequence of dictionaries that have position and radius

        for obs in obstacle_classes:
            obstacles_by_class[obs] = [{'pos': np.random.randint(low=10, high=self.W-7, size=2, dtype=int), 'rad': int(np.random.randint(low=1,high=3))} for _ in range(np.random.randint(low=self.min_num_obstacles,high= self.max_num_obstacles, dtype=int))] 

        rows, cols = np.ogrid[:self.H, :self.W]
        occupancy_map = np.zeros((self.H, self.W))
        for idx, (key, obstacle) in enumerate(obstacles_by_class.items()):
            for item in obstacle:
                r, c = item['pos']
                radius = item['rad']
                dist_sqrt = (rows - r)**2 + (cols - c)**2
                mask = (dist_sqrt <= radius**2)
                occupancy_map[mask] = 1

        indices_tuple = np.array(np.nonzero(occupancy_map))
        obs_coord = indices_tuple.T
        
        self.goal = np.array([
                np.random.randint(low=self.W-7, high = self.H, dtype=int),
                np.random.randint(low=self.W-7, high = self.W, dtype=int),
                ])

        cm = Costmap(H=self.H, W=self.W)
        cm.goal = self.goal
        costmaps, radii_maps, binary_occupancy_map = cm.calculateCost(obstacles_by_class)
        
        goal_map = np.zeros((self.H, self.W), dtype=np.float32)
        goal_map[self.goal[0], self.goal[1]] = 1.0
        goal_t = torch.from_numpy(goal_map).float().unsqueeze(0)
        
        keys = list(costmaps.keys())
        keys_to_i = {k: i for i, k in enumerate(keys)}

        bin_stack = torch.stack([torch.from_numpy(binary_occupancy_map[k]).float() for k in keys], dim=0)
        rad_stack = torch.stack([torch.from_numpy(radii_maps[k]).float() for k in keys], dim=0)
        
        features = {}
        targets = {}


        for key, cm in (costmaps.items()):
            channel_list = []
            i = keys_to_i[key]

            
            curr_bin = bin_stack[i:i+1]
            curr_rad = rad_stack[i:i+1]
            
            other_bin = torch.cat([bin_stack[:i], bin_stack[i+1:]], dim=0) 
            other_rad = torch.cat([rad_stack[:i], rad_stack[i+1:]], dim=0) 
            cost_np = costmaps[key].copy()
           
            x = torch.cat([curr_bin, curr_rad, other_bin, other_rad, goal_t], dim =0)
            features[key] = x
            
            targets[key] = torch.from_numpy(cost_np).float().unsqueeze(0)

        positions = {}
        radii = {}
        for cls, obstacles in obstacles_by_class.items():
            positions[cls] = [tuple(obs['pos']) for obs in obstacles]
            radii[cls] = [obs['rad'] for obs in obstacles]


        return features, targets #, positions, radii, self.goal


if __name__ == "__main__":
   cm_data = CostmapDataset() 
   cm_data.obstacle_classes = ["chair", "table"]
   cm_data.__getitem__(1) 
