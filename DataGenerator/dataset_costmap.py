
from sim import Costmap
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import List
from torch.utils.data.dataset import Dataset

class CostmapDataset(Dataset):

    def __init__(self, n_samples = 1000000, H=64, W=64, max_num_obstacles=3, min_total_obstacles=3, min_num_obstacles=0):
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
        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, obstacle_classes: List[str]):
        
        obstacles_by_class:Dict[str, List[Dict[str, int]]] = {} # obstacle_class -> List of all obstacle of that type, in that list is a sequence of dictionaries that have position and radius

        size = 0
        #Generate obstacles
        for obs in obstacle_classes:
            obstacles_by_class[obs] = [{'pos': np.random.randint(low=0, high=self.W, size=2, dtype=int), 'rad': np.random.randint(low=1,high=3, size=1, dtype=int)} for _ in range(np.random.randint(low=self.min_num_obstacles,high= self.max_num_obstacles, dtype=int))] 
            size += len(obstacles_by_class) 



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
        distances = np.linalg.norm(obs_coord - self.goal, axis=1)
        
        while(np.any(distances<4)):
            self.goal = np.random.randint(size=2, dtype=int)
        

        cm = Costmap()
        cm.goal = self.goal
        costmaps, occupancy_maps = cm.calculateCost(obstacles_by_class)
        #Build conditioning vectors

        #BINARY occupancy map
        #goal state
        #radius


        return x0


if __name__ == "__main__":
   cm_data = CostmapDataset()   
   cm_data.__getitem__(["chair", "table"]) 
