import numpy as np
import skimage.graph
import matplotlib
matplotlib.use("tkAgg")   # or "Qt5Agg"
from skimage.graph import route_through_array

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.ndimage import distance_transform_edt

class Costmap:
    def __init__(self, H=64, W=64):
        self.H = H
        self.W = W
        self.cost = np.zeros((H, W), dtype=np.float32)
        self.obstacles = []
        self.obstacles_byclass = {}
        self.robot = [H, W]
        self.goal = [10, 10]


    @staticmethod
    def cost_from_mask_gaussian(own_mask, sigma=6.0, other_mask=None):

        if not np.any(own_mask):
            cost = np.full(own_mask.shape, -1.0, dtype=np.float32)
            return cost

        d = distance_transform_edt(~own_mask).astype(np.float32)
        c01 = np.exp(-(d**2) / (2.0 * sigma**2)).astype(np.float32)  # 1 near, ~0 far
        cost = (2.0 * c01 - 1.0).astype(np.float32)
        return cost

    def calculateCostmaps(self, occupancy_map, sigma=6.0):
        mcps = {}

        combined_mask = np.any(np.stack(list(occupancy_map.values())) > 0, axis=0)
        
        for key in occupancy_map:
            own_mask = occupancy_map[key] > 0
            other_mask = combined_mask & (~own_mask)

            cost = Costmap.cost_from_mask_gaussian(
                own_mask=own_mask,
                sigma=sigma,
                other_mask=other_mask
            )

            mcps[key] = cost
        return mcps


    def calculateCost(self, obstacles_by_class: dict):

        self.obstacles_by_class = obstacles_by_class
        
        rows, cols = np.ogrid[:self.H, :self.W]
        num_classes = len(obstacles_by_class)
        
        occupancy_map = {}
        binary_occupancy_map = {}
        costmaps = {}
        sin_angle_maps = {}
        cos_angle_maps = {}

        for idx, (key, obstacle) in enumerate(obstacles_by_class.items()):
            #print(key)
            
            occupancy_map[key] = np.zeros((self.H, self.W))
            costmaps[key] = np.zeros((self.H, self.W))
            binary_occupancy_map[key] = np.zeros((self.H, self.W))
            sin_angle_maps[key] = np.zeros((self.H, self.W), dtype=np.float32)
            cos_angle_maps[key] = np.zeros((self.H, self.W), dtype=np.float32)
    

            for item in obstacle:
                r, c = item['pos']
                radius = item['rad']
                angle = item.get('angle', 0.0)

                dist_sqrt = (rows - r)**2 + (cols - c)**2
                mask = (dist_sqrt <= radius**2)
                occupancy_map[key][mask] = 1
                binary_occupancy_map[key][r,c] = 1

                sin_angle_maps[key][mask] = np.sin(angle)
                cos_angle_maps[key][mask] = np.cos(angle)
        
        final_mcps = self.calculateCostmaps(occupancy_map) #map seperated cm's 

        print(final_mcps)
        temp = [cm for _, cm in final_mcps.items()]
        full_cm = np.zeros_like(temp[0])
        for cm in temp: full_cm += cm 
        print(full_cm)

        return 
    
if __name__ == "__main__":
    cm = Costmap()

    data = {
        'chair': [{'pos': (0, 20), 'rad': 2}, {'pos': (50, 7), 'rad': 2}],
        'table': []#{'pos': (15, 12), 'rad': 2}, {'pos': (60, 7), 'rad': 2}]  # Note: (0, 1) is a duplicate
    }
    cm.calculateCost(data)
    #cm.calculateCostMapMulticlassVetorized(data)
