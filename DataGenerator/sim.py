import numpy as np
import skimage.graph
import matplotlib.pyplot as plt

class Costmap:
    def __init__(self, H=64, W=64):
        self.H = H
        self.W = W
        self.cost = np.zeros((H, W), dtype=np.float32)
        self.obstacles = []
        self.obstacles_byclass = {}
        self.robot = [H, W]
        self.goal = [10, 10]

    def calculateCostmaps(self, costmaps, occupancy_map):
    
        mcps = {}
        max_reasonable = (self.H + self.W)  * 2# Max possible path lengthv


        
        for key, value in occupancy_map.items():
            
            current_cost = costmaps[key].copy().astype(np.float32) +1.0
       

            mcps[key] = np.zeros((self.H, self.W))


            #occupancy_map[i] = 
            current_cost[occupancy_map[key] > 0] = 1000

            mcp = skimage.graph.MCP(current_cost, fully_connected=True)
            cumulative_costs, _ = mcp.find_costs(starts=[self.goal]) 
            cumulative_costs[np.isinf(cumulative_costs)] = max_reasonable*2

            cumulative_costs = np.clip(cumulative_costs, 0, max_reasonable * 2)
            
            min_val = np.min(cumulative_costs)
            max_val = np.max(cumulative_costs)
            
            denom = max_val -min_val
            if denom< 1e-5:
                denom = 1

            norm_dist = (cumulative_costs - min_val) / (denom)
            
            mcps[key] = -((1.0 - norm_dist) * 2.0 - 1.0).astype(np.float32)

        return mcps

    def calculateCost(self, obstacles_by_class: dict):
        """
        """


        self.obstacles_by_class = obstacles_by_class
        
        rows, cols = np.ogrid[:self.H, :self.W]
        num_classes = len(obstacles_by_class)
        
        occupancy_map = {}
        binary_occupancy_map = {}
        costmaps = {}
        for idx, (key, obstacle) in enumerate(obstacles_by_class.items()):
            #print(key)
            
            occupancy_map[key] = np.zeros((self.H, self.W))
            costmaps[key] = np.zeros((self.H, self.W))
            binary_occupancy_map[key] = np.zeros((self.H, self.W))

            for item in obstacle:
                r, c = item['pos']
                radius = item['rad']
                dist_sqrt = (rows - r)**2 + (cols - c)**2
                mask = (dist_sqrt <= radius**2)
                occupancy_map[key][mask] = 1
                binary_occupancy_map[key][r,c] = 1 
        
        final_mcps = self.calculateCostmaps(costmaps, occupancy_map)
        
        self.visualize_cm(final_mcps, occupancy_map)

        return final_mcps, occupancy_map, binary_occupancy_map

    def visualize_cm(self, final_mcps, occupancy_map):
        #figures = [plt.figure(num=i, figsize=(self.H, self.W)) for i in range(len(occupancy_map))]
        
        for key, value in final_mcps.items():
            plt.imshow(final_mcps[key], cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.show()

        #print(figures)
    

if __name__ == "__main__":
    cm = Costmap()

    data = {
        'chair': [{'pos': (0, 1), 'rad': 2}, {'pos': (2, 2), 'rad': 2}],
        'table': [{'pos': (1, 1), 'rad': 2}, {'pos': (0, 1), 'rad': 2}]  # Note: (0, 1) is a duplicate
    }
    cm.calculateCostMapMulticlassVectorized(data)
    #cm.calculateCostMapMulticlassVetorized(data)
