import numpy as np
import skimage.graph


class Costmap:
    def __init__(self, H=64, W=64):
        self.H = H
        self.W = W
        self.cost = np.zeros((H, W), dtype=np.float32)
        self.obstacles = []
        self.obstacles_byclass = {}
        self.robot = [H, W]
        self.goal = [0, 0]

    def calculateCostmaps(self, costmaps):

        mcps = np.zeros((len(costmaps)))
    # 4. Clip for visualization (handle unreachable areas)
        max_reasonable = self.H + self.W  # Max possible path lengthv
        for i in range(len(costmaps)):
            costmaps[i][occupancy_map[i] > 0] = 1000
            mcp = skimage.graph.MCP(costmaps[i], fully_connected=True)
            cumulative_costs, _ = mcp.find_costs(starts=[goal]) 
            cumulative_costs = np.clip(cumulative_costs, 0, max_reasonable * 2)
            norm_dist = (cumulative_costs - min_val) / (max_val - min_val + 1e-8)
            mcps[i] = ((1.0 - norm_dist) * 2.0 - 1.0).astype(np.float32)

        return mcps

    def calculateCostMapMulticlassVectorized(self, obstacles_by_class: dict):
        self.obstacles_by_class = obstacles_by_class
        
        rows, cols = np.ogrid[:self.H, :self.W]
        num_classes = len(obstacles_by_class)
        
        occupancy_map = np.zeros((num_classes, self.H, self.W))
        
        for idx, obstacle in enumerate(obstacles_by_class.values()):

            obstacle_rows = []
            obstacle_cols = []


            for item in obstacle:
                r, c = item['pos']
                obstacle_rows.append(r)
                obstacle_cols.append(c)

            occupancy_map[idx][obstacle_rows, obstacle_cols] = 1
        
        #print(occupancy_map)

        costmaps = np.empty((num_classes,), dtype=object)
        costmaps = [np.ogrid[0:self.H, 0:self.W] for _ in range(num_classes)]
#        print(costmaps)


        final_mcps = self.calculateCostmaps(occupancy_map)
        
        print(final_mcps)
        
        return final_mcps, occupancy_map

if __name__ == "__main__":
    cm = Costmap()

    data = {
    1: [{'pos': (0, 1)}, {'pos': (2, 2)}],
    2: [{'pos': (1, 1)}, {'pos': (0, 1)}]  # Note: (0, 1) is a duplicate
    }
    cm.calculateCostMapMulticlassVectorized(data)
    #cm.calculateCostMapMulticlassVetorized(data)
