import numpy as np
import skimage.graph
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
        for key, value in final_mcps.items():
            plt.figure(figsize=(10, 10))  # Create a new figure for each MCP
            plt.imshow(final_mcps[key], cmap='hot', interpolation='nearest')
            
            # Add goal circle
            circle = patches.Circle((self.goal[1], self.goal[0]), 1, color='blue', fill=False, linewidth=2)
            plt.gca().add_patch(circle)
            
            # Color map for different obstacle classes
            colors = plt.cm.Set1(range(len(self.obstacles_by_class)))
            color_idx = 0
            
            # Track which classes we've added to legend
            legend_elements = []
            
            for obstacle_class, obstacles in self.obstacles_by_class.items():
                class_color = colors[color_idx]
                color_idx += 1
                
                # Add one legend entry per class
                legend_elements.append(patches.Patch(color=class_color, label=obstacle_class))
                
                for obstacle in obstacles:
                    r, c = obstacle['pos']
                    r = float(r)
                    c = float(c)
                    radius = obstacle['rad']
                    radius = float(radius.flatten()[0])
                    
                    # Draw circle for each obstacle
                    circle = patches.Circle((c, r), radius, color=class_color, fill=False, linewidth=2)
                    plt.gca().add_patch(circle)
                    
                    # Improved text positioning - place label above the obstacle
                    label_offset = radius + 2  # Place text just outside the circle
                    plt.text(c, r - label_offset, obstacle_class, 
                            color='white',  # White text for visibility
                            fontsize=10,
                            ha='center',
                            va='bottom',
                            bbox=dict(boxstyle='round,pad=0.3', 
                                     facecolor=class_color, 
                                     alpha=0.8,
                                     edgecolor='black',
                                     linewidth=1))
            
            # Add legend with custom elements
            plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
            
            # Add goal to legend
            goal_patch = patches.Circle((0, 0), 1, color='blue', fill=False, linewidth=2)
            plt.gca().add_artist(plt.legend([goal_patch], ['Goal'], loc='upper left'))
            
            plt.colorbar()
            plt.title(f'MCP: {key}')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.show()
if __name__ == "__main__":
    cm = Costmap()

    data = {
        'chair': [{'pos': (0, 1), 'rad': 2}, {'pos': (2, 2), 'rad': 2}],
        'table': [{'pos': (1, 1), 'rad': 2}, {'pos': (0, 1), 'rad': 2}]  # Note: (0, 1) is a duplicate
    }
    cm.calculateCost(data)
    #cm.calculateCostMapMulticlassVetorized(data)
