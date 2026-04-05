import numpy as np
import skimage.graph
import matplotlib
matplotlib.use("tkAgg")   # or "Qt5Agg"

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
#        if other_mask is not None:
 #           cost[other_mask] = -1.0
        return cost

    def calculateCostmaps(self, occupancy_map, sigma=6.0):
        mcps = {}

        combined_mask = np.any(np.stack(list(occupancy_map.values())) > 0, axis=0)
        #print(combined_mask)
        for key in occupancy_map:
            own_mask = occupancy_map[key] > 0
            other_mask = combined_mask & (~own_mask)

            cost = Costmap.cost_from_mask_gaussian(
                own_mask=own_mask,
                sigma=sigma,
                other_mask=other_mask
            )

            mcps[key] = cost
        #print(mcps)
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
        
        final_mcps = self.calculateCostmaps(occupancy_map)
       
        

        #self.visualize_cm(final_mcps, occupancy_map)

        return final_mcps, occupancy_map, binary_occupancy_map, sin_angle_maps, cos_angle_maps
    
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
                    radius = float(radius)
                    
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
            plt.savefig(f"costmap_{key}.png")
            plt.close()

if __name__ == "__main__":
    cm = Costmap()

    data = {
        'chair': [{'pos': (0, 20), 'rad': 2}, {'pos': (50, 7), 'rad': 2}],
        'table': []#{'pos': (15, 12), 'rad': 2}, {'pos': (60, 7), 'rad': 2}]  # Note: (0, 1) is a duplicate
    }
    cm.calculateCost(data)
    #cm.calculateCostMapMulticlassVetorized(data)
