import numpy as np

OBSTACLE_CLASSES = {
    0: {'name': 'chair',  'amp': 10.0, 'sigma': 2.5},
    1: {'name': 'table',  'amp': 10.0, 'sigma': 3.0},
    2: {'name': 'person', 'amp': 10.0, 'sigma': 5.0},
    3: {'name': 'wall',   'amp': 10.0, 'sigma': 2.5},
}

NUM_CLASSES = len(OBSTACLE_CLASSES)

def calculateAttractivePotential(k_att, goalPosition, point):
    return k_att * np.linalg.norm(point - goalPosition)**2

def calculate_repulsive_potential_multiclass(point, obstacles_by_class):
    """
    Repulsive potential with different amp/sigma per class.

    Args:
        point (array-like): shape (2,), [x, y].
        obstacles_by_class (dict): {class_id: [(r, c), ...], ...}

    Returns:
        float: scalar potential U(p).
    """
    p = np.asarray(point, dtype=float).reshape(2,)
    total = 0.0

    for class_id, obstacles in obstacles_by_class.items():
        if len(obstacles) == 0:
            continue

        props = OBSTACLE_CLASSES[class_id]
        amp = props['amp']
        sigma = props['sigma']

        obs = np.asarray(obstacles, dtype=float).reshape(-1, 2)
        diff = obs - p
        d2 = np.einsum('ij,ij->i', diff, diff)
        total += float(np.sum(amp * np.exp(-0.5 * d2 / (sigma ** 2))))

    return total

import skimage.graph
import numpy as np

def make_geodesic_costmap(H, W, obstacles_binary_map, goal):
    """
    Creates a 'flood fill' map where value = distance to goal walking around walls.
    """
    # 1. Create a cost array for movement
    # Walking on empty space = cost 1
    # Walking on obstacle = cost Infinity (or very high)
    costs = np.ones((H, W))
    costs[obstacles_binary_map > 0] = 1000.0 
    
    # 2. Use MCP (Minimum Cost Path) / Dijkstra
    mcp = skimage.graph.MCP(costs, fully_connected=True)
    
    # 3. Calculate distance from every pixel TO the goal
    # cumulative_costs is the map we want
    cumulative_costs, _ = mcp.find_costs(starts=[goal])
    
    # 4. Handle the obstacles (they will have massive values, clamp them)
    cumulative_costs = np.clip(cumulative_costs, 0, 200) # Clip for visualization
   

    cumulative_costs = np.clip(cumulative_costs, 0, 200)

    min_val = cumulative_costs.min()
    max_val = cumulative_costs.max()
    # Normalize to -1 to 1 for Diffusion
    norm_map = (cumulative_costs - cumulative_costs.min()) / (cumulative_costs.max() - cumulative_costs.min())
# FIX: Check if map is flat to avoid 0/0
    if max_val == min_val:
        # If map is flat (e.g., unreachable), return a zero map or error
        return np.zeros((H, W), dtype=np.float32)

    # FIX: Add 1e-8 to denominator prevents crash if difference is tiny
    norm_dist = (cumulative_costs - min_val) / (max_val - min_val + 1e-8)

    return (1.0 - norm_dist) * 2.0 - 1.0

    return norm_map * 2 - 1


class Costmap:
    def __init__(self, H=64, W=64):
        self.H = H
        self.W = W
        self.cost = np.zeros((H, W), dtype=np.float32)
        self.obstacles = []
        self.obstacles_by_class = {}
        self.robot = [H, W]
        self.goal = [0, 0]

    def calculateCostMapMulticlassVectorized(self, obstacles_by_class: dict):
        """
        Args:
            obstacles_by_class (dict): {class_id: [(r, c), ...], ...}

        Returns:
            np.ndarray: [H, W] costmap
        """
        self.obstacles_by_class = obstacles_by_class
        
        # Create coordinate grids
        rows, cols = np.ogrid[:self.H, :self.W]
        binary_map = np.zeros((self.H, self.W), dtype = bool)        
        k_att = 0.5

        # Goal attractive potential (quadratic)
        goal_r, goal_c = self.goal
        attractive = k_att * ((rows - goal_r)**2 + (cols - goal_c)**2)
        
        # Repulsive potential per class
        repulsive = np.zeros((self.H, self.W), dtype=np.float32)
        
        for class_id, obstacles in obstacles_by_class.items():
            if len(obstacles) == 0:
                continue
            sigma = OBSTACLE_CLASSES[class_id]['sigma'] 
            #props = OBSTACLE_CLASSES[class_id]
            #amp = props['amp']
            #sigma = props['sigma']
            
            for (r, c) in obstacles:
                dist_sqrt = (rows - r)**2 + (cols - c)**2
                mask = dist_sqrt <= (sigma**2)
                #   repulsive += amp * np.exp(-0.5 * dist_sq / (sigma ** 2))
                binary_map[mask] = True

        reward_map = make_geodesic_costmap(self.H, self.W, binary_map, self.goal)

        # 3. Store and Return
        self.cost = reward_map
        return -self.cost

        #self.cost = (repulsive + attractive).astype(np.float32)
        #return -self.cost
