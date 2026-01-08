import numpy as np
import skimage.graph

# Orientation definitions (in radians, 0 = facing right/east)
ORIENTATIONS = {
    0: {'name': 'east',  'angle': 0.0},           # →
    1: {'name': 'north', 'angle': np.pi / 2},     # ↑
    2: {'name': 'west',  'angle': np.pi},         # ←
    3: {'name': 'south', 'angle': 3 * np.pi / 2}, # ↓
}

NUM_ORIENTATIONS = len(ORIENTATIONS)

OBSTACLE_CLASSES = {
    0: {'name': 'chair',  'amp': 10.0, 'sigma': 2.5},
    1: {'name': 'table',  'amp': 10.0, 'sigma': 3.0},
    2: {'name': 'person', 'amp': 10.0, 'sigma': 5.0},
    3: {'name': 'wall',   'amp': 10.0, 'sigma': 2.5},
}

NUM_CLASSES = len(OBSTACLE_CLASSES)


def orientation_to_vector(orientation_id):
    """Convert orientation ID to unit vector [dx, dy]."""
    angle = ORIENTATIONS[orientation_id]['angle']
    return np.array([np.cos(angle), np.sin(angle)])


def orientation_to_sincos(orientation_id):
    """Convert orientation ID to (sin, cos) tuple for encoding."""
    angle = ORIENTATIONS[orientation_id]['angle']
    return np.sin(angle), np.cos(angle)


def calculateAttractivePotential(k_att, goalPosition, point):
    return k_att * np.linalg.norm(point - goalPosition)**2


def calculate_repulsive_potential_multiclass(point, obstacles_by_class):
    """
    Repulsive potential with different amp/sigma per class.

    Args:
        point (array-like): shape (2,), [x, y].
        obstacles_by_class (dict): {class_id: [{'pos': (r, c), 'orientation': int}, ...], ...}

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

        # Extract positions (orientation not used for cost currently)
        positions = [obs['pos'] for obs in obstacles]
        obs = np.asarray(positions, dtype=float).reshape(-1, 2)
        diff = obs - p
        d2 = np.einsum('ij,ij->i', diff, diff)
        total += float(np.sum(amp * np.exp(-0.5 * d2 / (sigma ** 2))))

    return total


def make_geodesic_costmap(H, W, obstacles_binary_map, goal):
    """
    Creates a 'flood fill' map where value = distance to goal walking around walls.
    """
    # 1. Create a cost array for movement
    costs = np.ones((H, W))
    costs[obstacles_binary_map > 0] = 1000.0 
    
    # 2. Use MCP (Minimum Cost Path) / Dijkstra
    mcp = skimage.graph.MCP(costs, fully_connected=True)
    
    # 3. Calculate distance from every pixel TO the goal
    cumulative_costs, _ = mcp.find_costs(starts=[goal])
    
    # 4. Clip for visualization
    cumulative_costs = np.clip(cumulative_costs, 0, 200)

    min_val = cumulative_costs.min()
    max_val = cumulative_costs.max()
    
    # Handle flat map case
    if max_val == min_val:
        return np.zeros((H, W), dtype=np.float32)

    # Normalize to -1 to 1 for Diffusion
    norm_dist = (cumulative_costs - min_val) / (max_val - min_val + 1e-8)
    return (1.0 - norm_dist) * 2.0 - 1.0


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
            obstacles_by_class (dict): {class_id: [{'pos': (r, c), 'orientation': int}, ...], ...}

        Returns:
            np.ndarray: [H, W] costmap
        """
        self.obstacles_by_class = obstacles_by_class
        
        # Create coordinate grids
        rows, cols = np.ogrid[:self.H, :self.W]
        binary_map = np.zeros((self.H, self.W), dtype=bool)        

        # Build binary obstacle map
        for class_id, obstacles in obstacles_by_class.items():
            if len(obstacles) == 0:
                continue
            sigma = OBSTACLE_CLASSES[class_id]['sigma'] 
            
            for obs in obstacles:
                r, c = obs['pos']
                dist_sqrt = (rows - r)**2 + (cols - c)**2
                mask = dist_sqrt <= (sigma**2)
                binary_map[mask] = True

        reward_map = make_geodesic_costmap(self.H, self.W, binary_map, self.goal)

        self.cost = reward_map
        return -self.cost
