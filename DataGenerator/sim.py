import numpy as np

OBSTACLE_CLASSES = {
    0: {'name': 'chair',  'amp': 5.0, 'sigma': 4.0},
    1: {'name': 'table',  'amp': 7.0, 'sigma': 8.0},
    2: {'name': 'person', 'amp': 10.0, 'sigma': 5.0},
    3: {'name': 'wall',   'amp': 12.0, 'sigma': 3.0},
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
        
        # Goal attractive potential (quadratic)
        goal_r, goal_c = self.goal
        attractive = (rows - goal_r)**2 + (cols - goal_c)**2
        
        # Repulsive potential per class
        repulsive = np.zeros((self.H, self.W), dtype=np.float32)
        
        for class_id, obstacles in obstacles_by_class.items():
            if len(obstacles) == 0:
                continue
                
            props = OBSTACLE_CLASSES[class_id]
            amp = props['amp']
            sigma = props['sigma']
            
            for (r, c) in obstacles:
                dist_sq = (rows - r)**2 + (cols - c)**2
                repulsive += amp * np.exp(-0.5 * dist_sq / (sigma ** 2))
        
        self.cost = (repulsive + attractive).astype(np.float32)
        return self.cost
