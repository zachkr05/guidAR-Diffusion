import numpy as np
import torch
from torch.utils.data import Dataset
from .sim import Costmap, OBSTACLE_CLASSES, NUM_CLASSES


def make_goal_map(H, W, goal_coord, sigma=5.0):
    """Vectorized Gaussian blob for goal."""
    r, c = goal_coord
    rows, cols = np.ogrid[:H, :W]
    dist_sq = (rows - r)**2 + (cols - c)**2
    return np.exp(-dist_sq / (2 * sigma**2)).astype(np.float32)


def make_occupancy(H, W, obstacles_rc, sigma=3.0):
    """Vectorized Gaussian blobs at each obstacle (single class)."""
    if len(obstacles_rc) == 0:
        return np.zeros((H, W), dtype=np.float32)
    
    rows, cols = np.ogrid[:H, :W]
    occ = np.zeros((H, W), dtype=np.float32)
    
    for r, c in obstacles_rc:
        dist_sq = (rows - r)**2 + (cols - c)**2
        occ = np.maximum(occ, np.exp(-dist_sq / (2 * sigma**2)))
    
    return occ.astype(np.float32)


def make_class_occupancy_maps(H, W, obstacles_by_class):
    """
    Create separate occupancy channel for each class.
    
    Args:
        H, W: dimensions
        obstacles_by_class: {class_id: [(r, c), ...], ...}
    
    Returns:
        np.ndarray: [NUM_CLASSES, H, W] - one channel per class
    """
    rows, cols = np.ogrid[:H, :W]
    occ = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
    
    for class_id, obstacles in obstacles_by_class.items():
        if len(obstacles) == 0:
            continue
        
        # Use class-specific sigma for conditioning too
        sigma = OBSTACLE_CLASSES[class_id]['sigma']
        
        for (r, c) in obstacles:
            dist_sq = (rows - r)**2 + (cols - c)**2
            blob = np.exp(-dist_sq / (2 * sigma**2))
            occ[class_id] = np.maximum(occ[class_id], blob)
    
    return occ.astype(np.float32)

class MultiClassCostmapDataset(Dataset):
    """
    Dataset with multiple obstacle classes (chair, table, person, wall).
    
    Each class has different amp/sigma, creating different cost patterns.
    Conditioning has one channel per class + goal channel.
    
    Conditioning shape: [NUM_CLASSES + 1, H, W]
        - Channels 0 to NUM_CLASSES-1: obstacle classes
        - Channel NUM_CLASSES: goal
    """
    
    def __init__(self, n_samples=100000, H=64, W=64, n_obs_per_class=3, 
                 min_obs_per_class=0, min_total_obs=1):
        """
        Args:
            n_samples: number of samples
            H, W: grid dimensions
            n_obs_per_class: max obstacles per class
            min_obs_per_class: min obstacles per class
            min_total_obs: minimum total obstacles across all classes
        """
        self.n_samples = n_samples
        self.H, self.W = H, W
        self.n_obs_per_class = n_obs_per_class
        self.min_obs_per_class = min_obs_per_class
        self.min_total_obs = min_total_obs

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        H, W = self.H, self.W

        # Random goal
        goal = (np.random.randint(0, H), np.random.randint(0, W))

        # Generate obstacles for each class
        obstacles_by_class = {}
        total_obs = 0
        
        for class_id in range(NUM_CLASSES):
            n_obs = np.random.randint(self.min_obs_per_class, self.n_obs_per_class + 1)
            obstacles = []
            for _ in range(n_obs):
                r = np.random.randint(0, H)
                c = np.random.randint(0, W)
                obstacles.append((r, c))
            obstacles_by_class[class_id] = obstacles
            total_obs += len(obstacles)
        
        # Ensure minimum total obstacles
        while total_obs < self.min_total_obs:
            class_id = np.random.randint(0, NUM_CLASSES)
            r = np.random.randint(0, H)
            c = np.random.randint(0, W)
            obstacles_by_class[class_id].append((r, c))
            total_obs += 1

        # Generate costmap
        cm = Costmap(H, W)
        cm.goal = np.array(goal, dtype=np.float32)
        cost = cm.calculateCostMapMulticlassVectorized(obstacles_by_class)

        # Normalize to [-1, 1]
        cost = cost.astype(np.float32)
        cost01 = (cost - cost.min()) / (cost.max() - cost.min() + 1e-8)
        x0 = cost01 * 2.0 - 1.0
        x0 = x0[None, :, :]  # [1, H, W]

        # Create conditioning: one channel per class + goal
        class_maps = make_class_occupancy_maps(H, W, obstacles_by_class)  # [NUM_CLASSES, H, W]
        goal_map = make_goal_map(H, W, goal)  # [H, W]
        
        cond = np.concatenate([class_maps, goal_map[None, :, :]], axis=0)  # [NUM_CLASSES+1, H, W]

        return torch.from_numpy(cond), torch.from_numpy(x0), obstacles_by_class, cm.goal

