import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset
from .sim import (Costmap, OBSTACLE_CLASSES, NUM_CLASSES, 
                  ORIENTATIONS, NUM_ORIENTATIONS, orientation_to_sincos)


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
        obstacles_by_class: {class_id: [{'pos': (r, c), 'orientation': int}, ...], ...}
    
    Returns:
        np.ndarray: [NUM_CLASSES, H, W] - one channel per class
    """
    rows, cols = np.ogrid[:H, :W]
    occ = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
    
    for class_id, obstacles in obstacles_by_class.items():
        if len(obstacles) == 0:
            continue
        
        sigma = OBSTACLE_CLASSES[class_id]['sigma']
        
        for obs in obstacles:
            r, c = obs['pos']
            dist_sq = (rows - r)**2 + (cols - c)**2
            blob = np.exp(-dist_sq / (2 * sigma**2))
            occ[class_id] = np.maximum(occ[class_id], blob)
    
    return occ.astype(np.float32)


def make_orientation_maps(H, W, obstacles_by_class):
    """
    Create orientation encoding maps using sin/cos representation.
    
    For each class, creates 2 channels: sin(theta) and cos(theta) weighted by Gaussian.
    This avoids discontinuity issues with raw angle representation.
    
    Args:
        H, W: dimensions
        obstacles_by_class: {class_id: [{'pos': (r, c), 'orientation': int}, ...], ...}
    
    Returns:
        np.ndarray: [NUM_CLASSES * 2, H, W] - sin/cos channels per class
    """
    rows, cols = np.ogrid[:H, :W]
    # 2 channels per class: sin and cos
    orient_maps = np.zeros((NUM_CLASSES * 2, H, W), dtype=np.float32)
    
    for class_id, obstacles in obstacles_by_class.items():
        if len(obstacles) == 0:
            continue
        
        sigma = OBSTACLE_CLASSES[class_id]['sigma']
        sin_channel = class_id * 2
        cos_channel = class_id * 2 + 1
        
        for obs in obstacles:
            r, c = obs['pos']
            orientation = obs['orientation']
            sin_val, cos_val = orientation_to_sincos(orientation)
            
            dist_sq = (rows - r)**2 + (cols - c)**2
            weight = np.exp(-dist_sq / (2 * sigma**2))
            
            # Weight sin/cos by Gaussian blob
            orient_maps[sin_channel] = np.maximum(
                orient_maps[sin_channel], weight * sin_val
            )
            orient_maps[cos_channel] = np.maximum(
                orient_maps[cos_channel], weight * cos_val
            )
    
    return orient_maps.astype(np.float32)

def make_edf_maps(H,W,obstacles_by_class, normalize=True):
    edf_maps = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)

    for class_id in range(NUM_CLASSES):
        obstacles = obstacles_by_class.get(class_id, [])

        if len(obstacles) == 0:
            edf_maps[class_id] = np.ones((H,W), dtype=np.float32)
            if normalize:
                pass
            else:
                edf_maps[class_id] *= np.sqrt(H**2 + W**2)
        else:
            mask = np.ones((H,W), dtype=bool)
            for obs in obstacles:
                r,c = obs['pos']
                r_idx = int(np.clip(r,0,H-1))
                c_idx = int(np.clip(c,0,W-1))
                mask[r_idx,c_idx] = False

            edf = distance_transform_edt(mask)

            if normalize:
                max_dist = np.sqrt(H**2 + W**2)
                edf = edf / max_dist
                
            edf_maps[class_id] = edf.astype(np.float32)
        
    return edf_maps


def make_density_map(H, W, obstacles_by_class, sigma=10.0):
    """
    Create local object density map.
    
    Each pixel contains a measure of how many obstacles are nearby,
    weighted by Gaussian distance.
    
    Args:
        H, W: dimensions
        obstacles_by_class: {class_id: [{'pos': (r, c), 'orientation': int}, ...], ...}
        sigma: spread of density influence
    
    Returns:
        np.ndarray: [H, W] - density map
    """
    rows, cols = np.ogrid[:H, :W]
    density = np.zeros((H, W), dtype=np.float32)
    
    # Count all obstacles weighted by distance
    for class_id, obstacles in obstacles_by_class.items():
        for obs in obstacles:
            r, c = obs['pos']
            dist_sq = (rows - r)**2 + (cols - c)**2
            density += np.exp(-dist_sq / (2 * sigma**2))
    
    # Normalize to [0, 1]
    if density.max() > 0:
        density = density / density.max()
    
    return density.astype(np.float32)

class MultiClassCostmapDataset(Dataset):
    """
    Dataset with multiple obstacle classes (chair, table, person, wall).
    Each obstacle has a position and orientation.
    
    Each class has different amp/sigma, creating different cost patterns.
    
    Conditioning shape: [NUM_CLASSES * 3 + 1, H, W]
        - Channels 0 to NUM_CLASSES-1: obstacle occupancy per class
        - Channels NUM_CLASSES to NUM_CLASSES*3-1: orientation sin/cos per class (2 per class)
        - Last channel: goal
    """
    
    def __init__(self, n_samples=100000, H=64, W=64, n_obs_per_class=3, 
                 min_obs_per_class=0, min_total_obs=1):
        """
        Args:
            n_samples: number of smples
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

    def _generate_sample(self):
        """Generate a random sample with obstacles, orientations, and goal."""
        H, W = self.H, self.W

        # Random goal
        goal = (np.random.randint(0, H), np.random.randint(0, W))

        # Generate obstacles for each class with orientations
        obstacles_by_class = {}
        total_obs = 0
        
        for class_id in range(NUM_CLASSES):
            n_obs = np.random.randint(self.min_obs_per_class, self.n_obs_per_class + 1)
            obstacles = []
            for _ in range(n_obs):
                r = np.random.randint(0, H)
                c = np.random.randint(0, W)
                orientation = np.random.randint(0, NUM_ORIENTATIONS)
                obstacles.append({
                    'pos': (r, c),
                    'orientation': orientation
                })
            obstacles_by_class[class_id] = obstacles
            total_obs += len(obstacles)
        
        # Ensure minimum total obstacles
        while total_obs < self.min_total_obs:
            class_id = np.random.randint(0, NUM_CLASSES)
            r = np.random.randint(0, H)
            c = np.random.randint(0, W)
            orientation = np.random.randint(0, NUM_ORIENTATIONS)
            obstacles_by_class[class_id].append({
                'pos': (r, c),
                'orientation': orientation
            })
            total_obs += 1

        return obstacles_by_class, goal

    def __getitem__(self, idx):
        """
        Returns:
            cond: [NUM_CLASSES * 3 + 1, H, W] - conditioning tensor
            x0: [1, H, W] - ground truth costmap
        """
        H, W = self.H, self.W
        
        obstacles_by_class, goal = self._generate_sample()

        # Generate costmap
        cm = Costmap(H, W)
        cm.goal = np.array(goal, dtype=np.float32)
        cost = cm.calculateCostMapMulticlassVectorized(obstacles_by_class)
        

        if cost.max() - cost.min() < 0.01:
            print(f"WARNING: Flat costmap! Goal={goal}, num_obstacles={sum(len(v) for v in obstacles_by_class.values())}")

        # Normalize to [-1, 1]
        cost = cost.astype(np.float32)
        cost01 = (cost - cost.min()) / (cost.max() - cost.min() + 1e-8)
        x0 = cost01 * 2.0 - 1.0
        x0 = x0[None, :, :]  # [1, H, W]

        # Create conditioning maps
        class_maps = make_class_occupancy_maps(H, W, obstacles_by_class)  # [NUM_CLASSES, H, W]
        orient_maps = make_orientation_maps(H, W, obstacles_by_class)     # [NUM_CLASSES * 2, H, W]
        goal_map = make_goal_map(H, W, goal)  # [H, W]
        edf_maps = make_edf_maps(H, W, obstacles_by_class)                # [NUM_CLASSES, H, W]
        density_map = make_density_map(H, W, obstacles_by_class)          # [H, W] 



        # Stack: occupancy + orientation + goal
        cond = np.concatenate([
            class_maps,                 # [NUM_CLASSES, H, W]
            orient_maps,                # [NUM_CLASSES * 2, H, W]
            edf_maps, 
            goal_map[None, :, :],# [1, H, W]
            density_map[None, :, :]
        ], axis=0)  # Total: [NUM_CLASSES * 3 + 1, H, W]

        return torch.from_numpy(cond), torch.from_numpy(x0)

    def get_sample_with_metadata(self, idx=None):
        """
        Get a sample with full metadata (for visualization/debugging).
        
        Returns:
            cond: [NUM_CLASSES * 3 + 1, H, W]
            x0: [1, H, W]
            obstacles_by_class: dict
            goal: tuple
        """
        H, W = self.H, self.W
        
        obstacles_by_class, goal = self._generate_sample()

        # Generate costmap
        cm = Costmap(H, W)
        cm.goal = np.array(goal, dtype=np.float32)
        cost = cm.calculateCostMapMulticlassVectorized(obstacles_by_class)

        # Normalize to [-1, 1]
        cost = cost.astype(np.float32)
        cost01 = (cost - cost.min()) / (cost.max() - cost.min() + 1e-8)
        x0 = cost01 * 2.0 - 1.0
        x0 = x0[None, :, :]

        # Create conditioning maps
        class_maps = make_class_occupancy_maps(H, W, obstacles_by_class)
        orient_maps = make_orientation_maps(H, W, obstacles_by_class)
        goal_map = make_goal_map(H, W, goal)
        edf_maps = make_edf_maps(H,W,obstacles_by_class) 
        density_map = make_density_map(H,W,obstacles_by_class) 

        cond = np.concatenate([
            class_maps,
            orient_maps,
            edf_maps,
            density_map[None, :, :],
            goal_map[None, :, :]
        ], axis=0)

        return torch.from_numpy(cond), torch.from_numpy(x0), obstacles_by_class, goal


# Convenience function to get conditioning channel count
def get_cond_channels():
    """Returns the number of conditioning channels."""
    return NUM_CLASSES * 4 + 2  # occupancy + sin/cos orientation + goal + density
