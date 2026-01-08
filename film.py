"""
Class-Isolated Preference Adapter (Residual Delta Version).

CHANNEL STACK (Total: 3 + 2×NUM_CLASSES):
=============================================================
Index | Name              | Purpose
------|-------------------|--------------------------------
0     | Base Costmap      | "What does the world look like?"
1     | Start Map         | "Where am I coming from?"
2     | Goal Map          | "Where am I going?"
3     | Class 0 EDF       | Distance to nearest Chair
4     | Class 0 Angle     | Chair Facing Angle (normalized 0-1)
5     | Class 1 EDF       | Distance to nearest Table
6     | Class 1 Angle     | Table Facing Angle
...   | ...               | Repeat for all classes

CHANGES FROM V2 (FiLM):
- Removed "Gating" (Network learns localization from EDF input).
- Changed Output to "Delta" (Additive correction) instead of Gamma/Beta.
- Improved Stability (No fighting against decay factors).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from scipy import ndimage

# =============================================================================
# Pre-Computed Feature Maps
# =============================================================================

def compute_edf(obstacle_positions, H, W):
    """
    Compute Euclidean Distance Field for a single class.
    Returns: [H, W] normalized distance field (0 at obstacles, 1 at max distance)
    """
    if len(obstacle_positions) == 0:
        return np.ones((H, W), dtype=np.float32)
    
    # Create binary obstacle mask
    mask = np.zeros((H, W), dtype=bool)
    for r, c in obstacle_positions:
        r, c = int(np.clip(r, 0, H-1)), int(np.clip(c, 0, W-1))
        mask[r, c] = True
    
    # Distance transform
    dist = ndimage.distance_transform_edt(~mask)
    
    # Normalize by diagonal
    max_dist = np.sqrt(H**2 + W**2)
    return (dist / max_dist).astype(np.float32)


def compute_orientation_map(obstacle_positions, obstacle_orientations, H, W, sigma=5.0):
    """
    Compute Orientation Angle Map.
    """
    if len(obstacle_positions) == 0:
        return np.zeros((H, W), dtype=np.float32)
    
    # Orientation to angle mapping (Matches your sim.py)
    ORIENTATION_ANGLES = {
        0: 0.0,             # East  -> 0.0
        1: np.pi / 2,       # North -> 0.25
        2: np.pi,           # West  -> 0.50
        3: 3 * np.pi / 2    # South -> 0.75
    }

    angle_map = np.zeros((H, W), dtype=np.float32)
    weight_map = np.zeros((H, W), dtype=np.float32)
    rows, cols = np.ogrid[:H, :W]
    
    for (obs_r, obs_c), orient_id in zip(obstacle_positions, obstacle_orientations):
        angle_rad = ORIENTATION_ANGLES.get(orient_id, 0.0)
        norm_angle = angle_rad / (2 * np.pi) # Normalize 0 to 1
        
        # Gaussian weight around obstacle
        dist_sq = (rows - obs_r)**2 + (cols - obs_c)**2
        weight = np.exp(-dist_sq / (2 * sigma**2))
        
        # Accumulate weighted angle
        angle_map += weight * norm_angle
        weight_map += weight
    
    # Normalize
    mask = weight_map > 1e-6
    angle_map[mask] /= weight_map[mask]
    
    return angle_map.astype(np.float32)


def make_gaussian_map(H, W, position, sigma=5.0):
    """Create a Gaussian blob at the given position."""
    r, c = position
    rows, cols = np.ogrid[:H, :W]
    dist_sq = (rows - r)**2 + (cols - c)**2
    return np.exp(-dist_sq / (2 * sigma**2)).astype(np.float32)


def build_feature_stack(
    H, W,
    obstacles_by_class,
    start_pos,
    goal_pos,
    base_costmap,
    num_classes=4
):
    """
    Build the Clean feature stack.
    Returns: [3 + 2*num_classes, H, W] numpy array
    """
    channels = []
    
    # === Global Context (3 channels) ===
    channels.append(base_costmap)
    channels.append(make_gaussian_map(H, W, start_pos))
    channels.append(make_gaussian_map(H, W, goal_pos))
    
    # === Per-Class Features (2 channels each: EDF, Angle) ===
    for class_id in range(num_classes):
        obstacles = obstacles_by_class.get(class_id, [])
        
        if len(obstacles) > 0:
            positions = [obs['pos'] for obs in obstacles]
            orientations = [obs['orientation'] for obs in obstacles]
        else:
            positions = []
            orientations = []
        
        # 1. EDF
        edf = compute_edf(positions, H, W)
        channels.append(edf)
        
        # 2. Angle (Orientation)
        angle_map = compute_orientation_map(positions, orientations, H, W)
        channels.append(angle_map)
    
    return np.stack(channels, axis=0).astype(np.float32)


# =============================================================================
# Neural Network Architecture
# =============================================================================

class ClassSpecificBlock(nn.Module):
    """
    Processing block for a single obstacle class.
    
    RESIDUAL UPDATE:
    - Inputs: Global Context + [Class EDF, Class Angle]
    - Output: Single channel 'Delta' (Additive Correction).
    """
    def __init__(self, global_channels=3, class_channels=2, hidden_dim=32):
        super().__init__()
        
        in_channels = global_channels + class_channels
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            # CHANGED: Output 1 channel (Delta) instead of 2 (Gamma, Beta)
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1) 
        )
        
        # Initialize to zero so we start with Identity (No change)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(self, global_features, class_features, class_edf):
        """
        Args:
            global_features: [B, 3, H, W]
            class_features: [B, 2, H, W] - EDF, Angle
            class_edf: [B, 1, H, W] - Still passed for consistency, but logic uses class_features
        """
        x = torch.cat([global_features, class_features], dim=1)
        out = self.net(x) # [B, 1, H, W]
        
        # === NO HARD GATING ===
        # We rely on the network to learn gating from the input EDF (inside class_features).
        # This allows the "influence" to be as wide as the user demonstrates.
        return out


class ClassIsolatedPreferenceAdapter(nn.Module):
    """
    Pure summation of additive deltas.
    """
    
    def __init__(self, num_classes=4, global_channels=3, class_channels=2, hidden_dim=32):
        super().__init__()
        
        self.num_classes = num_classes
        self.global_channels = global_channels
        self.class_channels = class_channels
        
        # Separate network for each class
        self.class_blocks = nn.ModuleList([
            ClassSpecificBlock(global_channels, class_channels, hidden_dim)
            for _ in range(num_classes)
        ])
        
    def forward(self, features):
        B, C, H, W = features.shape
        
        # Extract global context
        global_features = features[:, :self.global_channels, :, :]
        
        class_outputs = []
        for c in range(self.num_classes):
            start_idx = self.global_channels + c * self.class_channels
            end_idx = start_idx + self.class_channels
            
            # Extract specific class features
            class_features = features[:, start_idx:end_idx, :, :]
            class_edf = features[:, start_idx:start_idx+1, :, :]
            
            class_out = self.class_blocks[c](global_features, class_features, class_edf)
            class_outputs.append(class_out)
        
        # Sum all deltas
        output = sum(class_outputs)
        
        return output


# =============================================================================
# Online Preference Learner (Residual Delta Version)
# =============================================================================

class OnlinePreferenceLearner:
    def __init__(self, num_classes=4, device="cuda", lr=0.005, buffer_size=10):
        self.device = device
        self.buffer_size = buffer_size
        self.num_classes = num_classes
        
        self.global_channels = 3
        self.class_channels = 2   # EDF + Angle (No Frontalness)
        self.total_channels = self.global_channels + self.class_channels * num_classes
        
        self.buffer = []
        
        self.model = ClassIsolatedPreferenceAdapter(
            num_classes=num_classes,
            global_channels=self.global_channels,
            class_channels=self.class_channels,
            hidden_dim=32
        ).to(device)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.training_count = 0
    
    def build_features(self, H, W, obstacles_by_class, start_pos, goal_pos, base_costmap_np):
        features_np = build_feature_stack(
            H, W, obstacles_by_class, start_pos, goal_pos,
            base_costmap_np, self.num_classes
        )
        return torch.from_numpy(features_np).float().unsqueeze(0).to(self.device)
    
    def predict(self, features):
        """
        API COMPATIBILITY MODE:
        Returns (delta, dummy_zeros) so eval.py thinks it's getting (gamma, beta).
        """
        self.model.eval()
        with torch.no_grad():
            delta = self.model(features)
            
            # Create dummy zeros for API compatibility
            dummy_beta = torch.zeros_like(delta)
            
        return delta, dummy_beta
    
    def apply_modulation(self, base_costmap_tensor, delta, dummy_beta=None):
        """
        RESIDUAL LOGIC: Base + Delta.
        Ignores dummy_beta (which is passed by eval.py).
        """
        modulated = base_costmap_tensor + delta
        return torch.clamp(modulated, 0, 1)

    # ... (Rest of training logic) ...

    def augment_batch(self, features, targets):
        """Augmentation via spatial shifts."""
        shifts = [(0, 0), (4, 4), (-4, -4), (4, -4), (-4, 4)]
        
        aug_features = []
        aug_targets = []
        
        for i in range(features.shape[0]):
            f_i = features[i:i+1]
            t_i = targets[i:i+1]
            
            selected = [shifts[0]] + random.sample(shifts[1:], min(3, len(shifts)-1))
            
            for dr, dc in selected:
                aug_features.append(torch.roll(f_i, shifts=(dr, dc), dims=(-2, -1)))
                aug_targets.append(torch.roll(t_i, shifts=(dr, dc), dims=(-2, -1)))
        
        return torch.cat(aug_features), torch.cat(aug_targets)

    def train_on_edit(self, features, base_costmap_tensor, target_costmap_tensor,
                      iterations=50, verbose=True):
        self.model.train()
        
        features = features.detach()
        base = base_costmap_tensor.detach()
        target = target_costmap_tensor.detach()
        
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append((features, base, target))
        
        for i in range(iterations):
            self.optimizer.zero_grad()
            
            # Sample from buffer
            current = [self.buffer[-1]]
            history = []
            if len(self.buffer) > 1:
                history = random.sample(self.buffer[:-1], min(len(self.buffer)-1, 3))
            samples = current + history
            
            b_features = torch.cat([s[0] for s in samples])
            b_target = torch.cat([s[2] for s in samples])
            
            # Augment
            b_features, b_target = self.augment_batch(b_features, b_target)
            
            # Extract base costmap (Channel 0)
            b_base_aug = b_features[:, 0:1, :, :]
            
            # === CHANGED: PREDICT DELTA ===
            delta = self.model(b_features)
            
            # Apply residual (base + delta)
            predicted = self.apply_modulation(b_base_aug, delta)
            
            # Loss
            diff = (predicted - b_target) ** 2
            
            # Weight the edited regions higher
            change_mask = torch.abs(b_target - b_base_aug) > 0.05
            weights = torch.ones_like(diff)
            weights[change_mask] = 20.0
            
            # Regularize Delta magnitude
            loss = (diff * weights).mean() + 0.001 * (delta**2).mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
        self.training_count += 1
        return loss.item()

    def get_class_weights_norm(self):
        norms = []
        for block in self.model.class_blocks:
            norm = sum(p.norm().item() for p in block.parameters())
            norms.append(norm)
        return norms


# =============================================================================
# Convenience Functions for Integration
# =============================================================================

def create_learner(num_classes=4, device="cuda", lr=0.005, buffer_size=10):
    """Factory function to create a new learner."""
    return OnlinePreferenceLearner(
        num_classes=num_classes,
        device=device,
        lr=lr,
        buffer_size=buffer_size
    )


# For backward compatibility
SpatialPreferenceAdapter = ClassIsolatedPreferenceAdapter
