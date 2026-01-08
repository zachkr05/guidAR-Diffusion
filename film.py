"""
Class-Isolated Preference Adapter with Pre-Computed Semantic Features.

CHANNEL STACK (Total: 3 + 3×NUM_CLASSES = 15 for 4 classes):
=============================================================
Index | Name                  | Purpose
------|-----------------------|--------------------------------
0     | Base Costmap          | "What does the world look like?"
1     | Start Map             | "Where am I coming from?"
2     | Goal Map              | "Where am I going?"
3     | Class 0 EDF           | Distance to nearest Chair (0=at chair, 1=far)
4     | Class 0 Frontalness   | "Am I in front of a Chair?" (pre-computed cone)
5     | Class 0 Orientation   | Facing direction encoded as angle or vector magnitude
6     | Class 1 EDF           | Distance to nearest Table
7     | Class 1 Frontalness   | "Am I in front of a Table?"
8     | Class 1 Orientation   | Table facing direction
...   | ...                   | Repeat for all classes

WHY THIS WORKS FOR CLASS ISOLATION:
===================================
When user corrects path near a Chair:
  - Chair_EDF at edit location ≈ 0.2 (close)
  - Table_EDF at edit location ≈ 0.9 (far)
  
Gradient flows to Chair weights (high signal), ignores Table weights (low signal).
No cross-class contamination!

WHY FRONTALNESS INSTEAD OF SIN/COS:
===================================
- Sin/Cos requires the network to learn trigonometry to understand "in front"
- Frontalness is PRE-COMPUTED: bright in the cone projecting from object front
- Network instantly learns: "High cost when Chair_Frontalness is high"
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from scipy import ndimage


# =============================================================================
# Pre-Computed Feature Maps
# =============================================================================

def compute_edf(obstacle_positions, H, W):
    """
    Compute Euclidean Distance Field for a single class.
    
    Args:
        obstacle_positions: List of (row, col) tuples
        H, W: Grid dimensions
    
    Returns:
        edf: [H, W] normalized distance field (0 at obstacles, 1 at max distance)
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


def compute_frontalness(obstacle_positions, obstacle_orientations, H, W, 
                        cone_angle=np.pi/3, max_distance=30.0):
    """
    Compute "Frontalness" map - lights up in the cone projecting from object fronts.
    
    Args:
        obstacle_positions: List of (row, col) tuples
        obstacle_orientations: List of orientation IDs (0=East, 1=North, 2=West, 3=South)
        H, W: Grid dimensions
        cone_angle: Half-angle of the frontal cone (default 60° = π/3)
        max_distance: How far the cone extends
    
    Returns:
        frontalness: [H, W] map (1.0 = directly in front, 0.0 = behind or far)
    """
    if len(obstacle_positions) == 0:
        return np.zeros((H, W), dtype=np.float32)
    
    # Orientation to angle mapping
    ORIENTATION_ANGLES = {
        0: 0.0,           # East  →
        1: np.pi / 2,     # North ↑
        2: np.pi,         # West  ←
        3: 3 * np.pi / 2  # South ↓
    }
    
    frontalness = np.zeros((H, W), dtype=np.float32)
    rows, cols = np.ogrid[:H, :W]
    
    for (obs_r, obs_c), orient_id in zip(obstacle_positions, obstacle_orientations):
        obs_r, obs_c = float(obs_r), float(obs_c)
        facing_angle = ORIENTATION_ANGLES.get(orient_id, 0.0)
        
        # Vector from obstacle to each pixel
        dy = rows - obs_r  # Positive = below obstacle
        dx = cols - obs_c  # Positive = right of obstacle
        
        # Distance to obstacle
        dist = np.sqrt(dx**2 + dy**2) + 1e-6
        
        # Angle from obstacle to each pixel
        # Note: atan2(y, x), and we negate dy because row increases downward
        angle_to_pixel = np.arctan2(-dy, dx)
        
        # Angular difference from facing direction
        angle_diff = np.abs(np.arctan2(
            np.sin(angle_to_pixel - facing_angle),
            np.cos(angle_to_pixel - facing_angle)
        ))
        
        # In the cone?
        in_cone = angle_diff < cone_angle
        
        # Distance falloff (1 at obstacle, 0 at max_distance)
        distance_weight = np.clip(1.0 - dist / max_distance, 0, 1)
        
        # Angular falloff (1 at center of cone, 0 at edges)
        angular_weight = np.clip(1.0 - angle_diff / cone_angle, 0, 1)
        
        # Combine: must be in cone AND close
        contribution = in_cone * distance_weight * angular_weight
        
        # Max with existing (multiple obstacles of same class)
        frontalness = np.maximum(frontalness, contribution)
    
    return frontalness.astype(np.float32)


def compute_orientation_magnitude(obstacle_positions, obstacle_orientations, H, W, sigma=5.0):
    """
    Compute orientation as a simple directional magnitude map.
    
    Encodes the facing direction as a smooth field around obstacles.
    Uses the dot product with a reference direction (East) for simplicity.
    
    Args:
        obstacle_positions: List of (row, col)
        obstacle_orientations: List of orientation IDs
        H, W: Grid dimensions
        sigma: Gaussian spread
    
    Returns:
        orient_map: [H, W] values from -1 to 1 indicating facing direction component
    """
    if len(obstacle_positions) == 0:
        return np.zeros((H, W), dtype=np.float32)
    
    ORIENTATION_VECTORS = {
        0: (1.0, 0.0),    # East  →
        1: (0.0, -1.0),   # North ↑ (negative row direction)
        2: (-1.0, 0.0),   # West  ←
        3: (0.0, 1.0)     # South ↓
    }
    
    orient_map = np.zeros((H, W), dtype=np.float32)
    weight_map = np.zeros((H, W), dtype=np.float32)
    rows, cols = np.ogrid[:H, :W]
    
    for (obs_r, obs_c), orient_id in zip(obstacle_positions, obstacle_orientations):
        dx, dy = ORIENTATION_VECTORS.get(orient_id, (1.0, 0.0))
        
        # Gaussian weight around obstacle
        dist_sq = (rows - obs_r)**2 + (cols - obs_c)**2
        weight = np.exp(-dist_sq / (2 * sigma**2))
        
        # Encode orientation as the x-component of facing direction
        # (Could also use angle, but this is simpler and continuous)
        orient_map += weight * dx
        weight_map += weight
    
    # Normalize
    mask = weight_map > 1e-6
    orient_map[mask] /= weight_map[mask]
    
    return orient_map.astype(np.float32)


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
    Build the complete feature stack for the preference adapter.
    
    Args:
        H, W: Grid dimensions
        obstacles_by_class: {class_id: [{'pos': (r,c), 'orientation': int}, ...]}
        start_pos: (row, col) tuple
        goal_pos: (row, col) tuple
        base_costmap: [H, W] numpy array (0-1 normalized)
        num_classes: Number of obstacle classes
    
    Returns:
        features: [3 + 3*num_classes, H, W] numpy array
    """
    channels = []
    
    # === Global Context (3 channels) ===
    # 0: Base Costmap
    channels.append(base_costmap)
    
    # 1: Start Map
    channels.append(make_gaussian_map(H, W, start_pos))
    
    # 2: Goal Map
    channels.append(make_gaussian_map(H, W, goal_pos))
    
    # === Per-Class Features (3 channels each) ===
    for class_id in range(num_classes):
        obstacles = obstacles_by_class.get(class_id, [])
        
        if len(obstacles) > 0:
            positions = [obs['pos'] for obs in obstacles]
            orientations = [obs['orientation'] for obs in obstacles]
        else:
            positions = []
            orientations = []
        
        # EDF: Distance to nearest obstacle of this class
        edf = compute_edf(positions, H, W)
        channels.append(edf)
        
        # Frontalness: Am I in front of this class?
        frontalness = compute_frontalness(positions, orientations, H, W)
        channels.append(frontalness)
        
        # Orientation: Which way is this class facing?
        orientation = compute_orientation_magnitude(positions, orientations, H, W)
        channels.append(orientation)
    
    return np.stack(channels, axis=0).astype(np.float32)


def build_feature_stack_torch(
    H, W,
    obstacles_by_class,
    start_pos,
    goal_pos,
    base_costmap_tensor,
    num_classes=4,
    device="cuda"
):
    """
    Torch wrapper for build_feature_stack.
    
    Args:
        base_costmap_tensor: [B, 1, H, W] or [1, H, W] tensor
    
    Returns:
        features: [B, 3 + 3*num_classes, H, W] tensor
    """
    # Handle batched or unbatched input
    if base_costmap_tensor.dim() == 3:
        base_costmap_tensor = base_costmap_tensor.unsqueeze(0)
    
    B = base_costmap_tensor.shape[0]
    base_np = base_costmap_tensor[:, 0].cpu().numpy()
    
    # Build features for each batch item
    # (In practice, usually B=1 for online learning)
    all_features = []
    for b in range(B):
        features = build_feature_stack(
            H, W, obstacles_by_class, start_pos, goal_pos,
            base_np[b], num_classes
        )
        all_features.append(features)
    
    return torch.from_numpy(np.stack(all_features)).float().to(device)


# =============================================================================
# Neural Network Architecture
# =============================================================================

class ClassSpecificBlock(nn.Module):
    """
    Processing block for a single obstacle class.
    
    Takes the 3 class-specific channels (EDF, Frontalness, Orientation)
    plus global context, and outputs gamma/beta for this class.
    """
    def __init__(self, global_channels=3, class_channels=3, hidden_dim=32):
        super().__init__()
        
        in_channels = global_channels + class_channels
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 2, kernel_size=1)  # gamma, beta
        )
        
        # Initialize to near-zero (identity transform)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(self, global_features, class_features, class_edf):
        """
        Args:
            global_features: [B, 3, H, W] - base_costmap, start, goal
            class_features: [B, 3, H, W] - EDF, frontalness, orientation for this class
            class_edf: [B, 1, H, W] - EDF for gating (same as class_features[:,0:1])
        
        Returns:
            output: [B, 2, H, W] - gamma, beta gated by proximity to this class
        """
        x = torch.cat([global_features, class_features], dim=1)
        out = self.net(x)  # [B, 2, H, W]
        
        # Gate by inverse EDF: modifications only apply NEAR this class
        # EDF=0 at obstacle, EDF=1 far away
        # We want strong signal near obstacles, weak far away
        proximity_gate = 1.0 - class_edf  # [B, 1, H, W]
        proximity_gate = proximity_gate.clamp(min=0.0)
        
        # Apply gate
        gated_out = out * proximity_gate
        
        return gated_out


class ClassIsolatedPreferenceAdapter(nn.Module):
    """
    Preference adapter with completely isolated per-class learning.
    
    Each class has its own network that ONLY sees that class's features.
    Cross-class contamination is impossible by design.
    """
    
    def __init__(self, num_classes=4, global_channels=3, class_channels=3, hidden_dim=32):
        super().__init__()
        
        self.num_classes = num_classes
        self.global_channels = global_channels
        self.class_channels = class_channels
        
        # Separate network for each class
        self.class_blocks = nn.ModuleList([
            ClassSpecificBlock(global_channels, class_channels, hidden_dim)
            for _ in range(num_classes)
        ])
        
        # Optional: Global refinement after combining class outputs
        self.refine = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 2, kernel_size=1)
        )
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)
        
    def forward(self, features):
        """
        Args:
            features: [B, 3 + 3*num_classes, H, W]
                - [0:3] = global (base_costmap, start, goal)
                - [3:6] = class 0 (EDF, frontalness, orientation)
                - [6:9] = class 1
                - etc.
        
        Returns:
            output: [B, 2, H, W] - gamma (channel 0), beta (channel 1)
        """
        B, C, H, W = features.shape
        
        # Extract global context
        global_features = features[:, :self.global_channels, :, :]  # [B, 3, H, W]
        
        # Process each class independently
        class_outputs = []
        for c in range(self.num_classes):
            start_idx = self.global_channels + c * self.class_channels
            end_idx = start_idx + self.class_channels
            
            class_features = features[:, start_idx:end_idx, :, :]  # [B, 3, H, W]
            class_edf = features[:, start_idx:start_idx+1, :, :]   # [B, 1, H, W]
            
            class_out = self.class_blocks[c](global_features, class_features, class_edf)
            class_outputs.append(class_out)
        
        # Sum contributions from all classes
        combined = sum(class_outputs)  # [B, 2, H, W]
        
        # Optional refinement
        output = combined + self.refine(combined)
        
        return output


# =============================================================================
# Online Preference Learner
# =============================================================================

class OnlinePreferenceLearner:
    """
    Manages training of the class-isolated preference adapter.
    
    Key features:
    - Pre-computes semantic feature maps (EDF, Frontalness, Orientation)
    - Maintains replay buffer for stable learning
    - Applies augmentation that preserves class relationships
    """
    
    def __init__(self, num_classes=4, device="cuda", lr=0.005, buffer_size=10):
        self.device = device
        self.buffer_size = buffer_size
        self.num_classes = num_classes
        
        # Calculate channel counts
        self.global_channels = 3  # base_costmap, start, goal
        self.class_channels = 3   # EDF, frontalness, orientation per class
        self.total_channels = self.global_channels + self.class_channels * num_classes
        
        # Replay buffer: list of (features, target_costmap) tuples
        self.buffer = []
        
        # Model
        self.model = ClassIsolatedPreferenceAdapter(
            num_classes=num_classes,
            global_channels=self.global_channels,
            class_channels=self.class_channels,
            hidden_dim=32
        ).to(device)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.training_count = 0
        
    def build_features(self, H, W, obstacles_by_class, start_pos, goal_pos, base_costmap_np):
        """
        Build the feature stack for a single scene.
        
        Returns:
            features: [1, total_channels, H, W] tensor
        """
        features_np = build_feature_stack(
            H, W, obstacles_by_class, start_pos, goal_pos,
            base_costmap_np, self.num_classes
        )
        return torch.from_numpy(features_np).float().unsqueeze(0).to(self.device)
    
    def predict(self, features):
        """
        Get gamma/beta modifiers from pre-built features.
        
        Args:
            features: [B, total_channels, H, W]
        
        Returns:
            gamma: [B, 1, H, W]
            beta: [B, 1, H, W]
        """
        self.model.eval()
        with torch.no_grad():
            output = self.model(features)
            gamma = output[:, 0:1, :, :]
            beta = output[:, 1:2, :, :]
        return gamma, beta
    
    def predict_full(self, H, W, obstacles_by_class, start_pos, goal_pos, base_costmap_np):
        """
        Convenience method: build features and predict in one call.
        
        Returns:
            gamma: [1, 1, H, W]
            beta: [1, 1, H, W]
        """
        features = self.build_features(H, W, obstacles_by_class, start_pos, goal_pos, base_costmap_np)
        return self.predict(features)
    
    def apply_modulation(self, base_costmap_tensor, gamma, beta):
        """
        Apply FiLM modulation to base costmap.
        
        Args:
            base_costmap_tensor: [B, 1, H, W]
            gamma, beta: [B, 1, H, W]
        
        Returns:
            modulated: [B, 1, H, W] clamped to [0, 1]
        """
        modulated = base_costmap_tensor * (1.0 + gamma) + beta
        return torch.clamp(modulated, 0, 1)
    
    def augment_batch(self, features, targets):
        """
        Augmentation via spatial shifts.
        
        Since we use pre-computed features (EDF, Frontalness), shifting
        preserves the semantic meaning: "near a chair" stays "near a chair"
        after shifting both the features and target.
        """
        shifts = [
            (0, 0),
            (4, 4), (-4, -4), (4, -4), (-4, 4),
            (8, 0), (-8, 0), (0, 8), (0, -8),
        ]
        
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
        """
        Train on a single user edit.
        
        Args:
            features: [1, total_channels, H, W] - pre-built feature stack
            base_costmap_tensor: [1, 1, H, W] - diffusion output
            target_costmap_tensor: [1, 1, H, W] - desired result after user edit
            iterations: training steps
            verbose: print progress
        
        Returns:
            final_loss: float
        """
        self.model.train()
        
        features = features.detach()
        base = base_costmap_tensor.detach()
        target = target_costmap_tensor.detach()
        
        # Update replay buffer
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append((features, base, target))
        
        if verbose:
            print(f"    Training Class-Isolated Adapter ({iterations} steps). Buffer: {len(self.buffer)}")
        
        initial_loss = None
        final_loss = 0
        
        for i in range(iterations):
            self.optimizer.zero_grad()
            
            # Sample from buffer
            current = [self.buffer[-1]]
            if len(self.buffer) > 1:
                n_hist = min(len(self.buffer) - 1, 3)
                history = random.sample(self.buffer[:-1], n_hist)
                samples = current + history
            else:
                samples = current
            
            # Collate
            b_features = torch.cat([s[0] for s in samples])
            b_base = torch.cat([s[1] for s in samples])
            b_target = torch.cat([s[2] for s in samples])
            
            # Augment
            b_features, b_target = self.augment_batch(b_features, b_target)
            b_base_aug = torch.roll(b_base, shifts=(0, 0), dims=(-2, -1))  # Base also needs augmenting
            # Actually, base is in features[0], so we need to extract it
            b_base_aug = b_features[:, 0:1, :, :]  # Base costmap is channel 0
            
            # Forward
            output = self.model(b_features)
            gamma = output[:, 0:1, :, :]
            beta = output[:, 1:2, :, :]
            
            # Apply modulation
            predicted = self.apply_modulation(b_base_aug, gamma, beta)
            
            # === Loss Calculation ===
            diff = (predicted - b_target) ** 2
            
            # 1. Edit-weighted loss
            change_mask = torch.abs(b_target - b_base_aug) > 0.05
            weights = torch.ones_like(diff)
            weights[change_mask] = 25.0
            reconstruction_loss = (diff * weights).mean()
            
            # 2. Sparsity loss
            sparsity_loss = 0.01 * (gamma**2 + beta**2).mean()
            
            # 3. Per-class locality loss
            # Modifications should be stronger near their respective class
            # This is already enforced by the EDF gating in ClassSpecificBlock,
            # but we add explicit regularization too
            locality_loss = 0.0
            for c in range(self.num_classes):
                edf_idx = self.global_channels + c * self.class_channels
                class_edf = b_features[:, edf_idx:edf_idx+1, :, :]
                # Penalize modifications far from ANY obstacle
                far_mask = (class_edf > 0.8).float()
                locality_loss += 0.01 * ((gamma**2 + beta**2) * far_mask).mean()
            
            total_loss = reconstruction_loss + sparsity_loss + locality_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            if initial_loss is None:
                initial_loss = total_loss.item()
            final_loss = total_loss.item()
        
        self.training_count += 1
        if verbose:
            print(f"    Done. Loss: {initial_loss:.5f} -> {final_loss:.5f}")
        
        return final_loss
    
    def get_params_summary(self):
        """Return summary of model state."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return (f"Class-Isolated Adapter | Classes: {self.num_classes} | "
                f"Params: {trainable_params:,} | Buffer: {len(self.buffer)} | "
                f"Trained: {self.training_count}x")
    
    def get_class_weights_norm(self):
        """Get the norm of weights for each class block (for debugging)."""
        norms = []
        for c, block in enumerate(self.model.class_blocks):
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
