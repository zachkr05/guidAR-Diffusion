import torch
import torch.nn as nn
from .UNet import LightweightUNet

class CompositionalModel(nn.Module):
    def __init__(self, num_classes, expert_input_channels):
        super().__init__()
        self.num_classes = num_classes
        
        # Channel Math:
        # Base = x_t(1) + Occ(1) + Sin(1) + Cos(1) + Goal(1) = 5
        self.base_channels = 5
        # Context = Total - Base
        self.context_channels = expert_input_channels - self.base_channels
        
        self.experts = nn.ModuleList([
            LightweightUNet(
                in_channels=self.base_channels,
                context_channels=self.context_channels
            ) for _ in range(num_classes)
        ])

    def extract_expert_input(self, full_cond, x_t, class_idx):
        """
        Returns (base_input, context_input)
        
        Expected full_cond layout (for N=4 classes):
          [0:4]   - Occupancy maps (N channels)
          [4:12]  - Orientation Sin/Cos (N*2 channels)
          [12:16] - EDF maps (N channels)
          [16:17] - Goal (1 channel)
          [17:18] - Density (1 channel)
        
        Total: 4 + 8 + 4 + 1 + 1 = 18 channels
        """
        B, _, H, W = full_cond.shape
        N = self.num_classes
        
        # Slicing Indices (18 channels total for N=4)
        idx_occ    = 0           # Occupancy: [0:N]
        idx_orient = N           # Orientation: [N:3*N]
        idx_edf    = 3 * N       # EDF: [3*N:4*N]
        idx_goal   = 4 * N       # Goal: [4*N:4*N+1]
        idx_dens   = 4 * N + 1   # Density: [4*N+1:4*N+2]
        
        # 1. Base Features (Always Active)
        self_occ = full_cond[:, idx_occ + class_idx : idx_occ + class_idx + 1]
        self_sin = full_cond[:, idx_orient + class_idx*2 : idx_orient + class_idx*2 + 1]
        self_cos = full_cond[:, idx_orient + class_idx*2 + 1 : idx_orient + class_idx*2 + 2]
        goal     = full_cond[:, idx_goal : idx_goal + 1]
        
        # Concatenate x_t with Base Features
        base_input = torch.cat([x_t, self_occ, self_sin, self_cos, goal], dim=1)
        
        # 2. Context Features (Peer EDFs + Orient + Density)
        peer_feats = []
        for i in range(N):
            if i == class_idx: 
                continue
            # EDF for this peer (1 channel)
            peer_feats.append(full_cond[:, idx_edf + i : idx_edf + i + 1])
            # Orientation Sin/Cos for this peer (2 channels)
            peer_feats.append(full_cond[:, idx_orient + i*2 : idx_orient + i*2 + 2])
        
        #context_input = torch.cat(peer_feats, dim=1)
        
        return base_input

    def forward(self, x_t, t, cond, training_phase='base'):
        expert_outputs = []
        
        for i, expert in enumerate(self.experts):
            # 1. Extract inputs (Using Explicit Keywords to prevent TypeError)
            base_in, context_in = self.extract_expert_input(
                full_cond=cond, 
                x_t=x_t, 
                class_idx=i
            )
            
            # 2. Handle Training Logic
            if training_phase == 'base':
                # Base Training: Pass None to disable sidecar (Isolation)
                out = expert(base_in, t, context_stack=None)
            else:
                # Finetuning: Pass real context
                out = expert(base_in, t, context_stack=context_in)
                
            expert_outputs.append(out)
            
        all_costs = torch.stack(expert_outputs, dim=1)
        total_cost, _ = torch.max(all_costs, dim=1)
        
        return total_cost, all_costs
    
    def get_expert(self, class_id):
        return self.experts[class_id]
