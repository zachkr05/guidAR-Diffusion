import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialContextMasker:

    def __init__(self, feature_dim, temperature=0.07, similiarity_threshold=0.85):
        self.temperature = temperature
        self.threshold = similiarity_threshold

    def compute_context_mask(self, geometric_delta, feature_map, threshold=0.05, device='cuda'):
        B,C,H,W = feature_map.shape

        anchor_mask = (torch.abs(geometric_delta) > threshold).float()

        if anchor_mask.sum() == 0: #return empty mask
            return torch.abs(geometric_delta).unsqueeze(0).unsqueeze(0)

        features_flat = feature_map.view(B,C,-1)
        mask_flat = anchor_mask.view(-1)

        #weighted avg of delta magnitudes
        weights = torch.abs(geometric_delta).view(-1)
        weights = weights / (weights.sum() + 1e-8)

        context_vector  = torch.sum(features_flat * weights.view(1,1,-1), dim=2, keepdim=True)

        features_norm = F.normalize(features_flat, p=2, dim=1)
        context_norm = F.normalize(context_vector, p=2, dim=1)

        similarity = torch.bmm(context_norm.transpose(1, 2), features_norm)
        similarity = similarity.view(B, 1, H, W)

        similarity = (similarity - self.threshold) / self.temperature
        attention_mask = torch.sigmoid(similarity)

        geo_mask = torch.abs(geometric_delta).unsqueeze(0).unsqueeze(0).to(device)
        geo_mask = geo_mask / (geo_mask.max() + 1e-8)
        
        final_mask = torch.max(geo_mask, attention_mask * 0.5)
        
        return final_mask


def extract_features(cond_input, num_classes):
    """
    Helper to slice the conditioning tensor into specific feature sets.
    Assumes cond is [Occupancy(N), Orient(2N), EDF(N), Density(1), Goal(1)]
    """
    # We want EDF, Orientation, and Density for context matching
    # We DO NOT want Goal (that's global) or simple Occupancy (EDF is better)
    
    # Indices based on your dataset_costmap.py
    # Occ: 0 to N
    # Orient: N to 3N
    # EDF: 3N to 4N
    # Density: 4N
    
    n = num_classes
    orient_feats = cond_input[:, n:3*n, :, :]
    edf_feats = cond_input[:, 3*n:4*n, :, :]
    density_feat = cond_input[:, 4*n:4*n+1, :, :]
    
    return torch.cat([orient_feats, edf_feats, density_feat], dim=1)
