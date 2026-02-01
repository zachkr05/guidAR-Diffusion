import torch
import torch.nn.functional as F

class SoftGridPlanner(torch.nn.Module):
    def __init__(self, iters=80, tau=1.0, step_cost=0.1):
        """
        Differentiable Soft Value Iteration Planner.
        
        Args:
            iters (int): Number of Value Iteration steps. (Like limiting A* search depth).
                         Needs to be roughly H + W to propagate across the map.
            tau (float): Temperature. 
                         Low (0.1) = Sharp, A*-like, but hard to train.
                         High (5.0) = Blurry, diffuse, easier gradients.
            step_cost (float): Cost added for every movement (like g-score in A*).
        """
        super().__init__()
        self.iters = iters
        self.tau = tau
        self.step_cost = step_cost

    def forward(self, cost_map, goal_yx):
        """
        Args:
            cost_map: (B, 1, H, W) tensor. Higher values = obstacles.
            goal_yx: (B, 2) tensor of [y, x] coordinates.
        Returns:
            visitation: (B, 1, H, W) tensor. Map of probability of visiting each cell.
        """
        B, C, H, W = cost_map.shape
        device = cost_map.device
        
        # --- 1. Setup Goal ---
        # Create a map where Goal = 1, everywhere else = 0
        goal_mask = torch.zeros_like(cost_map)
        gy = goal_yx[:, 0].long().clamp(0, H-1)
        gx = goal_yx[:, 1].long().clamp(0, W-1)
        batch_idx = torch.arange(B, device=device)
        goal_mask[batch_idx, 0, gy, gx] = 1.0

        # --- 2. Initialize Values ---
        # V represents "Cost to go to goal". 
        # Goal has 0 cost. Everywhere else starts at Infinity.
        INF = 1e4 
        V = torch.full_like(cost_map, INF)
        V = V * (1.0 - goal_mask) 

        # Helper to shift images to get neighbors: [Up, Down, Left, Right]
        def get_neighbors(val, pad_val=INF):
            up    = F.pad(val[..., 1:, :], (0,0,0,1), value=pad_val)
            down  = F.pad(val[..., :-1, :], (0,0,1,0), value=pad_val)
            left  = F.pad(val[..., :, 1:], (0,1,0,0), value=pad_val)
            right = F.pad(val[..., :, :-1], (1,0,0,0), value=pad_val)
            return torch.stack([up, down, left, right], dim=1).squeeze(2)

        # --- 3. Soft Value Iteration (The "A*" part) ---
        # We iteratively update the cost of every cell based on its neighbors.
        # Instead of V(s) = min(neighbors), we use softmin.
        for _ in range(self.iters):
            V_neigh = get_neighbors(V, pad_val=INF)    # (B, 4, H, W)
            C_neigh = get_neighbors(cost_map, pad_val=INF) # Cost of stepping onto neighbor
            
            # The cost to go from here is: Step Cost + Neighbor's Cost + Neighbor's Value
            Q = self.step_cost + C_neigh + V_neigh
            
            # Softmin update: V_new = -tau * log( sum( exp(-Q/tau) ) )
            V_update = -self.tau * torch.logsumexp(-Q / self.tau, dim=1, keepdim=True)
            
            # Reset goal to 0 (it acts as a "sink" or "ground")
            V = (1.0 - goal_mask) * V_update + goal_mask * 0.0

        # --- 4. Compute Policy (The arrows) ---
        # Calculate probability of moving to each neighbor based on which is cheapest.
        V_neigh = get_neighbors(V, pad_val=INF)
        C_neigh = get_neighbors(cost_map, pad_val=INF)
        Q = self.step_cost + C_neigh + V_neigh
        
        # Softmax gives us a probability distribution over [Up, Down, Left, Right]
        pi = F.softmax(-Q / self.tau, dim=1) # (B, 4, H, W)

        # --- 5. Rollout (The simulation) ---
        # Start with "probability mass" at [0,0] and flow it through the map using pi.
        visitation = torch.zeros_like(cost_map)
        visitation[..., 0, 0] = 1.0 # Start condition
        
        current_density = visitation.clone()
        
        # Propagate mass for H+W steps
        for _ in range(H + W):
            # Calculate how much mass moves in each direction
            p_up    = current_density * pi[:, 0:1]
            p_down  = current_density * pi[:, 1:2]
            p_left  = current_density * pi[:, 2:3]
            p_right = current_density * pi[:, 3:4]

            # Shift the mass to the receiving cells
            # Note: If mass moved UP, it arrives from the DOWN side.
            d_from_down = F.pad(p_up[..., :-1, :], (0,0,1,0))
            d_from_up   = F.pad(p_down[..., 1:, :], (0,0,0,1))
            d_from_right= F.pad(p_left[..., :, :-1], (1,0,0,0))
            d_from_left = F.pad(p_right[..., :, 1:], (0,1,0,0))
            
            # Sum up new density
            current_density = d_from_down + d_from_up + d_from_right + d_from_left
            
            # Mass that hits the goal disappears (absorbed) so it doesn't loop forever
            current_density = current_density * (1.0 - goal_mask)
            
            # Add to total visitation history
            visitation = visitation + current_density

        # Normalize 
        visitation = visitation / (visitation.sum() + 1e-8)
        return visitation
