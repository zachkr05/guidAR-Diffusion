# film_class_lora.py
"""
FiLM-Modulated Class-Specific LoRA

Instead of directly updating LoRA weights, FiLM learns to modulate LoRA
based on user trajectory edits. This allows:
1. Multiple user preferences without overwriting
2. One-shot adaptation from a single trajectory edit
3. Interpolation between different preferences

Architecture:
    Trajectory Edit → TrajectoryEncoder → FiLM (γ, β) per class
                                              ↓
                                    Modulates Class LoRA
                                              ↓
                                    Adapted Costmap
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

from DataGenerator.sim import NUM_CLASSES, OBSTACLE_CLASSES


# ==============================================================================
# Trajectory Encoder
# ==============================================================================

class TrajectoryEncoder(nn.Module):
    """
    Encodes a trajectory edit into a conditioning vector.
    
    Input: User trajectory as sequence of waypoints
    Output: Fixed-size embedding that captures the "intent" of the edit
    """
    def __init__(self, hidden_dim=128, output_dim=256, max_points=100):
        super().__init__()
        self.max_points = max_points
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Encode each waypoint (x, y) -> hidden_dim
        self.point_embed = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Positional encoding for sequence order
        self.pos_embed = nn.Parameter(torch.randn(1, max_points, hidden_dim) * 0.02)
        
        # Transformer to capture trajectory structure
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Project to output dimension
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, trajectory: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            trajectory: [B, N, 2] - waypoints normalized to [0, 1]
            mask: [B, N] - 1 for valid points, 0 for padding
        
        Returns:
            [B, output_dim] - trajectory embedding
        """
        B, N, _ = trajectory.shape
        
        # Embed points
        h = self.point_embed(trajectory)  # [B, N, hidden_dim]
        
        # Add positional encoding
        h = h + self.pos_embed[:, :N, :]
        
        # Transformer
        if mask is not None:
            # TransformerEncoder expects True = ignore
            src_key_padding_mask = (mask == 0)
        else:
            src_key_padding_mask = None
        
        h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)
        
        # Pool over sequence (mean of valid points)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)  # [B, N, 1]
            h = (h * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
        else:
            h = h.mean(dim=1)
        
        return self.output_proj(h)


class TrajectoryDifferenceEncoder(nn.Module):
    """
    Encodes the DIFFERENCE between original and user trajectory.
    
    This is more informative than just encoding the user trajectory,
    because it captures "what changed" rather than "what is".
    """
    def __init__(self, hidden_dim=128, output_dim=256, max_points=50):
        super().__init__()
        self.max_points = max_points
        
        # Encode difference vectors (user - original)
        self.diff_embed = nn.Sequential(
            nn.Linear(4, hidden_dim,  # (orig_x, orig_y, user_x, user_y)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Output
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, original_traj: torch.Tensor, user_traj: torch.Tensor, 
                mask: Optional[torch.Tensor] = None):
        """
        Args:
            original_traj: [B, N, 2] - original A* trajectory
            user_traj: [B, N, 2] - user edited trajectory
            mask: [B, N] - valid points mask
        
        Returns:
            [B, output_dim] - difference embedding
        """
        # Concatenate original and user trajectories
        combined = torch.cat([original_traj, user_traj], dim=-1)  # [B, N, 4]
        
        h = self.diff_embed(combined)
        h = self.transformer(h)
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            h = (h * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
        else:
            h = h.mean(dim=1)
        
        return self.output_proj(h)


# ==============================================================================
# FiLM Generator
# ==============================================================================

class FiLMGenerator(nn.Module):
    """
    Generates FiLM parameters (γ, β) for each class from trajectory embedding.
    
    γ (gamma): scale factor for LoRA output
    β (beta): shift/bias for LoRA output
    
    Output: LoRA_out * (1 + γ) + β
    """
    def __init__(self, cond_dim=256, num_classes=NUM_CLASSES, lora_channels=32):
        super().__init__()
        self.num_classes = num_classes
        self.lora_channels = lora_channels
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.ReLU(),
            nn.Linear(cond_dim, cond_dim)
        )
        
        # Per-class FiLM generators
        self.class_film = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cond_dim, cond_dim // 2),
                nn.ReLU(),
                nn.Linear(cond_dim // 2, lora_channels * 2)  # γ and β
            )
            for _ in range(num_classes)
        ])
        
        # Initialize to identity (γ=0, β=0 means output = LoRA_out * 1 + 0)
        for film in self.class_film:
            nn.init.zeros_(film[-1].weight)
            nn.init.zeros_(film[-1].bias)
    
    def forward(self, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            cond: [B, cond_dim] - trajectory difference embedding
        
        Returns:
            gammas: [B, num_classes, lora_channels]
            betas: [B, num_classes, lora_channels]
        """
        B = cond.shape[0]
        h = self.shared(cond)
        
        gammas = []
        betas = []
        
        for class_id in range(self.num_classes):
            film_out = self.class_film[class_id](h)  # [B, lora_channels * 2]
            gamma, beta = film_out.chunk(2, dim=-1)  # each [B, lora_channels]
            gammas.append(gamma)
            betas.append(beta)
        
        gammas = torch.stack(gammas, dim=1)  # [B, num_classes, lora_channels]
        betas = torch.stack(betas, dim=1)
        
        return gammas, betas


# ==============================================================================
# FiLM-Modulated Class LoRA
# ==============================================================================

class FiLMClassLoRA(nn.Module):
    """
    Class-specific LoRA that is modulated by FiLM parameters.
    
    Instead of updating LoRA weights directly, FiLM scales/shifts the output.
    """
    def __init__(self, channels, num_classes=NUM_CLASSES, rank=8, alpha=16.0):
        super().__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.rank = rank
        self.scaling = alpha / rank
        
        # Base LoRA per class (weights are FROZEN after pretraining)
        self.lora_A = nn.ModuleList([
            nn.Conv2d(channels, rank, kernel_size=3, padding=1, bias=False)
            for _ in range(num_classes)
        ])
        self.lora_B = nn.ModuleList([
            nn.Conv2d(rank, channels, kernel_size=1, bias=False)
            for _ in range(num_classes)
        ])
        
        # Initialize
        for i in range(num_classes):
            nn.init.kaiming_uniform_(self.lora_A[i].weight, a=5**0.5)
            nn.init.zeros_(self.lora_B[i].weight)
    
    def forward(
        self, 
        x: torch.Tensor, 
        class_activations: torch.Tensor,
        gammas: Optional[torch.Tensor] = None,
        betas: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] - feature map
            class_activations: [B, num_classes, H, W] - where each class exists
            gammas: [B, num_classes, C] - FiLM scale (optional)
            betas: [B, num_classes, C] - FiLM shift (optional)
        
        Returns:
            [B, C, H, W] - LoRA output (add to main branch)
        """
        B, C, H, W = x.shape
        output = torch.zeros_like(x)
        
        for class_id in range(self.num_classes):
            # Base LoRA computation
            lora_out = self.lora_B[class_id](self.lora_A[class_id](x))
            lora_out = lora_out * self.scaling
            
            # Apply FiLM modulation if provided
            if gammas is not None and betas is not None:
                # gammas/betas: [B, num_classes, C]
                # Need to handle channel dimension mismatch
                g = gammas[:, class_id, :]  # [B, C] or [B, lora_channels]
                b = betas[:, class_id, :]
                
                # If FiLM output dim != channels, we apply globally
                if g.shape[1] == C:
                    g = g.view(B, C, 1, 1)
                    b = b.view(B, C, 1, 1)
                else:
                    # Average FiLM params and apply as scalar
                    g = g.mean(dim=1, keepdim=True).view(B, 1, 1, 1)
                    b = b.mean(dim=1, keepdim=True).view(B, 1, 1, 1)
                
                lora_out = lora_out * (1 + g) + b
            
            # Spatial masking by class activation
            weight = class_activations[:, class_id:class_id+1, :, :]  # [B, 1, H, W]
            lora_out = lora_out * weight
            
            output = output + lora_out
        
        return output


# ==============================================================================
# Full FiLM-LoRA UNet
# ==============================================================================

class SinusoidalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.scale = (channels // num_heads) ** -0.5
        self.num_heads = num_heads
        self.channels = channels
    
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.view(B, self.num_heads, C // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W)
        
        attn = torch.einsum('bhci,bhcj->bhij', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        h = torch.einsum('bhij,bhcj->bhci', attn, v)
        h = h.view(B, C, H, W)
        
        return x + self.proj(h)


class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_ch), nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1)
        )
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch * 2))
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_ch), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x, t_emb):
        h = self.conv1(x)
        t_out = self.time_mlp(t_emb)[:, :, None, None]
        scale, shift = t_out.chunk(2, dim=1)
        h = h * (1 + scale) + shift
        return self.conv2(h) + self.residual(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, use_attn=False):
        super().__init__()
        self.conv1 = ResidualConvBlock(in_ch, out_ch, time_dim)
        self.conv2 = ResidualConvBlock(out_ch, out_ch, time_dim)
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
        self.attn = AttentionBlock(out_ch) if use_attn else None
    
    def forward(self, x, t_emb):
        h = self.conv1(x, t_emb)
        h = self.conv2(h, t_emb)
        if self.attn:
            h = self.attn(h)
        return self.down(h), h


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, use_attn=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.conv1 = ResidualConvBlock(in_ch + out_ch, out_ch, time_dim)
        self.conv2 = ResidualConvBlock(out_ch, out_ch, time_dim)
        self.attn = AttentionBlock(out_ch) if use_attn else None
    
    def forward(self, x, skip, t_emb):
        h = self.up(x)
        if h.shape != skip.shape:
            h = F.interpolate(h, size=skip.shape[2:], mode='bilinear', align_corners=True)
        h = torch.cat([h, skip], dim=1)
        h = self.conv1(h, t_emb)
        h = self.conv2(h, t_emb)
        if self.attn:
            h = self.attn(h)
        return h


class FiLMLoRAUNet(nn.Module):
    """
    UNet with FiLM-modulated class-specific LoRA.
    
    The trajectory encoder and FiLM generator are trained to map
    trajectory edits to LoRA modulations, while the base UNet and
    LoRA weights can be frozen.
    """
    def __init__(
        self,
        in_channels=6,
        out_channels=1,
        base_channels=32,
        channel_mults=(1, 2, 4, 4),
        attention_resolutions=(2, 3),
        time_dim=128,
        traj_cond_dim=256,
        num_classes=NUM_CLASSES,
        lora_rank=8
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.time_dim = time_dim
        self.base_channels = base_channels
        self.channel_mults = channel_mults
        
        # Trajectory encoder (TRAINABLE for adaptation)
        self.traj_encoder = TrajectoryDifferenceEncoder(
            hidden_dim=128,
            output_dim=traj_cond_dim,
            max_points=50
        )
        
        # FiLM generator (TRAINABLE for adaptation)
        # One FiLM generator per decoder level
        self.film_generators = nn.ModuleList([
            FiLMGenerator(traj_cond_dim, num_classes, base_channels * mult)
            for mult in reversed(channel_mults)
        ])
        self.film_mid = FiLMGenerator(traj_cond_dim, num_classes, base_channels * channel_mults[-1])
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        
        # UNet encoder (can be frozen)
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        self.downs = nn.ModuleList()
        ch = base_channels
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            use_attn = i in attention_resolutions
            self.downs.append(DownBlock(ch, out_ch, time_dim, use_attn))
            ch = out_ch
        
        # Bottleneck
        self.mid_block1 = ResidualConvBlock(ch, ch, time_dim)
        self.mid_attn = AttentionBlock(ch)
        self.mid_block2 = ResidualConvBlock(ch, ch, time_dim)
        
        # UNet decoder (can be frozen)
        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            use_attn = i in attention_resolutions
            self.ups.append(UpBlock(ch, out_ch, time_dim, use_attn))
            ch = out_ch
        
        # Final conv
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1)
        )
        
        # FiLM-modulated LoRA (LoRA weights frozen, FiLM trainable)
        self.lora_mid = FiLMClassLoRA(
            base_channels * channel_mults[-1], num_classes, lora_rank
        )
        self.lora_ups = nn.ModuleList([
            FiLMClassLoRA(base_channels * mult, num_classes, lora_rank)
            for mult in reversed(channel_mults)
        ])
    
    def forward(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor,
        original_traj: Optional[torch.Tensor] = None,
        user_traj: Optional[torch.Tensor] = None,
        traj_mask: Optional[torch.Tensor] = None
    ):
        """
        Args:
            x: [B, in_channels, H, W] - [classes, goal, noisy_costmap]
            t: [B] - timesteps
            original_traj: [B, N, 2] - original trajectory (optional)
            user_traj: [B, N, 2] - user edited trajectory (optional)
            traj_mask: [B, N] - valid points mask (optional)
        
        If trajectories are provided, FiLM modulation is applied.
        If not, standard forward pass without FiLM.
        """
        # Extract class activations
        class_activations = x[:, :self.num_classes, :, :]
        
        # Compute FiLM parameters if trajectories provided
        if original_traj is not None and user_traj is not None:
            traj_embedding = self.traj_encoder(original_traj, user_traj, traj_mask)
            
            # Get FiLM params for each level
            film_params_mid = self.film_mid(traj_embedding)
            film_params_ups = [fg(traj_embedding) for fg in self.film_generators]
        else:
            film_params_mid = (None, None)
            film_params_ups = [(None, None)] * len(self.ups)
        
        # Time embedding
        t_emb = self.time_mlp(t.float())
        
        # Encoder
        h = self.init_conv(x)
        
        skips = [h]
        for down in self.downs:
            h, skip = down(h, t_emb)
            skips.append(skip)
        
        # Bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
        
        # FiLM-modulated LoRA at bottleneck
        class_acts_mid = F.interpolate(class_activations, size=h.shape[2:], mode='bilinear', align_corners=True)
        h = h + self.lora_mid(h, class_acts_mid, *film_params_mid)
        
        # Decoder with FiLM-modulated LoRA
        for i, up in enumerate(self.ups):
            skip = skips.pop()
            h = up(h, skip, t_emb)
            
            class_acts_up = F.interpolate(class_activations, size=h.shape[2:], mode='bilinear', align_corners=True)
            h = h + self.lora_ups[i](h, class_acts_up, *film_params_ups[i])
        
        return self.final_conv(h)
    
    def freeze_base_and_lora(self):
        """Freeze UNet and LoRA weights. Only trajectory encoder + FiLM trainable."""
        for name, param in self.named_parameters():
            if 'traj_encoder' in name or 'film' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    def freeze_all_except_film(self):
        """Only FiLM generators trainable (trajectory encoder also frozen)."""
        for name, param in self.named_parameters():
            if 'film' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze everything."""
        for param in self.parameters():
            param.requires_grad = True
    
    def get_film_parameters(self):
        """Get all FiLM-related parameters."""
        params = []
        params.extend(self.traj_encoder.parameters())
        params.extend(self.film_mid.parameters())
        for fg in self.film_generators:
            params.extend(fg.parameters())
        return params
    
    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def create_film_lora_unet(lora_rank=8):
    """Create UNet with FiLM-modulated class-specific LoRA."""
    return FiLMLoRAUNet(
        in_channels=NUM_CLASSES + 2,
        out_channels=1,
        base_channels=32,
        channel_mults=(1, 2, 4, 4),
        attention_resolutions=(2, 3),
        time_dim=128,
        traj_cond_dim=256,
        num_classes=NUM_CLASSES,
        lora_rank=lora_rank
    )


# ==============================================================================
# Updated IRL Agent with FiLM
# ==============================================================================

class FiLMIRLAgent:
    """
    IRL Agent that learns to map trajectory edits to FiLM parameters.
    
    Instead of directly updating LoRA weights, we train the FiLM generator
    to produce the right modulation for any trajectory edit.
    """
    
    def __init__(
        self,
        model: FiLMLoRAUNet,
        device: str = "cuda",
        lr: float = 1e-3,
        lambda_sparse: float = 0.01,
        n_optim_steps: int = 100
    ):
        self.model = model
        self.device = device
        self.lr = lr
        self.lambda_sparse = lambda_sparse
        self.n_optim_steps = n_optim_steps
        
        # Freeze base model and LoRA, only train FiLM
        self.model.freeze_base_and_lora()
        
        # Optimizer for FiLM parameters
        self.optimizer = torch.optim.Adam(self.model.get_film_parameters(), lr=lr)
    
    def train_on_trajectory_edit(
        self,
        conditioning: torch.Tensor,
        target_costmap: torch.Tensor,
        original_trajectory: torch.Tensor,
        user_trajectory: torch.Tensor,
        traj_mask: Optional[torch.Tensor] = None,
        n_steps: int = 50,
        verbose: bool = True
    ) -> Dict:
        """
        Train FiLM to produce costmap matching target when given this trajectory edit.
        
        Args:
            conditioning: [B, NUM_CLASSES+1, H, W] - class maps + goal
            target_costmap: [B, 1, H, W] - desired costmap (from IRL)
            original_trajectory: [B, N, 2] - original A* path
            user_trajectory: [B, N, 2] - user edited path
            traj_mask: [B, N] - valid points
            n_steps: training steps
        
        Returns:
            info dict
        """
        from diffusion_utils import schedule_betas, q_sample
        
        self.model.train()
        
        # Setup diffusion
        T = 1000
        betas, alphas, alpha_bar = schedule_betas(T, 1e-4, 0.02, device=self.device)
        
        losses = []
        
        for step in range(n_steps):
            self.optimizer.zero_grad()
            
            B = target_costmap.shape[0]
            
            # Sample random timestep
            t = torch.randint(0, T, (B,), device=self.device, dtype=torch.long)
            
            # Add noise to target
            xt, noise = q_sample(target_costmap, t, alpha_bar)
            
            # Model input
            x_in = torch.cat([conditioning, xt], dim=1)
            
            # Forward with trajectory conditioning
            pred_noise = self.model(
                x_in, t,
                original_traj=original_trajectory,
                user_traj=user_trajectory,
                traj_mask=traj_mask
            )
            
            # Loss
            loss = F.mse_loss(pred_noise, noise)
            
            loss.backward()
            self.optimizer.step()
            
            losses.append(loss.item())
            
            if verbose and step % 10 == 0:
                print(f"  Step {step}: loss={loss.item():.4f}")
        
        self.model.eval()
        
        return {
            'losses': losses,
            'final_loss': losses[-1] if losses else 0.0
        }
    
    def adapt_to_user(
        self,
        base_costmap: torch.Tensor,
        conditioning: torch.Tensor,
        original_trajectory: List[Tuple[int, int]],
        user_trajectory: List[Tuple[int, int]],
        n_steps: int = 50,
        verbose: bool = True
    ) -> Dict:
        """
        High-level API: adapt model to user's trajectory preference.
        
        Args:
            base_costmap: [H, W] numpy array
            conditioning: [NUM_CLASSES+1, H, W] numpy array
            original_trajectory: list of (r, c) tuples
            user_trajectory: list of (r, c) tuples
        """
        from irl_agent import IRLAgent, resample_trajectory
        
        H, W = base_costmap.shape
        
        # Step 1: Use standard IRL to find target costmap
        if verbose:
            print("Step 1: Finding target costmap via IRL...")
        
        # Create temporary IRL agent for costmap optimization
        temp_agent = IRLAgent(
            self.model, 
            device=self.device,
            n_optim_steps=self.n_optim_steps
        )
        
        delta, mod_info = temp_agent.find_minimal_modification(
            base_costmap=base_costmap,
            user_trajectory=user_trajectory,
            class_activations=conditioning[:NUM_CLASSES],
            goal=(int(conditioning[NUM_CLASSES].argmax() // W), 
                  int(conditioning[NUM_CLASSES].argmax() % W)),
            verbose=verbose
        )
        
        target_costmap = base_costmap + delta
        target_costmap = np.clip(target_costmap, 0, None)
        
        # Normalize target
        target_norm = (target_costmap - target_costmap.min()) / (target_costmap.max() - target_costmap.min() + 1e-8)
        target_norm = target_norm * 2 - 1  # [-1, 1]
        
        # Step 2: Prepare tensors
        target_t = torch.from_numpy(target_norm).float().unsqueeze(0).unsqueeze(0).to(self.device)
        cond_t = torch.from_numpy(conditioning).float().unsqueeze(0).to(self.device)
        
        # Resample trajectories to fixed length
        orig_resampled = resample_trajectory(original_trajectory, 50)
        user_resampled = resample_trajectory(user_trajectory, 50)
        
        orig_t = torch.tensor(orig_resampled, dtype=torch.float32).unsqueeze(0).to(self.device)
        user_t = torch.tensor(user_resampled, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Normalize to [0, 1]
        orig_t[:, :, 0] /= H
        orig_t[:, :, 1] /= W
        user_t[:, :, 0] /= H
        user_t[:, :, 1] /= W
        
        # Step 3: Train FiLM
        if verbose:
            print("\nStep 2: Training FiLM generator...")
        
        train_info = self.train_on_trajectory_edit(
            conditioning=cond_t,
            target_costmap=target_t,
            original_trajectory=orig_t,
            user_trajectory=user_t,
            n_steps=n_steps,
            verbose=verbose
        )
        
        return {
            'delta_costmap': delta,
            'target_costmap': target_costmap,
            'modification_info': mod_info,
            'training_info': train_info
        }


# ==============================================================================
# Example
# ==============================================================================

if __name__ == "__main__":
    import numpy as np
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Create model
    model = create_film_lora_unet(lora_rank=8).to(device)
    
    total, trainable = model.count_parameters()
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    
    # Test forward pass WITHOUT trajectory conditioning
    print("\nTest forward pass (no FiLM):")
    x = torch.randn(2, 6, 64, 64, device=device)
    t = torch.randint(0, 1000, (2,), device=device)
    out = model(x, t)
    print(f"  Input: {x.shape}, Output: {out.shape}")
    
    # Test forward pass WITH trajectory conditioning
    print("\nTest forward pass (with FiLM):")
    orig_traj = torch.rand(2, 50, 2, device=device)
    user_traj = torch.rand(2, 50, 2, device=device)
    out = model(x, t, original_traj=orig_traj, user_traj=user_traj)
    print(f"  Input: {x.shape}, Output: {out.shape}")
    
    # Test freezing
    print("\nTest freeze_base_and_lora:")
    model.freeze_base_and_lora()
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable after freeze: {trainable_after:,}")
    print(f"  (Only trajectory encoder + FiLM generators)"))
