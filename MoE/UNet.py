import torch
import torch.nn as nn
import torch.nn.functional as F
from .time_emb import SinusoidalEmbeddings
from .attention import AttentionBlock

class TimeAwareBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.act1 = nn.SiLU()

        self.time_proj = nn.Linear(time_dim, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.act2 = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)

        t_proj = self.time_proj(t_emb)[:, :, None, None]
        h = h + t_proj

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)
        return h

class LightweightUNet(nn.Module):
    def __init__(self, in_channels, context_channels, out_channels=1, base_channels=32, time_dim=128):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalEmbeddings(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        self.enc1 = self.conv_block(in_channels, base_channels, time_dim)
        self.enc2 = self.conv_block(base_channels, base_channels*2, time_dim)
        self.center = self.conv_block(base_channels*2, base_channels*4, time_dim)
       
        self.context = nn.Sequential(
                nn.Conv2d(context_channels, base_channels*2, 3, padding=1),
                nn.SiLU(),
                nn.AvgPool2d(4),
                nn.Conv2d(base_channels*2, base_channels * 4, 1)
                ) #This and the attention block is what the user trains

        self.mid_attn = AttentionBlock(base_channels*4, num_heads=4)

        self.dec2 = self.conv_block(base_channels*6, base_channels*2, time_dim)
        
        self.dec1 = self.conv_block(base_channels*3, base_channels, time_dim)

        self.final = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def conv_block(self, in_c, out_c, time_dim):
        return TimeAwareBlock(in_c, out_c, time_dim)

    def set_finetune(self, active=True):
        for param in self.parameters():
            param.requires_grad = not active

        if active:
            for param in self.context.parameters():
                param.requires_grad = True

            for param in self.mid_attn.parameters():
                param.requires_grad = True

    def forward(self, x, t, context_stack = None):
        t_emb = self.time_mlp(t)

        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(F.max_pool2d(e1, 2), t_emb)

        c = self.center(F.max_pool2d(e2, 2), t_emb)
       
        if context_stack is not None:
            c_context = self.context(context_stack)
            c = c + c_context
        
        c = self.mid_attn(c)
        c_up = F.interpolate(c, scale_factor=2, mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([c_up, e2], dim=1), t_emb)

        d2_up = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d2_up, e1], dim=1), t_emb)

        return self.final(d1)

    def set_active_peers(self, active_peer_indices):
        """
        NEW: Surgical masking.
        Registers a hook to zero out gradients for context channels 
        that belong to peers NOT in the active set.
        """
        # We assume input channels are ordered: [Peer0_EDF, Peer0_Sin, Peer0_Cos, Peer1_EDF...]
        CHANNELS_PER_PEER = 3 
        
        def filter_grads_hook(grad):
            # grad shape: [Out, In_Channels, K, K] for the first Conv layer
            mask = torch.zeros_like(grad)
            
            for peer_idx in active_peer_indices:
                start_ch = peer_idx * CHANNELS_PER_PEER
                end_ch = start_ch + CHANNELS_PER_PEER
                
                # Boundary check to prevent crashing if indices are off
                if start_ch < grad.shape[1]:
                    mask[:, start_ch:end_ch, :, :] = 1.0
            
            return grad * mask

        # Attach to the first layer of the context sidecar
        # self.context is a Sequential, so index [0] is the first Conv2d
        if hasattr(self.context[0], 'weight'):
            self.context[0].weight.register_hook(filter_grads_hook)
