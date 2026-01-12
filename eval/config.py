


import torch
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class EvalConfig:
    checkpoint: str
    timesteps: int = 1000
    img_size: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    n_samples: int = 3  

