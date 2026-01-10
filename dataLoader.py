

import argparse
import os
import numpy as np
import torch
import plotly.graph_objects as go
from skimage.graph import route_through_array



class MetadataDataLoader:
    def __init__(self, dataset, n_samples):
        self.dataset = dataset
        self.n_samples = n_samples
    def __iter__(self):
        for i in range(self.n_samples):
            yield self.dataset.get_sample_with_metadata(i)
    def __len__(self):
        return self.n_samples
