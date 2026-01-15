

from torch.utils.data import DataLoader
import torch


def create_dataloaders(dataset, obstacle_classes, batch_size = 32, val_split = 0.1, num_workers =4, seed=42):
    dataset.obstacle_classes = obstacle_classes

    total = len(dataset)
    val_size = int(total * val_split)
    train_size = total - val_size

    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size],
            generator=torch.Generator.manual_seed(seed)
            )


    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader
