# src/dataset.py
import pandas as pd
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ChestXrayDataset(Dataset):
    def __init__(self, metadata, all_labels, transform=None):
        self.metadata = metadata
        self.all_labels = all_labels
        self.transform = transform
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = row['Image Path']
        image = Image.open(img_path).convert("L")  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
        
        # Extract labels as a tensor: Ensure the labels are numeric (0 or 1)
        labels = row[self.all_labels].values.astype(np.float32)  # Convert to float32
        labels_tensor = torch.tensor(labels, dtype=torch.float32)  # Create tensor
        
        return image, labels_tensor
