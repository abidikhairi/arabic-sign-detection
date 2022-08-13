import os
import pandas as pd
import torch as th
from PIL import Image
from torch.utils.data import Dataset


class SignDataset(Dataset):
    def __init__(self, image_dir, metadata, label_col, transform=None) -> None:
        super().__init__()

        self.image_dir = image_dir
        self.transform = transform

        self.frame = pd.read_csv(metadata, header=0) 
        self.frame['label'] = pd.Categorical(self.frame[label_col]).codes

        
    def __len__(self):
        return len(self.frame)

    
    def __getitem__(self, index):
        if th.is_tensor(index):
            index = index.tolist()

        image_file = self.frame.iloc[index, 'File_Name']
        image_dir = self.frame.iloc[index, 'Class']

        label = self.frame.iloc[index, 'label']
        image = Image.open(os.path.join(self.image_dir, image_dir, image_file))

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label
