from torch.utils.data import Dataset
from PIL import Image
import os

class CharacterDataset(Dataset):
    def __init__(self, root, transform=None, digit_maps=None):
        self.root = root
        self.transform = transform
        self.files = os.listdir(root)
        self.digit_maps = digit_maps

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        img_path = os.path.join(self.root, filename)
        img = Image.open(img_path).convert("RGB")

        code_str = filename.split(".")[0]
        label = [self.digit_maps[i][ch] for i, ch in enumerate(code_str)]

        if self.transform:
            img = self.transform(img)

        return img, label
