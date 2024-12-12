from torch.utils.data import Dataset
import torch
import torchvision

class FumoDataset(Dataset):
    def __init__(self, dataset, transform: torchvision.transforms.Compose = None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, boxes = self.dataset[idx]
        if self.transform:
            if isinstance(image, torch.Tensor):
                image = image.permute(1, 2, 0).numpy()
            h, w, c = image.shape
            image = self.transform(image)
            n_c, n_h, n_w  = image.shape
            # transform empty boxes to tensor (0, 4)
            if len(boxes) == 0:
                boxes = torch.zeros((0, 4))
            else:
                if not isinstance(boxes, torch.Tensor):
                    boxes = torch.tensor(boxes, dtype=torch.float32)
            boxes = boxes * torch.tensor([n_w / w, n_h / h, n_w / w, n_h / h])
        return image, boxes