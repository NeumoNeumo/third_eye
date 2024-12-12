import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from modules import YuNet
from loss import DetectionLoss
from fumo_dataset import FumoDataset
from dataset.fumoDataset import getFumoDataset

def collate_fn_images_bboxes(batch):
    images, boxes = zip(*batch)
    images = torch.stack(images)
    return images, boxes

class FumoYunet(pl.LightningModule):
    def __init__(self, model, loss_function, learning_rate=0.001, max_lr=0.01, warmup_iters=1500, step_milestones=[400, 544]):
        super().__init__()
        self.model = model
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.max_lr = max_lr
        self.warmup_iters = warmup_iters
        self.step_milestones = step_milestones
        self.current_iter = 0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.loss_function(outputs, targets)
        self.log('train_loss', loss)
        self.current_iter += 1
        return loss

    def configure_optimizers(self):
        # from YuNet paper
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=0.0005)
        def linear_warmup_lr(epoch):
            if epoch < self.warmup_iters:
                return (self.max_lr - self.learning_rate) / self.warmup_iters * epoch + self.learning_rate
            return self.max_lr
        
        def step_lr(epoch):
            current_lr = self.max_lr
            for milestone in self.step_milestones:
                if epoch >= milestone:
                    current_lr *= 0.1
            return current_lr
        
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_warmup_lr)
        
        step_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=step_lr)
        
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, step_scheduler],
            milestones=[self.warmup_iters]
        )
        
        return [optimizer], [scheduler]

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        
        transforms.Resize((640, 640))

    ])

    SEED = 42
    generator = torch.Generator().manual_seed(SEED)
    dataset = FumoDataset(getFumoDataset('./dataset'), transform=transform)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2], generator)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, generator=generator, collate_fn=collate_fn_images_bboxes)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, generator=generator, collate_fn=collate_fn_images_bboxes)

    model = YuNet()
    loss_function = DetectionLoss()

    detection_model = FumoYunet(model, loss_function)

    trainer = pl.Trainer(
        max_epochs=1500 + 544,
        accelerator='auto',
        devices='auto',
        log_every_n_steps=10,
        enable_checkpointing=True,
    )

    trainer.fit(detection_model, train_loader, val_loader)

    torch.save(model.state_dict(), 'fumo_yunet.pth')

if __name__ == '__main__':
    main()