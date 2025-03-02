'''
# resnet50_lightning.py
## Author: @hamzamooraj99 (Hamza Hassan Mooraj)
Description: This file contains the Lightning Module for a ResNet50 pre-trained CNN model as well as transfer learning steps on the AgriPath-LF16-30k Dataset
'''

import itertools
from datasets import load_dataset, Dataset
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from torchvision.models.resnet import ResNet50_Weights
import pytorch_lightning as pl

# A class to load AgriPath variant, transform according to model specs and create loaders
class AgriPathDataModule(pl.LightningDataModule):
    def __init__(self, hf_repo, batch_size):
        super().__init__()
        self.hf_repo = hf_repo
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        self.dataset = load_dataset(self.hf_repo)
        self.train_set = self.dataset['train']
        self.val_set = self.dataset['validation']
        self.test_set = self.dataset['test']
    
    def collate_fn(self, batch):
        images = [self.transform(sample['image'].convert('RGB')) for sample in batch]
        labels = [sample['numeric_label'] for sample in batch]

        # print(f"Image batch shape: {[image.shape for image in images]}")
        # print(f"Label batch shape: {labels}")

        return torch.stack(images), torch.tensor(labels)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn, num_workers=29, persistent_workers=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn, num_workers=29, persistent_workers=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn, num_workers=29, persistent_workers=True)

# A class that defines and modifies the ResNet50 model so that it can be used with the DataModule defined above
class ResNet50TLModel(pl.LightningModule):
    def __init__(self, num_classes, input_shape=2048, learning_rate=2e-4):
        super().__init__()
        
        # Log HyperParams
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.dim = input_shape
        self.num_classes = num_classes

        # Load pre-trained ResNet50 model and freeze early layers for feature extraction
        self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Unfreeze last residual block for fine-tuning
        for param in list(self.backbone.layer4.parameters()):
            param.requires_grad = True

        # Remove original Fully Connected Layer (optimised for ImageNet)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Create custom classification head
        self.classifier = nn.Linear(in_features, num_classes)

        # Loss function and metrics
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = MulticlassAccuracy(num_classes=num_classes, average='macro')
        # self.per_class_accuracy = MulticlassAccuracy(num_classes=num_classes, average=None)
    
    def forward(self, x):
        features = self.backbone(x) #Extract features
        out = self.classifier(features) #Final pass through custom classifier
        return out
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        out = self.forward(images)
        loss = self.criterion(out, labels)
        acc = self.accuracy(out, labels)

        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_epoch=True, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        out = self.forward(images)
        loss = self.criterion(out, labels)
        acc = self.accuracy(out, labels)
        # pc_acc = self.per_class_accuracy(out, labels)

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_epoch=True, prog_bar=True)
        # for i, class_acc in enumerate(pc_acc):
        #     self.log(f"val/acc_class{i}", class_acc, prog_bar=False)

        return loss
    
    def test_step(self, batch, batch_idx):
        images, labels = batch
        out = self.forward(images)
        loss = self.criterion(out, labels)
        acc = self.accuracy(out, labels)
        # pc_acc = self.per_class_accuracy(out, labels)
        
        self.log("test/loss", loss)
        self.log("test/acc", acc)
        # for i, class_acc in enumerate(pc_acc):
        #     self.log(f"test/acc_class{i}", class_acc, prog_bar=False)

        return {'loss': loss, 'outputs': out, 'labels': labels}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)

def check_loader(hf_repo: str):
    train_loader = AgriPathDataModule(hf_repo, batch_size=4)
    train_loader.setup()
    train_loader = train_loader.train_dataloader()

    images, labels = next(iter(train_loader))

    print(type(images), images.shape)
    print(type(labels), labels.shape)

if __name__ == '__main__':
    hf_repo = "hamzamooraj99/AgriPath-CNN"
    batch_sizes = [16, 32, 64]
    learning_rates = [1e-4, 5e-4, 2e-4]
    num_classes = 65
    max_epochs = 10
    experiment_id = 0

    for batch_size, lr in itertools.product(batch_sizes, learning_rates):
        print(f"\n==== Running Experiment {experiment_id}: Batch Size = {batch_size}, LR = {lr} ====\n")

        # Load dataset with new batch_size
        print(f"Loading dataset with batch size {batch_size}")
        datamodule = AgriPathDataModule(hf_repo=hf_repo, batch_size=batch_size)
        datamodule.setup()

        # Define model with new learning rate
        print(f"\nDefine model with learning rate {lr}")
        model = ResNet50TLModel(num_classes=num_classes, learning_rate=lr)

        # Trainer setup
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator='gpu',
            devices=1,
            log_every_n_steps=10
        )

        # Train model
        print(f"\nTraining model")
        trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader())

        # Test model
        print(f"\nTesting model")
        trainer.test(model, datamodule.test_dataloader())

        # Save model
        print(f"\nSaving model")
        torch.save(model.state_dict(), f"resnet50_agripath_exp_{experiment_id}.pth")

        experiment_id += 1