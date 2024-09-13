from multiprocessing import Pool, Process
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.io import read_image
import torch.multiprocessing as mp
import random
import time
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
import warnings
import json
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes, self.class_to_idx = self._find_classes(self.root_dir)
        self.imgs = self.make_dataset()

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def make_dataset(self):
        images = []
        for target_class in sorted(self.class_to_idx.keys()):
            class_index = self.class_to_idx[target_class]
            target_dir = os.path.join(self.root_dir, target_class)
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    if not fname.startswith("."):  # Skip hidden files and folders
                        path = os.path.join(root, fname)
                        item = (path, class_index)
                        images.append(item)
        return images

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        path, class_index = self.imgs[idx]
        image = read_image(path)

        # Convert grayscale to RGB if necessary
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        image = image.float() / 255.0

        if self.transform:
            image = self.transform(image)

        image = T.Resize((299, 299), antialias=True)(image)
        return image, class_index, path

def prepare_dataloader(dataset, batch_size, num_workers=4):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=num_workers,
    )
    return loader

def initialize_resnet_model(num_classes=4):
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Dropout(0.1),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.1),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.1),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.1),
        nn.Linear(128, num_classes),
    )

    return model

def initialize_optimizer_and_criterion(model):
    optimizer = optim.RMSprop(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    return optimizer, criterion

def train_resnet_model(
    model_state_dict, criterion, optimizer, data_loaders, num_epochs=3
):
    model = initialize_resnet_model()
    model.load_state_dict(model_state_dict)

    since = time.time()
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    metrics = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "time": [],
    }

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            data_iter = data_loaders[phase]
            if phase == "train":
                data_iter = tqdm(data_iter, desc=f"{phase.capitalize()} Epoch {epoch + 1}/{num_epochs}")

            for inputs, labels, _ in data_iter:
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                total_samples += inputs.size(0)

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects / total_samples

            metrics[f"{phase}_loss"].append(epoch_loss)
            metrics[f"{phase}_acc"].append(epoch_acc)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    metrics["time"].append(time_elapsed)

    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {:4f}".format(best_acc))
    
    return best_model_wts, metrics

def main(rank, num_epochs, batch_size, train_dir, val_dir):
    try:
        train_dataset = CustomDataset(root_dir=train_dir)
        val_dataset = CustomDataset(root_dir=val_dir)
        train_loader = prepare_dataloader(train_dataset, batch_size)
        val_loader = prepare_dataloader(val_dataset, batch_size)

        model = initialize_resnet_model()
        model_state_dict = torch.load("resnet_model.pth")

        optimizer, criterion = initialize_optimizer_and_criterion(model)

        data_loaders = {"train": train_loader, "val": val_loader}

        best_model_wts, metrics = train_resnet_model(model_state_dict, criterion, optimizer, data_loaders, num_epochs)

        torch.save(best_model_wts, "resnet_model.pth")

        print("Training metrics:")
        for key, value in metrics.items():
            if isinstance(value, list):
                print(f"{key}: {value}")

    except Exception as e:
        print(f"Error in process {rank}: {e}")


if __name__ == "__main__":
    cpu_counts = [1, 2, 4]
    num_epochs = 5
    batch_size = 256
    train_dir = "RevisedData/train"
    val_dir = "RevisedData/validation"

    processes = []
    for num_cpus in cpu_counts:
        print(f"Training with {num_cpus} CPU core(s)")
        for i in range(num_cpus):
            p = Process(target=main, args=(i, num_epochs, batch_size, train_dir, val_dir))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        print(f"Completed training with {num_cpus} CPU core(s)\n")
