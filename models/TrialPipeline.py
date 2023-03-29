import numpy as np
import torch
from torch.functional import F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from models.Resnetwork import ResNetNetwork

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TrialPipeline():

    def __init__(self, batch_size: int, num_workers: int, epochs: int, lr: float, device,
                 scaled=True) -> None:
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = MNIST(root="data", download=True, transform=ToTensor(), train=True)
        self.test_dataset = MNIST(root="data", download=True, transform=ToTensor(), train=False)
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.model = ResNetNetwork(scaled=scaled)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                           num_workers=self.num_workers, shuffle=True)
        self.val_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                         num_workers=self.num_workers, shuffle=False)

    #TODO: maybe kfold and reset weights

    def train_pipeline(self):
        # train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        # val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

        net = self.model
        net = net.to(self.device)

        adam = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=0.001)
        best_loss = float('inf')
        best_epoch = 0

        for epoch in range(self.epochs):
            print(f"--EPOCH {epoch + 1}--")

            train_loss, train_acc = self.train_single_epoch(model=net, optimizer=adam, dataloader=self.train_dataloader)
            print(f"Train loss {train_loss:.5f}, Train acc {train_acc:.5f}")

            val_loss, val_acc = self.validate(model=net, dataloader=self.val_dataloader)
            print(f"Val loss {val_loss:.5f}, Val acc {val_acc:.5f}")

        return net

    def train_single_epoch(self, model, optimizer, dataloader):
        model.train()

        losses = []
        total = 0.
        correct = 0
        for imgs, labels in dataloader:
            optimizer.zero_grad()
            imgs, labels = imgs.to(device), labels.to(self.device)

            out = model(imgs)
            _, predicted = torch.max(out.data, 1)

            loss = F.cross_entropy(out, labels)
            losses.append(loss.cpu().detach().numpy())
            loss.backward()

            optimizer.step()
            total += len(imgs)
            correct += (predicted == labels).sum().item()

        losses = np.array(losses)
        return losses.mean() / len(dataloader), correct / total

    def validate(self, model, dataloader):
        model.eval()

        losses = []
        total = 0.
        correct = 0

        with torch.no_grad():
            for imgs, labels in dataloader:
                imgs, labels = imgs.to(device), labels.to(self.device)

                out = model(imgs)
                _, predicted = torch.max(out.data, 1)

                loss = F.cross_entropy(out, labels)
                losses.append(loss.cpu().detach().numpy())

                total += len(imgs)
                correct += (predicted == labels).sum().item()

        losses = np.array(losses)
        return losses.mean() / len(dataloader), correct / total

