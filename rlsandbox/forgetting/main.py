import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from rlsandbox.base.torch import parallel_eval_on_input


def main() -> None:
    Main().run()


class Main:
    def __init__(
            self,
            device: torch.device = 'cuda',
            epochs: int = 10,
            batch_size: int = 1024,
            grad_min: float = None,
    ) -> None:
        self.device = device
        self.epochs = epochs
        self.grad_min = grad_min

        # Define transformations for the dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Load the MNIST training and test datasets
        self.train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        self.test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        self.train_dataset_0_4 = [(image, label) for image, label in self.train_dataset if label <= 4]
        self.train_dataset_5_9 = [(image, label) for image, label in self.train_dataset if label >= 5]

        print(f'len(train_dataset) = {len(self.train_dataset)}')
        print(f'len(test_dataset) = {len(self.test_dataset)}')

        # Create DataLoader for batching and shuffling
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=batch_size, shuffle=False)

        self.train_loader_0_4 = DataLoader(dataset=self.train_dataset_0_4, batch_size=batch_size, shuffle=True)
        self.train_loader_5_9 = DataLoader(dataset=self.train_dataset_5_9, batch_size=batch_size, shuffle=True)

        # self.model = BasicCNN().to(device)
        self.model = Elephant().to(device)
        print(f'Trainable parameters: {count_trainable_parameters(self.model)}')

        self.loss_fn = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-2)

    def run(self):
        self.normal_train_test()
        # self.forgetting_train_test()

    def normal_train_test(self):
        for epoch in range(self.epochs):
            self.epoch = epoch

            self.train(self.train_loader)
            self.eval(self.test_loader)

            print()

    def forgetting_train_test(self):
        print('Train on 0..4')
        for epoch in range(5):
            self.epoch = epoch

            self.train(self.train_loader_0_4)
        print()

        print('Train on 5..9')
        for epoch in range(1):
            self.epoch = epoch

            self.train(self.train_loader_5_9)
        print()

        print('Eval on 0..4')
        self.eval(self.train_loader_0_4)

    def train(self, data_loader) -> None:
        self.model.train()

        all_losses = []
        for images, labels in data_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            y_pred = self.model(images)

            self.optimizer.zero_grad()
            loss = self.loss_fn(y_pred, labels)
            loss.backward()

            if self.grad_min is not None:
                clip_gradients_below_threshold(self.model, self.grad_min)

            self.optimizer.step()

            all_losses.append(loss.item())

        self.log('Train loss:', sum(all_losses) / len(all_losses))

    def eval(self, data_loader) -> None:
        self.model.eval()

        all_losses = []
        for images, labels in data_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            y_pred = self.model(images)

            loss = self.loss_fn(y_pred, labels)
            all_losses.append(loss.item())

        self.log('Test loss:', sum(all_losses) / len(all_losses))

    def log(self, *args):
        print(f'[Epoch {self.epoch}]', *args)


class BasicCNN(torch.nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.num_classes = num_classes

        activation = nn.ReLU()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1),
            activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.out_layers = nn.Sequential(
            nn.Linear(1600, 128),
            activation,
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_layers(x)
        x = x.flatten(start_dim=1)
        x = self.out_layers(x)
        return x


class Elephant(torch.nn.Module):
    def __init__(self, num_classes: int = 10, num_cnns: int = 50) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.num_cnns = num_cnns

        self.l1_cnns = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1),
                nn.ReLU(),
            )
            for _ in range(num_cnns)
        ])

        self.l1_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.l2_cnns = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=1),
                nn.ReLU(),
            )
            for _ in range(num_cnns)
        ])

        self.l2_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.out_layers = nn.Sequential(
            nn.Linear(1600, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor, temp: float = 1.) -> torch.Tensor:
        dev = x.device
        batch_size = x.size(0)
        num_cnns = self.num_cnns

        t0 = time.monotonic()
        x = parallel_eval_on_input(self.l1_cnns, x)
        t1 = time.monotonic()
        dt = t1 - t0

        x = torch.stack(x)
        activity = x.reshape(num_cnns, batch_size, -1).mean(dim=2)

        choice_probs = torch.softmax(activity / temp, dim=0)
        x *= choice_probs.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = x.sum(dim=0)

        # from collections import Counter
        # choices = torch.multinomial(choice_probs.T, num_samples=1).flatten()
        # c = Counter(choices.tolist())
        # c = dict(sorted(c.items()))
        # fractions = {k: v / batch_size for k, v in c.items()}

        # x = x[choices, range(batch_size)]
        x = self.l1_pool(x)

        x = parallel_eval_on_input(self.l2_cnns, x)

        x = torch.stack(x)
        activity = x.reshape(num_cnns, batch_size, -1).mean(dim=2)

        choice_probs = torch.softmax(activity / temp, dim=0)
        x *= choice_probs.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = x.sum(dim=0)

        # choice_probs = torch.softmax(activity / temp, dim=0)
        # choices = torch.multinomial(choice_probs.T, num_samples=1).flatten()
        # x = x[choices, range(batch_size)]

        x = self.l2_pool(x)

        x = x.flatten(start_dim=1)
        x = self.out_layers(x)
        return x


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def clip_gradients_below_threshold(model: nn.Module, threshold: float) -> None:
    for param in model.parameters():
        if param.grad is not None:
            grad = param.grad.data
            grad[grad.abs() < threshold] = 0
