import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def main() -> None:
    Main().run()


class Main:
    def __init__(
            self,
            device: torch.device = 'cuda',
            epochs: int = 10,
            batch_size: int = 1024,
            train_on_labels: bool = False,
            show_images: bool = True,
    ) -> None:
        self.device = device
        self.epochs = epochs
        self.train_on_labels = train_on_labels
        self.show_images = show_images

        # Define transformations for the dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Load the MNIST training and test datasets
        self.train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        self.test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        print(f'len(train_dataset) = {len(self.train_dataset)}')
        print(f'len(test_dataset) = {len(self.test_dataset)}')

        # Create DataLoader for batching and shuffling
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=batch_size, shuffle=False)

        self.model = Autoencoder().to(device)
        print(f'Trainable parameters: {count_trainable_parameters(self.model)}')

        self.image_loss_fn = nn.MSELoss()
        self.label_loss_fn = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-2)

    def run(self):
        for epoch in range(self.epochs):
            self.epoch = epoch

            self.train(self.train_loader)
            self.eval(self.test_loader)

            print()

    def train(self, data_loader) -> None:
        self.model.train()

        all_losses = []
        for images, labels in data_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            pred_images, pred_labels = self.model(images)

            self.optimizer.zero_grad()
            loss = self.image_loss_fn(pred_images, images)

            if self.train_on_labels:
                loss += self.image_loss_fn(pred_labels, labels.unsqueeze(1))

            loss.backward()

            self.optimizer.step()

            all_losses.append(loss.item())

        self.log('Train loss:', sum(all_losses) / len(all_losses))

    def eval(self, data_loader) -> None:
        self.model.eval()

        all_losses = []
        show_images = self.show_images
        for images, labels in data_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            pred_images, pred_labels = self.model(images)

            if show_images:
                show_images = False

                i = 9

                fig, axes = plt.subplots(1, 2)
                axes[0].imshow(images[i].detach().cpu().numpy().squeeze(), cmap='gray')
                axes[0].set_title('Original Image')

                # Plot the reconstructed images
                axes[1].imshow(pred_images[i].detach().cpu().squeeze(), cmap='gray')
                axes[1].set_title('Reconstructed Image')

                plt.show()

            loss = self.image_loss_fn(pred_images, images)
            if 0 and self.train_on_labels:
                loss += self.image_loss_fn(pred_labels, labels)

            all_losses.append(loss.item())

        self.log('Test loss:', sum(all_losses) / len(all_losses))

    def log(self, *args):
        print(f'[Epoch {self.epoch}]', *args)


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        emb_dim = 24

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 16 x 14 x 14
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 32 x 7 x 7
            nn.ReLU(),
            nn.Conv2d(32, emb_dim, kernel_size=7),  # 64 x 1 x 1
            nn.LayerNorm([emb_dim, 1, 1]),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(emb_dim, 32, kernel_size=7),  # 32 x 7 x 7
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16 x 14 x 14
            nn.GELU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 1 x 28 x 28
            nn.Sigmoid(),  # We use Sigmoid to normalize output to [0, 1]
        )

        self.label_predictor = nn.Sequential(
            nn.Linear(emb_dim, 32),
            nn.GELU(),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        pred_label_logits = self.label_predictor(encoded.squeeze())

        return decoded, pred_label_logits


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    main()
