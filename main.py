import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torchvision

print(torch.cuda.is_available())  # Should return True if CUDA is enabled
print(torch.cuda.current_device())  # Shows the current GPU device if available

# Define a simple augmentation pipeline
class SimCLRAugmentation:
    def __init__(self, size=224):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.ToTensor()
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)  # Two different augmentations

# SimCLR projection head
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# Define the SimCLR model
class SimCLR(nn.Module):
    def __init__(self, encoder, projection_dim=128):
        super(SimCLR, self).__init__()
        self.encoder = encoder
        self.encoder.fc = nn.Identity()  # Remove the original classification layer
        self.projection_head = ProjectionHead(input_dim=512, output_dim=projection_dim)

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return z

# NT-Xent Loss for SimCLR
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)
        sim = torch.mm(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        positive_pairs = torch.cat([sim_i_j, sim_j_i], dim=0)

        # Mask to ignore the similarity of a sample with itself
        mask = (~torch.eye(2 * batch_size, dtype=bool)).float().to(z.device)
        sim = sim * mask
        negative_pairs = sim.exp().sum(dim=1)

        loss = -torch.log(positive_pairs.exp() / negative_pairs)
        return loss.mean()

# Training function
def train(model, data_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        total_batches = len(data_loader)
        batch_idx = 0
        for (x1, x2), _ in data_loader:
            x1, x2 = x1.cuda(), x2.cuda()
            z1, z2 = model(x1), model(x2)
            loss = criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress = (batch_idx + 1) / total_batches * 100

            if batch_idx % (total_batches // 10) == 0:
                print(f"Epoch {epoch+1}/{epochs} - Progress: {progress:.2f}%")
            batch_idx += 1
            
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {epoch_loss / len(data_loader)}")

# Sample Dataset (CIFAR-10 or any image dataset with two augmentations per image)
class CustomDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        img, _ = self.dataset[index]
        x1, x2 = self.transform(img)
        return (x1, x2), _

    def __len__(self):
        return len(self.dataset)

# Main
if __name__ == "__main__":
    # Load a pretrained ResNet model as the encoder
    encoder = models.resnet18(pretrained=True)
    model = SimCLR(encoder).cuda()
    
    # CIFAR-10 dataset with SimCLR-style augmentations
    transform = SimCLRAugmentation()
    train_dataset = CustomDataset(
        dataset=torchvision.datasets.CIFAR10(root='./data', train=True, download=True),
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Define the loss and optimizer
    criterion = NTXentLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train(model, train_loader, criterion, optimizer)

    torch.save(model.encoder.state_dict(), "simclr_encoder.pth")