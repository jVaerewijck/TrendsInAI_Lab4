import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Laad het encoder model
encoder = models.resnet18()
encoder.fc = nn.Identity()  # Verwijder classificatielaag
encoder.load_state_dict(torch.load("simclr_encoder.pth"))
encoder = encoder.eval().cuda()

# Transform voor de dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Maak embeddings
all_embeddings = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.cuda()
        embeddings = encoder(images)  # Haal embeddings uit de encoder
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

# Zet lists om naar numpy arrays
all_embeddings = np.concatenate(all_embeddings, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(all_embeddings)

# Plot maken
plt.figure(figsize=(10, 10))
num_classes = 10 #Cifar-10
colors = plt.cm.get_cmap("tab10", num_classes)

for i in range(num_classes):
    indices = all_labels == i
    plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=test_dataset.classes[i], alpha=0.5, color=colors(i))

plt.legend()
plt.title("t-SNE visualization of SimCLR embeddings")
plt.show()
