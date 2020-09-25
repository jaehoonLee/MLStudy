import torch
import numpy as np
import matplotlib.pyplot as plt
from vision.MNIST.classifier import Net

from torchvision import datasets
import torchvision.transforms as transforms

num_workers = 0
batch_size = 20
valid_size = 0.2

# Loading Data
transform = transforms.ToTensor()

train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

# visualize a batch of training data
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()

fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    ax.set_title(str(labels[idx].item()))

plt.show()

# Loss function & Optimizer
model = Net()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# number of epochs to train the model
n_epochs = 30

model.train()

for epoch in range(n_epochs):
    train_loss = 0.0

    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)# Since loss.item() have mean of batch, multiple batch size

    train_loss = train_loss/len(train_loader.dataset)# Average training loss
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))