import torch
import time
from torchvision import datasets, transforms
from vision.F_MNIST.classifier import Classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

model = Classifier()
model.to(device)
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.003)

epochs = 30
steps = 0

train_losses, test_losses = [], []
for e in range(epochs):

    running_loss = 0
    start = time.time()
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        test_loss = 0
        accuracy = 0

        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                log_ps = model(images)
                test_loss += criterion(log_ps, labels)

                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(top_class.shape)

                accuracy += torch.mean(equals.type(torch.FloatTensor))

    train_losses.append(running_loss)
    test_losses.append(test_loss)

    print("==============================================")
    print(f"Device = {device}; Elapsed time {(time.time() - start):.3f} seconds")
    print(f'Training loss: {running_loss / len(trainloader)}')
    print(f'Test loss: {test_loss / len(testloader)}')
    print(f'Accuracy: {accuracy / len(testloader)}')
