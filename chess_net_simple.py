#!/usr/bin/env python
# coding: utf-8

# The SimpleNet just decides whether a given chess tile is empty or full. This class is just intended for testing and
# to simplify the labeling progress while sorting out empty tiles that make up a large part of the data.

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import time
import progressbar

from torchvision import transforms

# Define a normalization function for the analyzed data
# Normalization values are from imagenet data
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Defining classes
classes = ("empty", "full")


# Defining the Neural Network
class SimpleChessNet(nn.Module):
    def __init__(self):
        super(SimpleChessNet, self).__init__()

        # Defining the convolutional layers of the net
        self.conv1 = nn.Conv2d(3, 12, kernel_size=5)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=5)

        # Defining the fully connected layers of the net
        self.fc1 = nn.Linear(600, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = x.view(-1, 600)  # Convert 2d data to 1d

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        return x


# Training
def train(model, optimizer, criterion, train_loader):
    model.train()
    running_loss = 0.0
    with progressbar.ProgressBar(max_value=len(train_loader)) as bar:
        for i, t_data in enumerate(train_loader):
            data, target = t_data
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            out = model(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            bar.update(i)
            if i % 2000 == 1999:
                print(" => Loss:", running_loss / 2000)
                running_loss = 0.0


# Validation
def validate(model, val_loader, epoch=0):
    model.eval()
    correct = 0
    total = 0
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    with torch.no_grad():
        for data, target in val_loader:
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            out = model(data)
            _, prediction = torch.max(out.data, 1)
            total += target.size(0)
            if torch.cuda.is_available():
                correct += prediction.eq(target).sum().cpu().item()
            else:
                correct += prediction.eq(target).sum().item()

            c = (prediction == target).squeeze()
            for i in range(target.size(0)):
                label = target[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print("\nValidation")
    print("###################################")
    print("Epoch", epoch)
    print("Accuracy: %.2f%%" % (100 * correct / total))
    print("###################################\n")
    for i in range(len(classes)):
        try:
            print('Accuracy of %5s : %2d%% [%2d/%2d]' %
                  (classes[i], 100 * class_correct[i] / class_total[i], class_correct[i], class_total[i]))
        except ZeroDivisionError:
            print('No Accuracy for %s' % classes[i])


# The save model function will save the state of a model after a specific epoch.

def save_model(model, epoch):
    torch.save(model.state_dict(), "model/simple-net_{}.pt".format(epoch))
    print("\n------- Checkpoint saved -------\n")


def main():
    # Reading the data
    train_set = torchvision.datasets.ImageFolder(root="./data/binary/train", transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2)

    val_set = torchvision.datasets.ImageFolder(root="./data/binary/validation", transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=4, shuffle=True, num_workers=2)

    model = SimpleChessNet()

    # Activate cuda support if available
    if torch.cuda.is_available():
        model = model.cuda()

    # Defining the loss function
    criterion = nn.CrossEntropyLoss()

    # Defining the optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Start training
    epochs = 3  # We don't need that many epochs for this simple purpose
    start = time.time()
    print("Starting training for %s epochs on %s" % (epochs, time.ctime()))
    for epoch in range(epochs):
        train(model, optimizer, criterion, train_loader)
        validate(model, val_loader, epoch)
        save_model(model, epoch)
    end = time.time()
    print("Training of the neuroal network done.")
    print("Time spent:", end - start, "s")


if __name__ == "__main__":
    main()
