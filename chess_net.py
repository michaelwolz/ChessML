#!/usr/bin/env python
# coding: utf-8

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import time
import numpy as np
import progressbar

from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

"""
Sampling data for debugging
"""
# Training
n_training_samples = 1000
train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))

"""
Loading the data
"""
# Train Data
train_set = torchvision.datasets.ImageFolder(root="./data/train", transform=transform)
train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=4,
                                           num_workers=2,
                                           shuffle=True,
                                           # sampler=train_sampler,
                                           drop_last=True
                                           )

# Validation Data
val_set = torchvision.datasets.ImageFolder(root="./data/validation", transform=transform)
val_loader = torch.utils.data.DataLoader(val_set,
                                         batch_size=4,
                                         num_workers=2,
                                         shuffle=True,
                                         drop_last=True
                                         )


'''
Defining classes

bb = Black Bishop
bk = Black King
bn = Black Knight
bp = Black Pawn
bq = Black Queen
br = Black Rook
'''

classes = ("bb", "bk", "bn", "bp", "bq", "br", "empty", "wb", "wk", "wn", "wp", "wq", "wr")


class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()

        # Defining the convolutional layers of the net
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 25, kernel_size=5)
        self.conv3 = nn.Conv2d(25, 50, kernel_size=5)

        self.dropout1 = nn.Dropout()
        self.dropout2 = nn.Dropout()

        # Defining the fully connected layers of the net
        self.fc1 = nn.Linear(4 * 4 * 50, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 13)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)

        x = x.view(-1, 4 * 4 * 50)  # Convert 2d data to 1d

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def train(model, optimizer, criterion):
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


def validate(model, epoch=0):
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
    return correct / total  # Returning accuracy


def save_model(model, epoch):
    torch.save(model.state_dict(), "model/chess-net.pt".format(epoch))
    print("\n------- Checkpoint saved -------\n")


def main():
    model = ChessNet()

    # Activate cuda support if available
    if torch.cuda.is_available():
        print("Activating cuda support!")
        model = model.cuda()

    # Defining the loss function
    criterion = nn.CrossEntropyLoss()

    # Defining the optimizer
    # optimizer = optim.Adam(model.parameters())
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.ASGD(model.parameters())

    # Start training
    epochs = 20
    best_acc = 0
    start = time.time()
    print("Starting training for %s epochs on %s" % (epochs, time.ctime()))
    for epoch in range(epochs):
        train(model, optimizer, criterion)
        acc = validate(model, epoch)
        if acc > best_acc:
            best_acc = acc
            save_model(model, epoch)
    end = time.time()
    print("Training of the neuroal network done.")
    print("Time spent:", end - start, "s")

    # Testing the NN
    # print("\nTest:")
    # for i in range(4):
    #     test(model)


if __name__ == "__main__":
    main()
