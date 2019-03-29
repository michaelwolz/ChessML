import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import time
import progressbar
import os

from torchvision import transforms, models

# Implementation based on resnet18
# Accuracy of 99% after 12 Epochs of training with 31.200 training images and 7.800 validation images

# Where to store the model
MODELPATH = "/content/drive/My Drive/ChessNetData/model/chess-net-v2-sgd.tar"

# Defining basic transform operations. Image size of 224x224 is required by underlying resnet
# The normalization function based on the ImageNet data which was used to train the resnet model
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# Loading the training and validation data

# Train Data
train_set = torchvision.datasets.ImageFolder(root="/content/data/augmented/train", transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=25, num_workers=2, shuffle=True, drop_last=True)

# Validation Data
val_set = torchvision.datasets.ImageFolder(root="/content/data/augmented/validation", transform=transform)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=25, num_workers=2, shuffle=True, drop_last=True)

# Defining classes:
# bb = Black Bishop
# bk = Black King
# bn = Black Knight
# bp = Black Pawn
# bq = Black Queen
# br = Black Rook

classes = ("bb", "bk", "bn", "bp", "bq", "br", "empty", "wb", "wk", "wn", "wp", "wq", "wr")


def train(model, optimizer, criterion):
    model.train()
    running_loss = 0.0
    with progressbar.ProgressBar(max_value=len(train_loader)) as bar:
        for i, t_data in enumerate(train_loader):
            data, target = t_data

            # put data on the gpu if available
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
            if i % 200 == 199:
                print(" => Loss:", running_loss / 200)
                running_loss = 0.0


def validate(model, epoch=0):
    model.eval()
    correct = 0
    total = 0
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    with torch.no_grad():
        for data, target in val_loader:
            # put data on the gpu if available
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


def save_model(model, optimizer, epoch, best_acc):
    # Saving a checkpoint of the training. This is essential for using the trained network and also to resume training
    # if it stopped for some reason (e.g. limitations of Google Colab)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'bestacc': best_acc,
    }, MODELPATH)
    print("\n------- Checkpoint saved -------\n")


def main():
    resume_training = True  # resuming training or starting a new one

    model = models.resnet18(pretrained=True)  # use pretrained version of resnet18

    for param in model.parameters():
        param.require_grad = False  # freeze model to modify just the last layer of the nn

    n_features = model.fc.in_features  # get the number of features for the new last layer

    fc = nn.Sequential(
        nn.Linear(n_features, 320),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(460, 13)  # one output for every class
    )

    model.classifier = fc

    # Activate cuda support if available
    if torch.cuda.is_available():
        print("### Activating cuda support! ###\n")
        model = model.cuda()

    # Defining the loss function
    criterion = nn.CrossEntropyLoss()

    # Defining the optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Loading model for resuming training
    starting_epoch = 0
    best_acc = 0
    best_epoch = 0

    if resume_training:
        if os.path.exists(MODELPATH):
            state = torch.load(MODELPATH)
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            starting_epoch = state["epoch"]
            best_acc = state["bestacc"]
            best_epoch = state["epoch"]
            print("=> Resuming training at epoch %d with best-accuracy of: %.2f%%" % (starting_epoch, 100 * best_acc))
    else:
        if os.path.exists(MODELPATH):
            answer = input("This will overwrite your existing model! Do you want to continue? [y, n]")
            if answer != 'y':
                exit(0)
        print("=> Starting first training of model")

    # Start training
    epochs = 20  # amount of epochs for training
    start = time.time()
    print("Start training for %s epochs on %s" % (epochs - starting_epoch, time.ctime()))
    for epoch in range(starting_epoch, epochs):
        train(model, optimizer, criterion)
        acc = validate(model, epoch)
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            save_model(model, optimizer, epoch, acc)
    end = time.time()

    print("Training of the model done.")
    print("Time spent:", end - start, "s")
    print("Best-Accuracy: %.2f%% after epoch %d" % (100 * best_acc, best_epoch))


if __name__ == "__main__":
    main()
