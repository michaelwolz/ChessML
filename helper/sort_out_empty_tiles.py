import torch
import sys
import os

from PIL import Image
from chess_net_simple import SimpleChessNet
from chess_net_simple import transform


# Removes empty tiles from a given dataset to simplify labeling process
def sort_out_empty_tiles(model, file_path):
    files = os.listdir(file_path)

    for file in files:
        if file.lower().endswith(".jpg") or file.lower().endswith(".jpeg"):
            img = Image.open(os.path.join(file_path, file))
            img = transform(img)
            img = torch.unsqueeze(img, 0)

            out = model(img)
            _, prediction = torch.max(out.data, 1)

            if prediction == 1:
                print("full")
            else:
                print("empty")

            # if prediction == 0:
            #     os.remove(os.path.join(file_path, file))


def main():
    model_path = "../model/simple-net.pt"
    file_path = "../data/test/empty"

    if not os.path.exists(model_path):
        print("Model weights were not found")
        sys.exit(1)

    model = SimpleChessNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    sort_out_empty_tiles(model, file_path)


if __name__ == "__main__":
    main()
