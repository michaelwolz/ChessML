import chessboard_processor as cp
from os import listdir


def generate_training_images(path_to_input_files):
    for i, filename in enumerate(listdir(path_to_input_files)):
        if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
            print("Processing " + filename + "...")
            cp.process_chessboard(path_to_input_files + filename, "data/eval/", str(i), False)


def main():
    generate_training_images("data/chessboards/")


if __name__ == "__main__":
    main()
