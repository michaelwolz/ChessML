import cv2
import os

"""
We will flip all training images vertically and rotate them 3x by 90 degrees.
Result will be 2 x 4 x #traindata 
"""


class Image:
    path = ""
    filename = ""
    extension = ""
    img = None

    def __init__(self, path, filename):
        self.path = path
        self.filename, self.extension = os.path.splitext(filename)
        self.img = cv2.imread(path + filename)

    def flip(self):
        print("Flipping Image", self.path, self.filename)
        flipped = cv2.flip(self.img, +1)
        cv2.imwrite(self.path + self.filename + "_flipped" + self.extension, flipped)

    def rotate(self):
        print("Rotating Image", self.path, self.filename)
        rot90 = cv2.rotate(self.img, cv2.ROTATE_90_CLOCKWISE)
        rot180 = cv2.rotate(self.img, cv2.ROTATE_180)
        rot270 = cv2.rotate(self.img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(self.path + self.filename + "_90" + self.extension, rot90)
        cv2.imwrite(self.path + self.filename + "_180" + self.extension, rot180)
        cv2.imwrite(self.path + self.filename + "_270" + self.extension, rot270)


def augment_images(path):
    for filename in os.listdir(path):
        if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
            img = Image(path, filename)
            img.flip()

    for filename in os.listdir(path):
        if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
            img = Image(path, filename)
            img.rotate()


if __name__ == "__main__":
    augment_images("../data/augmented/train/empty/")
    augment_images("../data/augmented/train/bb/")
    augment_images("../data/augmented/train/bk/")
    augment_images("../data/augmented/train/bn/")
    augment_images("../data/augmented/train/bp/")
    augment_images("../data/augmented/train/bq/")
    augment_images("../data/augmented/train/br/")
    augment_images("../data/augmented/train/wb/")
    augment_images("../data/augmented/train/wk/")
    augment_images("../data/augmented/train/wn/")
    augment_images("../data/augmented/train/wp/")
    augment_images("../data/augmented/train/wq/")
    augment_images("../data/augmented/train/wr/")
    
    augment_images("../data/augmented/validation/empty/")
    augment_images("../data/augmented/validation/bb/")
    augment_images("../data/augmented/validation/bk/")
    augment_images("../data/augmented/validation/bn/")
    augment_images("../data/augmented/validation/bp/")
    augment_images("../data/augmented/validation/bq/")
    augment_images("../data/augmented/validation/br/")
    augment_images("../data/augmented/validation/wb/")
    augment_images("../data/augmented/validation/wk/")
    augment_images("../data/augmented/validation/wn/")
    augment_images("../data/augmented/validation/wp/")
    augment_images("../data/augmented/validation/wq/")
    augment_images("../data/augmented/validation/wr/")
