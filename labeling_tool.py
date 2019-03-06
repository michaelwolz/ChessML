import os
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import shutil

"""
Simple Labeling Tool for the generated tiles
"""


class Window:
    src_path = ""
    save_path = ""
    images = []
    current_index = 0
    panel = None

    def __init__(self, master):
        self.master = master
        self.master.title("JAFLT")

        self.master.bind("<space>", lambda empty: self.label_img("empty"))

        self.open_button = Button(self.master, text="Open Dir", command=self.open_src).grid(row=1, column=1)
        self.save_button = Button(self.master, text="Save Dir", command=self.set_save_path).grid(row=1, column=2)
        self.prev_button = Button(self.master, text="<", command=self.prev_image).grid(row=1, column=5, sticky=E)
        self.next_button = Button(self.master, text=">", command=self.next_image).grid(row=1, column=6, sticky=E)

        self.wk_button = Button(self.master, text="W. King", command=lambda: self.label_img("wk")).grid(row=3, column=1)
        self.wq_button = Button(self.master, text="W. Queen", command=lambda: self.label_img("wq")).grid(row=3,
                                                                                                         column=2)
        self.wb_button = Button(self.master, text="W. Bishop", command=lambda: self.label_img("wb")).grid(row=3,
                                                                                                          column=3)
        self.wr_button = Button(self.master, text="W. Rook", command=lambda: self.label_img("wr")).grid(row=3, column=4)
        self.wn_button = Button(self.master, text="W. Knight", command=lambda: self.label_img("wn")).grid(row=3,
                                                                                                          column=5)
        self.wp_button = Button(self.master, text="W. Pawn", command=lambda: self.label_img("wp")).grid(row=3, column=6)

        self.bk_button = Button(self.master, text="B. King", command=lambda: self.label_img("bk")).grid(row=5, column=1)
        self.bq_button = Button(self.master, text="B. Queen", command=lambda: self.label_img("bq")).grid(row=5,
                                                                                                         column=2)
        self.bb_button = Button(self.master, text="B. Bishop", command=lambda: self.label_img("bb")).grid(row=5,
                                                                                                          column=3)
        self.br_button = Button(self.master, text="B. Rook", command=lambda: self.label_img("br")).grid(row=5, column=4)
        self.bn_button = Button(self.master, text="B. Knight", command=lambda: self.label_img("bn")).grid(row=5,
                                                                                                          column=5)
        self.bp_button = Button(self.master, text="B. Pawn", command=lambda: self.label_img("bp")).grid(row=5, column=6)

    def next_image(self):
        self.current_index += 1
        self.show_photo()

    def prev_image(self):
        self.current_index -= 1
        self.show_photo()

    def open_src(self):
        self.src_path = filedialog.askdirectory()
        if self.src_path:
            for filename in os.listdir(self.src_path):
                if filename.endswith(".jpg"):
                    self.images.append(filename)
            self.show_photo()

    def set_save_path(self):
        self.save_path = filedialog.askdirectory()
        self.initialize_safe_path()

    def initialize_safe_path(self):
        try:
            os.makedirs(self.save_path + "/" + "wk")
            os.makedirs(self.save_path + "/" + "wq")
            os.makedirs(self.save_path + "/" + "wb")
            os.makedirs(self.save_path + "/" + "wr")
            os.makedirs(self.save_path + "/" + "wn")
            os.makedirs(self.save_path + "/" + "wp")
            os.makedirs(self.save_path + "/" + "empty")
            os.makedirs(self.save_path + "/" + "bk")
            os.makedirs(self.save_path + "/" + "bq")
            os.makedirs(self.save_path + "/" + "bb")
            os.makedirs(self.save_path + "/" + "br")
            os.makedirs(self.save_path + "/" + "bn")
            os.makedirs(self.save_path + "/" + "bp")
        except FileExistsError:
            pass

    def show_photo(self):
        if self.panel is not None:
            self.panel.destroy()
            self.panel.image = None

        if self.current_index < len(self.images):
            img = ImageTk.PhotoImage(Image.open(self.src_path + "/" + self.images[self.current_index]).resize((200, 200)))
            self.panel = Label(self.master, image=img)
            self.panel.image = img
            self.panel.grid(row=2, columnspan=6)
            print(str(self.current_index + 1) + "/" + str(len(self.images)))

    def label_img(self, label):
        if not self.save_path:
            self.save_path = "out"
            self.initialize_safe_path()
        shutil.copy2(self.src_path + "/" + self.images[self.current_index], self.save_path + "/" + label)
        self.next_image()


if __name__ == '__main__':
    # starting GUI
    root = Tk()
    gui = Window(root)
    root.mainloop()
