import tkinter as tk
import sys
from PIL import Image, ImageTk

WIDTH, HEIGHT = 640, 480

class VideoDisplay:
    
    def __init__(self):
        self.root = tk.Tk()  # base GUI container
        self.data = []  # bytes to store RGB values
        self.label = tk.Label(self.root)  # tkinter label to show image

    def read_image_RGB(self, fpath: str) -> None:
        with open(fpath, 'rb') as file:
            rgbs = file.read()
        n = WIDTH*HEIGHT  # num of pixels in the image
        
        # convert the dimension of rgbs values from
        # R1, R2, ..., Rn, G1, G2, ..., Gn, B1, B2, ..., Bn to
        # R1, G1, B1, R2, G2, B2, ..., Rn, Gn, Bn
        rgbs = zip(rgbs[:n], rgbs[n:n*2], rgbs[n*2:])
        self.data = bytes(v for tup in rgbs for v in tup)
        
    def show_image(self, fpath: str) -> None:
        self.read_image_RGB(fpath)
        img = ImageTk.PhotoImage(Image.frombytes('RGB', (WIDTH, HEIGHT), self.data))
        self.label.configure(image=img)
        self.label.pack()
        self.root.mainloop()
        
        
if __name__ == '__main__':
    args = sys.argv
    file_path = args[1]
    vd = VideoDisplay()
    vd.show_image(file_path)
