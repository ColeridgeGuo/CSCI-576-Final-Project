import tkinter as tk
import sys
import os
from typing import NoReturn
from PIL import Image
from PIL.ImageTk import PhotoImage
import numpy as np

WIDTH, HEIGHT = 640, 360
VID_LEN, FPS = 20, 24
K = 1  # search parameter for search area
B_SIZE = 20  # size of macroblocks

class VideoDisplay:
    
    def __init__(self):
        self.root = tk.Tk()  # base GUI container
        self.frames = []  # list of PhotoImages to display
        # When you add a PhotoImage or other Image object to a Tkinter widget,
        # you must keep your own reference to the image object.
        self.data = []
        self.label = tk.Label(self.root)  # tkinter label to show image
        
    def read_image_RGB(self, fpath: str) -> np.ndarray:
        binary_file = np.fromfile(fpath, dtype='uint8')
        N = WIDTH * HEIGHT  # num of pixels in the image
        
        # store rgbs values into a 3-D array to be used for Image.fromarray()
        rgbs = np.zeros((HEIGHT, WIDTH, 3), dtype='uint8')
        for i in range(3):
            rgbs[:, :, i] = np.frombuffer(binary_file[i*N:(i+1)*N],
                                          dtype='uint8').reshape((-1, WIDTH))
        return rgbs
    
    def update(self, ind: int) -> NoReturn:
        frame = self.frames[ind]  # get the frame at index
        ind += 1  # increment frame index
        if ind == VID_LEN * FPS:  # return if reached end of the video
            return
        self.label.configure(image=frame)  # display frame
        self.root.after(40, self.update, ind)  # call to display next frame
        
    def show_image(self, fp: str) -> NoReturn:
        vid_name = fp.split('/')[-1]  # extract video name from file path
        
        # file paths to all frames
        dirs = sorted(os.listdir(fp), key=lambda x: int(x[5:-4]))
        fpaths = [f'{fp}/{frame}' for frame in dirs[:VID_LEN * FPS]]
        
        # read in all frames' rgb values
        print(f"Reading rgb values of video '{vid_name}'...")
        self.data = [self.read_image_RGB(fpath) for fpath in fpaths]
        # convert rgb values to PhotoImage for display in GUI
        self.frames = [PhotoImage(Image.fromarray(d)) for d in self.data]
        
        self.label.pack()
        # callback update() to automatically update frames
        self.root.after(0, self.update, 0)
        self.root.mainloop()
        
        
if __name__ == '__main__':
    args = sys.argv
    file_path = args[1]
    vd = VideoDisplay()
    vd.show_image(file_path)
