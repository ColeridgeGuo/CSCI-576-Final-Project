import tkinter as tk
import sys
import os
from PIL import Image
from PIL.ImageTk import PhotoImage
from progressbar import progressbar

WIDTH, HEIGHT = 640, 360
VID_LEN, FPS = 20, 24

class VideoDisplay:
    
    def __init__(self):
        self.root = tk.Tk()  # base GUI container
        self.frames = []  # list of PhotoImages to display
        # When you add a PhotoImage or other Image object to a Tkinter widget,
        # you must keep your own reference to the image object.
        self.label = tk.Label(self.root)  # tkinter label to show image
        
    def read_image_RGB(self, fpath: str) -> bytes:
        with open(fpath, 'rb') as file:
            rgbs = file.read()
        n = WIDTH * HEIGHT  # num of pixels in the image
        
        # convert the format of rgbs values from
        # R1, R2, ..., Rn, G1, G2, ..., Gn, B1, B2, ..., Bn to
        # R1, G1, B1, R2, G2, B2, ..., Rn, Gn, Bn
        rgbs = zip(rgbs[:n], rgbs[n:n * 2], rgbs[n * 2:])
        return bytes(v for tup in rgbs for v in tup)
    
    def update(self, ind: int) -> None:
        frame = self.frames[ind]  # get the frame at index
        ind += 1  # increment frame index
        if ind == VID_LEN * FPS:  # return if reached end of the video
            return
        self.label.configure(image=frame)  # display frame
        self.root.after(40, self.update, ind)  # call to display next frame
        
    def show_image(self, fp: str) -> None:
        vid_name = fp.split('/')[-1]  # extract video name from file path
        
        # file paths to all frames
        dirs = sorted(os.listdir(fp), key=lambda x: int(x[5:-4]))
        fpaths = [f'{fp}/{frame}' for frame in dirs[:VID_LEN * FPS]]
        
        # read in all frames' rgb values
        print(f"\n\033[95mReading rgb values of video '{vid_name}': \033[0m")
        data = [self.read_image_RGB(fpath) for fpath in progressbar(fpaths)]
        
        # convert rgb values to PhotoImage for display in GUI
        self.frames = [PhotoImage(Image.frombytes('RGB', (WIDTH, HEIGHT), d))
                       for d in data]
        
        self.label.pack()
        # callback update() to automatically update frames
        self.root.after(0, self.update, 0)
        self.root.mainloop()
        
        
if __name__ == '__main__':
    args = sys.argv
    file_path = args[1]
    vd = VideoDisplay()
    vd.show_image(file_path)
