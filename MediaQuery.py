import tkinter as tk
import sys
import os
from typing import NoReturn
from PIL import Image
from PIL.ImageTk import PhotoImage
import numpy as np
import time
from progressbar import ProgressBar, progressbar

WIDTH, HEIGHT = 640, 360
VID_LEN, FPS = 20, 24
K = 1  # search parameter for search area
B_SIZE = 20  # size of macroblocks


def read_image_RGB(fpath: str) -> np.ndarray:
    binary_file = np.fromfile(fpath, dtype='uint8')
    N = WIDTH * HEIGHT  # num of pixels in the image
    
    # store rgbs values into a 3-D array to be used for Image.fromarray()
    rgbs = np.zeros((HEIGHT, WIDTH, 3), dtype='uint8')
    for i in range(3):
        rgbs[:, :, i] = np.frombuffer(binary_file[i*N:(i+1)*N],
                                      dtype='uint8').reshape((-1, WIDTH))
    return rgbs

def RGB_to_YUV(rgb: np.ndarray) -> np.ndarray:
    return np.dot([.299, .587, 0.114], rgb)


class MediaQuery:
    
    def __init__(self, fpath: str):
        vid_name = fpath.split('/')[-1]  # extract video name from file path
        self.vid_name = vid_name
        self.fp = fpath
        
        self.root = tk.Tk()  # base GUI container
        self.frames = None  # list of PhotoImages to display
        # When you add a PhotoImage or other Image object to a Tkinter widget,
        # you must keep your own reference to the image object.
        self.data = None
        self.label = tk.Label(self.root)  # tkinter label to show image
        
    def calc_motion(self) -> int:
        # recast datatype to avoid over/underflow
        self.data = self.data.astype('int64')
        total_motion = 0
        print(f'Calculating motion of video "{self.vid_name}" ...')
        with ProgressBar(max_value=VID_LEN * FPS - 1) as bar:
            for frame_idx in range(VID_LEN * FPS - 1):
                for y in range(0, HEIGHT, B_SIZE):
                    for x in range(0, WIDTH, B_SIZE):
                        # for each block, find the search area with k
                        start_x, start_y = max(0, x - K), max(0, y - K)
                        end_x, end_y = min(WIDTH - B_SIZE, x + K), \
                                       min(HEIGHT - B_SIZE, y + K)
                        # calculate SAD for each target macro-block and compare
                        # it with the current macro-block to find the best match
                        min_SAD = np.inf
                        best_match = (0, 0)
                        for i in range(start_y, end_y + 1):
                            for j in range(start_x, end_x + 1):
                                sad = self.calc_SAD(frame_idx, i, j, y, x)
                                if sad < min_SAD:
                                    min_SAD = sad
                                    best_match = (i, j)
                        if best_match != (y, x):
                            total_motion += 1
                bar.update(frame_idx)
        return total_motion
    
    def calc_SAD(self, frame_idx: int,
                 c_y: int, c_x: int, n_y: int, n_x: int) -> np.ndarray:
        """
        :param frame_idx: frame index
        :param c_y: current frame Y-coordinate
        :param c_x: current frame X-coordinate
        :param n_y: next frame Y-coordinate
        :param n_x: next frame X-coordinate
        :return: The SAD of Y-values of the macro-block and its co-located
            macro-block in the next frame
        """
        curr_RGB = self.data[frame_idx, c_y:c_y + B_SIZE, c_x:c_x + B_SIZE]
        # curr_YUV = np.apply_along_axis(RGB_to_YUV, 2, curr_RGB)
        next_RGB = self.data[frame_idx + 1, n_y:n_y + B_SIZE, n_x:n_x + B_SIZE]
        # next_YUV = np.apply_along_axis(RGB_to_YUV, 2, next_RGB)
        diff = next_RGB - curr_RGB
        # diff = next_YUV - curr_YUV
        return np.sum(np.abs(diff))
    
    def update(self, ind: int) -> NoReturn:
        frame = self.frames[ind]  # get the frame at index
        ind += 1  # increment frame index
        if ind == VID_LEN * FPS:  # close GUI if reached end of the video
            # comment out if you want the video to persist after playing 20s
            self.root.destroy()
            return
        self.label.configure(image=frame)  # display frame
        self.root.after(40, self.update, ind)  # call to display next frame
        
    def show_video(self) -> NoReturn:
        
        # file paths to all frames
        dirs = sorted(os.listdir(self.fp), key=lambda x: int(x[5:-4]))
        fpaths = [f'{self.fp}/{frame}' for frame in dirs[:VID_LEN * FPS]]
        
        # read in all frames' rgb values
        print(f'Reading RGB values of video "{self.vid_name}"...')
        self.data = np.array([read_image_RGB(fpath)
                              for fpath in progressbar(fpaths)])
        # convert rgb values to PhotoImage for display in GUI
        self.frames = [PhotoImage(Image.fromarray(d)) for d in self.data]
        
        self.label.pack()
        # callback update() to automatically update frames
        self.root.after(0, self.update, 0)
        self.root.mainloop()
        
        
if __name__ == '__main__':
    args = sys.argv
    file_path = args[1]
    vd = MediaQuery(file_path)
    vd.show_video()

    motion_in_vid = vd.calc_motion()
    print(f'The motion in video "{vd.vid_name}": {motion_in_vid}')
