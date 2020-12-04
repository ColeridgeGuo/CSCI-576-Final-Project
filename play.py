import tkinter as tk
from tkinter import *
from tkinter import filedialog
import PIL.Image, PIL.ImageTk
import json
import sys
import os
from typing import NoReturn
from PIL import Image,ImageTk
from PIL.ImageTk import PhotoImage
import numpy as np
from tqdm import tqdm
import cv2
from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector
import pygame
import pyaudio
import wave
import simpleaudio as sa

WIDTH, HEIGHT = 640, 360
VID_LEN, FPS = 20, 24
K = 1  # search parameter for search area
B_SIZE = 20  # size of macroblocks


def read_image_RGB(fp: str) -> np.ndarray:
    binary_file = np.fromfile(fp, dtype='uint8')
    N = WIDTH * HEIGHT  # num of pixels in the image

    # store rgbs values into a 3-D array to be used for Image.fromarray()
    rgbs = np.zeros((HEIGHT, WIDTH, 3), dtype='uint8')
    for i in range(3):
        rgbs[:, :, i] = np.frombuffer(binary_file[i * N:(i + 1) * N],
                                      dtype='uint8').reshape((-1, WIDTH))
    return rgbs


def scene_detect(name: str, threshold: float = 30.0) -> int:
    video_manager = VideoManager([f'output_video\\{name}.avi'])

    # Warning: windows system use "\"; change to "/" if using linux or macos
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    base_timecode = video_manager.get_base_timecode()
    video_manager.set_downscale_factor()
    print(f'\nDetecting scenes in video "{name}"...')
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list(base_timecode)
    return len(scene_list) - 1


class VideoQuery:

    def __init__(self, fp: str):
        name = fp.split(os.sep)[-1]  # extract video name from file path
        category = name.split('_')[0]  # extract category from file path
        self.name = name
        self.listname = name
        self.category = category
        self.fp = fp
        self.pause = False
        # file paths to all frames
        dirs = sorted(os.listdir(self.fp), key=lambda x: int(x[5:-4]))
        fpaths = [f'{self.fp}\\{frame}' for frame in dirs[:VID_LEN * FPS]]
        # read in all frames' rgb values
        print(f'\nReading RGB values of video "{self.name}"...')
        self.data = np.array([read_image_RGB(fp) for fp in tqdm(fpaths)])

    def calc_motion(self) -> int:
        # recast datatype to avoid over/underflow
        self.data = self.data.astype('int64')
        total_motion = 0
        print(f'Calculating motion of video "{self.name}"...')
        with tqdm(total=VID_LEN * FPS - 1) as bar:
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
                bar.update(1)
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

    def to_video(self) -> NoReturn:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        vid_writer = cv2.VideoWriter(f'output_video/{self.name}.avi', fourcc,
                                     FPS, (WIDTH, HEIGHT))
        print(f'Converting "{self.name}" to .avi videos...')
        for frame in self.data:
            vid_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return

    # def show_video(self) -> NoReturn:
    #
    #     def update(ind: int) -> NoReturn:
    #         # the callback function to automatically update frames
    #         frame = frames[ind]  # get the frame at index
    #         ind += 1  # increment frame index
    #         if ind == VID_LEN * FPS:  # close GUI if reached end of the video
    #             # comment out if you want the video to persist after playing 20s
    #             root.destroy()
    #             return
    #         label.configure(image=frame)  # display frame
    #         root.after(40, update, ind)  # call to display next frame
    #
    #     root = tk.Tk()
    #     root.title("player")
    #     root.geometry('1920x1080')
    #     root['bg'] = '#333333'
    #     label = tk.Label(root,width = 640, height = 360)
    #     # convert rgb values to PhotoImage for display in GUI
    #     #   when you add a PhotoImage to a Tkinter widget, you must keep your
    #     #   own reference to the image object, hence storing them in a list
    #     frames = [PhotoImage(Image.fromarray(d)) for d in self.data]
    #     label.pack()
    #     root.mainloop()




    # def gui(self)->NoReturn:
    #     self.to_video()
    #     self.pause = False
    #     window = tk.Tk()
    #     window.title("Player")
    #     window.geometry('1920x1080')
    #     window['bg'] = '#333333'
    #
    #     frame1 = tk.LabelFrame(window,height = 360,width = 640)
    #     # frame1.pack(side = 'left')
    #     frame1.place(x = 250, y = 360)
    #
    #     frame2 = tk.LabelFrame(window, height=360, width=640)
    #     # frame2.pack(side='right')
    #     frame2.place(x = 1000, y = 360)
    #
    #     def pause_video(self):
    #         self.pause = True
    #
    #
    #
    #
    #     # Read wav file
    #
    #     def playFunction() -> NoReturn:
    #
    #         moviePath = f'output_video/{self.name}.avi'
    #         name2 = self.name.split('_')[0]
    #         audioPath = f'output_video/{name2}/{self.name}.wav'
    #
    #         movie = cv2.VideoCapture(moviePath)
    #         waitTime = int(1000 / movie.get(5))
    #         movieTime = int(movie.get(7) / movie.get(5))
    #         # playBar.configure(to_=movieTime)
    #
    #         wave_read = wave.open(audioPath, 'rb')
    #         sample_rate = wave_read.getframerate()
    #         audio_data = wave_read.readframes(sample_rate * 20)
    #         num_channels = wave_read.getnchannels()
    #         bytes_per_sample = wave_read.getsampwidth()
    #         play_obj = sa.play_buffer(audio_data, num_channels, bytes_per_sample, sample_rate)
    #         ret, frame = movie.read()
    #         if ret == True:
    #             movieFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    #             image1 = Image.fromarray(movieFrame).resize((640, 360))
    #             newCover = ImageTk.PhotoImage(image=image1)
    #             videoLabel.configure(image=newCover)
    #             videoLabel.image = newCover
    #         if not self.pause:
    #             videoLabel.after(25, playFunction)
    #             # videoLabel.update()
    #             # cv2.waitKey(24)
    #         # while movie.isOpened():
    #             ret, frame = movie.read()
    #             # if ret == True:
    #             #     movieFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA)
    #             #     image1 = Image.fromarray(movieFrame).resize((640,360))
    #             #     newCover = ImageTk.PhotoImage(image = image1)
    #             #     videoLabel.configure(image = newCover)
    #             #     videoLabel.image = newCover
    #             #     videoLabel.update()
    #             #     cv2.waitKey(24)
    #             # else:
    #             #
    #             #     break
    #
    #
    #     def playFunction2() -> NoReturn:
    #
    #
    #         videoType = self.listname.split('_')[0]
    #         moviePath = f'output_video/{self.category}/{self.listname}.avi'
    #         audioPath = f'output_video/{self.category}/{self.listname}.wav'
    #         print(audioPath)
    #         print(moviePath)
    #         movie = cv2.VideoCapture(moviePath)
    #         waitTime = int(1000 / movie.get(5))
    #         movieTime = int(movie.get(7) / movie.get(5))
    #         # playBar.configure(to_=movieTime)
    #
    #         wave_read = wave.open(audioPath, 'rb')
    #         sample_rate = wave_read.getframerate()
    #         audio_data = wave_read.readframes(sample_rate * 20)
    #         num_channels = wave_read.getnchannels()
    #         bytes_per_sample = wave_read.getsampwidth()
    #         play_obj = sa.play_buffer(audio_data, num_channels, bytes_per_sample, sample_rate)
    #         while movie.isOpened():
    #             ret, frame = movie.read()
    #             if ret == True:
    #                 movieFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    #                 image1 = Image.fromarray(movieFrame).resize((640, 360))
    #                 newCover = ImageTk.PhotoImage(image=image1)
    #                 videoLabe2.configure(image=newCover)
    #                 videoLabe2.image = newCover
    #                 videoLabe2.update()
    #                 cv2.waitKey(24)
    #             else:
    #                 break
    #
    #     def playFunction3() -> NoReturn:
    #
    #         moviePath = f'output_video/{self.name}.avi'
    #         name2 = self.name.split('_')[0]
    #         audioPath = f'output_video/{name2}/{self.name}.wav'
    #
    #         movie = cv2.VideoCapture(moviePath)
    #         waitTime = int(1000 / movie.get(5))
    #         movieTime = int(movie.get(7) / movie.get(5))
    #         # playBar.configure(to_=movieTime)
    #
    #
    #         while movie.isOpened():
    #             ret, frame = movie.read()
    #             if ret == True:
    #                 movieFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    #                 movieFrame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    #                 image1 = Image.fromarray(movieFrame).resize((640, 360))
    #                 image2 = Image.fromarray(movieFrame).resize((640, 360))
    #                 newCover = ImageTk.PhotoImage(image=image1)
    #                 newCover2 = ImageTk.PhotoImage(image=image2)
    #                 videoLabe2.configure(image=newCover)
    #                 videoLabe2.image = newCover
    #                 videoLabe2.update()
    #
    #                 videoLabel.configure(image=newCover)
    #                 videoLabel.image = newCover
    #                 videoLabel.update()
    #                 cv2.waitKey(24)
    #             else:
    #                 break
    #
    #
    #     videoLabel = tk.Label(frame1)
    #     videoLabel.pack()
    #
    #     videoLabe2 = tk.Label(frame2)
    #
    #     videoLabe2.pack()
    #
    #     playButton = tk.Button(window, text='play', width=15,
    #                    height=2, command = playFunction)
    #     playButton.pack()
    #     playButton.place(x= 0,y=840)
    #
    #     playButton2 = tk.Button(window, text='play', width=15,
    #                            height=2, command=playFunction2)
    #     playButton2.pack()
    #     playButton2.place(x=300, y=840)
    #     # playButton.bind("<ButtonRelease-1>", )
    #
    #
    #     stopButton = tk.Button(window, text='stop', width=15,
    #                            height=2, command = pause_video)
    #     stopButton.pack()
    #     stopButton.place(x=150, y=840)
    #     # playButton.bind("<ButtonRelease-1>", )
    #
    #
    #
    #
    #     var1 = tk.StringVar()
    #     l1 = tk.Label(window, bg='yellow', width=4, textvariable=var1)
    #     l1.pack()
    #
    #     def print_selection():
    #         self.listname = top5List.get(top5List.curselection())
    #         var1.set(self.listname)
    #
    #     b1 = tk.Button(window, text='ok', width=15,
    #                    height=2, command=print_selection)
    #     b1.pack()
    #
    #     var2 = tk.StringVar()
    #     topList = 'ads_0', 'ads_1', 'ads_2', 'ads_3'
    #     var2.set((topList))
    #     top5List = tk.Listbox(window, listvariable=var2, font=('Times', 25))
    #     top5List.pack()
    #     top5List.place(height=300, width=200)
    #
    #
    #
    #     def print_selection2(v):
    #         playBarDesc.config(text='you have selected ' + v)
    #
    #     playBar = tk.Scale(window, label='try me', from_=0, to=20, orient=tk.HORIZONTAL,
    #                  length=640, showvalue=0, tickinterval=2, resolution=0.01, command=print_selection2)
    #
    #     playBar.pack()
    #     playBar.place(y=1000)
    #
    #     playBarDesc = tk.Label(window, bg='yellow', text='empty')
    #     playBarDesc.pack()
    #     playBarDesc.place(x= 641,y=1000)
    #     window.mainloop()


class videoGUI:
    def __init__(self, window, window_title,query_name):
        self.name = query_name.split(os.sep)[-1]  # extract video name from file path
        self.category = self.name.split('_')[0]  # extract category from file path


        self.moviePath = f'output_video/{self.name}.avi'
        self.audioPath = f'output_video/{self.category}/{self.name}.wav'
        print(self.audioPath)
        self.listath = self.moviePath
        self.window = window
        self.window.title(window_title)
        self.window.geometry('1920x1080')
        self.window['bg'] = '#333333'

        self.frame1 = Frame(window, height=360, width=640)
        self.frame1.pack()
        self.frame1.place(x = 250, y = 360)
        self.canvas = Canvas(self.frame1,height=360, width=640)
        self.canvas.pack()


        self.frame2 = Frame(window, height=360, width=640)
        self.frame2.pack()
        self.frame2.place(x = 1000, y = 360)
        self.canvas2 = Canvas(self.frame2, height=360, width=640)
        self.canvas2.pack()

        wave_read = wave.open(self.audioPath, 'rb')
        sample_rate = wave_read.getframerate()
        audio_data = wave_read.readframes(sample_rate * 20)
        num_channels = wave_read.getnchannels()
        bytes_per_sample = wave_read.getsampwidth()
        #play_obj = sa.play_buffer(audio_data, num_channels, bytes_per_sample, sample_rate)

        bottom_frame = tk.LabelFrame(self.window)
        bottom_frame.pack(side=BOTTOM, pady=0)

        bottom_frame2 = tk.LabelFrame(self.window)
        bottom_frame2.pack(side=BOTTOM, pady=50)

        # Select Button

        self.btn_select = Button(bottom_frame, text="Loading Query", width=15, command = self.open_file)
        self.btn_select.grid(row=0, column=0)

            # Play Button
        self.btn_play = Button(bottom_frame, text="Play audio", width=15,command = self.play_video)
        self.btn_play.grid(row=0, column=1)

            # Pause Button
        self.btn_pause = Button(bottom_frame, text="Pause", width=15, command = self.pause_video)
        self.btn_pause.grid(row=0, column=2)

            # Resume Button
        self.btn_resume = Button(bottom_frame, text="resume", width=15,command = self.resume_video)
        self.btn_resume.grid(row=0, column=3)

        # Select Button 2
        self.btn_select2 = Button(bottom_frame2, text="Select file from list", width=15, command = self.open_list_file)
        self.btn_select2.grid(row=0, column=0)

        # Play Button
        self.btn_play2 = Button(bottom_frame2, text="Play", width=15,command = self.play_video2)
        self.btn_play2.grid(row=0, column=1)

        # Pause Button
        self.btn_pause2 = Button(bottom_frame2, text="Pause", command = self.pause_video2)
        self.btn_pause2.grid(row=0, column=2)

        # Resume Button
        self.btn_resume2 = Button(bottom_frame2, text="resume", width=15,command = self.resume_video2)
        self.btn_resume2.grid(row=0, column=3)

        self.var1 = tk.StringVar()
        self.l1 = tk.Label(self.window, bg='yellow', width=4, textvariable=self.var1)
        self.l1.pack()

        self.b1 = tk.Button(window, text='Select', width=15, height=2, command=self.print_selection)
        self.b1.pack()
        self.var2 = tk.StringVar()
        self.topList = 'ads_0', 'ads_1', 'ads_2', 'ads_3'
        self.var2.set((self.topList))
        self.top5List = tk.Listbox(window, listvariable=self.var2, font=('Times', 25))
        self.top5List.pack()
        self.top5List.place(height=300, width=200)

        self.delay = 25  # ms
        self.window.mainloop()

        # Get video's information
        # Open the video file


    #top 5 list
    def print_selection(self):
        self.listname = self.top5List.get(self.top5List.curselection())
        self.var1.set(self.listname)

    def open_file(self):

        self.pause = False


        # Open the video file
        self.cap = cv2.VideoCapture(self.moviePath)

        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.canvas.config(width=self.width, height=self.height)

    def open_list_file(self):

        self.pause = False
        # Open the video file
        self.listpath = f'output_video/{self.listname}.avi'

        self.cap = cv2.VideoCapture(self.listpath)

        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.canvas2.config(width=self.width, height=self.height)


    #Get each frame from video
    def get_frame(self):
        try:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        except:
                print('File not Found')

    # def play_audio(self):
    #     wave_read = wave.open(self.audioPath, 'rb')
    #     #     sample_rate = wave_read.getframerate()
    #     #     audio_data = wave_read.readframes(sample_rate * 20)
    #     #     num_channels = wave_read.getnchannels()
    #     #     bytes_per_sample = wave_read.getsampwidth()
    #     #     play_obj = sa.play_buffer(audio_data, num_channels, bytes_per_sample, sample_rate)
    #        # play_obj.wait_done()


    # play_obj.wait_done()
    def play_video(self):
        ret, frame = self.get_frame()

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))

            self.canvas.create_image(0, 0, image=self.photo, anchor=NW)
        if self.pause:
            self.window.after_cancel(self.after_id)
        else:
            self.after_id = self.window.after(self.delay, self.play_video)



    def play_video2(self):
        ret, frame = self.get_frame()

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))

            self.canvas2.create_image(0, 0, image=self.photo, anchor=NW)
        if self.pause:
            self.window.after_cancel(self.after_id)
        else:
            self.after_id = self.window.after(self.delay, self.play_video2)


    #button stop and resume
    def pause_video(self):
        if self.pause == False:
            self.pause = True
        else:
            self.pause = False
            self.play_video()

    def pause_video2(self):
        if self.pause == False:
            self.pause = True
        else:
            self.pause = False
            self.play_video2()

    def resume_video(self):

        self.open_file()
        self.get_frame()
        self.play_video()

    def resume_video2(self):
        self.open_list_file()
        self.get_frame()
        self.play_video2()
    #Release the video source when the object is destroyed
    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

if __name__ == '__main__':
    args = sys.argv
    if len(args) > 1:  # if input video specified, process single video
        fq = args[1]
        print(fq)

        vid_name = fq.split('\\')[-1]

        vq = VideoQuery(fq)
        vq.to_video()
        gui = videoGUI(Tk(),'Player',fq)
        # vq.gui()
        #gui = videoGUI(Tk(),'player')

        sc = scene_detect(vid_name, threshold=20)
        sys.exit()
    #fpath is the address of query's rgb data; fpath_wav is the address of wav data
    fpath = f"C:/Users/Tooth/OneDrive/Desktop/USC/2020_fall/csci576/assignment/project/Data_rgb"
    #fpath = r"C:\Users\Tooth\OneDrive\Desktop\USC\2020_fall\csci576\assignment\project\Data_rgb"

    categories = next(os.walk(fpath))[1]

    cat_paths = [os.path.join(fpath, cat) for cat in categories]
    vid_names = [next(os.walk(cat))[1] for cat in cat_paths]
    #print(vid_names)
    # # commented code below used for converting form rgb to .avi video files
    # vid_paths = [[os.path.join(cat_paths[i], v) for v in cat]
    #              for i, cat in enumerate(vid_names)]
    # videos = [[VideoQuery(vid) for vid in cat] for cat in vid_paths]
    # to_vid = [[vid.to_video() for vid in cat] for cat in videos]
    scenes = {categories[i]:
                  {vid_names[i][j]: scene_detect(vid_names[i][j], 25)
                   for j, vid in enumerate(c)}
              for i, c in enumerate(vid_names)}
    scenes = {"feature_name": "scene_cuts", "values": scenes}
    with open('data.json', 'w') as f:
        json.dump(scenes, f, indent=2, sort_keys=True)