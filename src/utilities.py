# =============================================================================
# XAVAD: Unified Explainable Anonymized Video Anomaly Detection (VAD) System
# =============================================================================
# This script provides utils functions for XAVAD

# Author: Mulugeta W. Asres
# Email: muleina2000@gmail.com
# Date: October 2025
# =============================================================================

import os, sys, io, re, gc, itertools, functools, copy, shutil
import json
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
import queue 
import torch
import torch.nn.functional as F
import cv2
import moviepy.editor as mpy
import time
from pathlib import Path
import PIL
from typing import List, Tuple, Dict, Union, Optional, Literal, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

###########################################
# Util Classes
class ProcTimer():
    """
    Utility class for measuring process time.
    """
    def __init__(self):
        self.start_time = time.time()
        self.end_time = 0
        self.proc_time = 0
        self.time_status = True

    def restart(self):
        """
        Restart the timer
        """
        self.start_time = time.time()
        self.end_time = 0
        self.time_status = True

    def stop(self):
        """
        Stop the timer
        """
        self.time_status = False
        self.end_time = time.time()
        self.proc_time = self.end_time - self.start_time

    def get_proctime(self, time_format: str = "s") -> float:
        """
        Get the elapsed process time.

        Args:
            time_format (str): Format of time (default: "s" for seconds).

        Returns:
            float: Process time in seconds.
        """
        if self.time_status:
            self.end_time = time.time()
            self.proc_time = self.end_time - self.start_time

        if time_format == "s":
            return self.proc_time

    def display_proctime(self, time_format: str = "s") -> None:
        """
        Print the process time.

        Args:
            time_format (str): Format of time (default: "s" for seconds).
        """
        print("process time: {:0.5} seconds.".format(self.get_proctime(time_format=time_format)))

class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src, isthreaded=False):
        self.stream = cv2.VideoCapture(src)
        self.isthreaded = isthreaded

        # Check if camera opened successfully
        if (self.stream.isOpened() == False): 
            print("Error opening video stream or file")

        (self.grabbed, self.frame) = self.stream.read()
        
        self.stopped = False

    def start(self):    
        if self.isthreaded:
            Thread(target=self.get, args=()).start()
        return self

    def get(self):
        if self.isthreaded:
            while (not self.stopped) and self.stream.isOpened():
                if not self.grabbed:
                    self.stop()
                else:
                    (self.grabbed, self.frame) = self.stream.read()
        else:
            if self.stream.isOpened():
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True
        
class VideoCameraThreadQueue:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread for real-time video streaming.
    """

    def __init__(self, src, timeout: float = 0.0, fps: int = 5, width: int = 640, height: int = 480, **kwargs):
        """
        Initialize the threaded video capture.

        Args:
            src: Video source (file path or camera index).
            timeout (float): Not used.
            fps (int): Frames per second.
            width (int): Frame width.
            height (int): Frame height.
            **kwargs: Additional arguments.
        """     
        self.stopped = True

        try:
            self.stream = cv2.VideoCapture(src)
            # set video properties
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.stream.set(cv2.CAP_PROP_FPS, fps)

        except Exception as ex:
            print(f"Video capture init error: {ex}. Retying with CAP_DSHOW...")
            try:
                self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
                # set video properties
                self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.stream.set(cv2.CAP_PROP_FPS, fps)
            except Exception as ex:
                print(f"VideoCapture init error: {ex}")
                self.stream = None
                self.stop()

        self.start()
        # _ = self.stream.read()
        self.stopped = False

    def start(self):   
        """
        Start the video reading thread.
        """
        try:
            self.q = queue.Queue()
            self.stream_thread = Thread(target=self._reader, daemon=True)
            self.stopped = False
            self.stream_thread.start()
        except Exception as ex:
            print(f"Thread init failed: {ex}")
            self.stopped = True
            self.stop()

        return self

    def _reader(self):
        """
        Internal thread function to read frames and put them in a queue.
        """
        while (not self.stopped) and self.stream.isOpened():
            (self.grabbed, self.frame) = self.stream.read()
            if not self.grabbed:
                break

            if not self.q.empty():
                try:
                    self.q.get_nowait()   
                except queue.Empty:
                    pass

            self.q.put(self.frame)

    def read(self):
        """
        Get the latest frame from the queue.

        Returns:
            np.ndarray or None: The latest frame, or None if not available.
        """
        try:
            return self.q.get()
        except queue.Empty:
            if self.stream is None:
                self.stop()
                return None
            
            if not self.stream_thread.is_alive() and (not self.stopped):
                print("stream thread is dead. restarting....")
                try:
                    self.stream_thread = Thread(target=self._reader, daemon=True)
                    self.stream_thread.start()
                    self.stopped = False
                except Exception as ex:
                    print(f"VideoCapture init error: {ex}")
            return None
    
    def stop(self):
        """
        Stop the video stream and thread.
        """
        self.stream = None
        self.stopped = True
    
# End Util Classes
###########################################
###########################################
# Start Decorators
def timer_eval(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try: 
            print(f"{func.__qualname__}: ") 
        except: 
            pass
        timer = ProcTimer()
        timer.restart()
        result = func(*args, **kwargs)
        timer.stop()
        timer.display_proctime()
        return result
    return wrapper

def clean_cuda(func):
    def wrapper(*args, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()
        result = func(*args, **kwargs)
        gc.collect()
        torch.cuda.empty_cache()
        return result
    return wrapper

def process_logger(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try: 
            print(f"Processing {func.__qualname__}: ") 
        except: 
            pass
        result = func(*args, **kwargs)
        print(f"{func.__qualname__} is done!")
        return result
    return wrapper
    
# End Decorators
###########################################

def plot_image_grid(imgs, rows, cols=1, return_fig=True, border=0, border_color="white", color_format="RGB"):
    N = len(imgs)
    try:
        assert N == rows*cols, "The number of images does not match the specified grid dimensions."
    except AssertionError:
        # raise ValueError("The number of images does not match the specified grid dimensions.")
        rows = int(np.ceil(N/cols))

    w, h = imgs[0].shape[:2][::-1] if isinstance(imgs[0], (torch.Tensor, np.ndarray)) else imgs[0].size
    w, h = w+2*border, h+2*border
    # print(w,h, color_format)
    
    grid = PIL.Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    for i, img in enumerate(imgs):
        if isinstance(img, (torch.Tensor, np.ndarray)):
            if color_format == "BGR":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = F.to_pil_image(img) 
            img = PIL.Image.fromarray(img) 
        grid.paste(PIL.ImageOps.expand(img, border=border, fill=border_color) if border>0 else img, box=(i%cols*w, i//cols*h))
    if return_fig:
        return grid
    else:
        grid.show()

def get_video_meta(source_datapath: str):
    video_getter = VideoGet(source_datapath, isthreaded=False).start()
    video_fps = round(video_getter.stream.get(cv2.CAP_PROP_FPS))
    video_length = int(video_getter.stream.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = video_length/video_fps if video_length else 0.0
    return {'fps':video_fps, 'video_duration':video_duration, 'video_length':video_length}

def show_video(video_path: Path):
    video_cap = cv2.VideoCapture(str(video_path))
    while video_cap.isOpened():
        ret,  frame = video_cap.read()
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(25)
        if key == ord("q"):
            break
    video_cap.release()
    cv2.destroyAllWindows()

def videofile_loader(
                    source_datapath: str,
                    start_sec: int = 0,
                    clip_duration: int = None,
                    fps: int = None,
                    target_resolution: Tuple[int, int] = None,
                    **kwargs
                ) -> Tuple[list, dict]:
    """
    Load frames from a video file.

    Args:
        source_datapath (str): Path to the video file.
        start_sec (int): Start time in seconds.
        clip_duration (int): Duration of the clip to load (seconds).
        fps (int): Frames per second to sample.
        target_resolution (Tuple[int, int]): Target resolution (width, height).
        **kwargs: Additional arguments.

    Returns:
        Tuple[list, dict]: List of frames (BGR) and video metadata, [height, width, 3].
    """
    print("videofile_loader...")

    timer = ProcTimer()
    timer.restart()

    mpyVidObj = mpy.VideoFileClip(source_datapath, target_resolution=target_resolution)
    mpyVidObj = mpyVidObj if fps is None else mpyVidObj.set_fps(fps)
    mpyVidObj: mpy.VideoClip = mpyVidObj.subclip(start_sec, start_sec + mpyVidObj.duration if clip_duration is None else start_sec + clip_duration)
    video_duration = mpyVidObj.duration
    video_fps = round(mpyVidObj.fps)
    # frame_size = mpyVidObj.size
    # num_frames = round(video_duration*video_fps)
    frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in mpyVidObj.iter_frames(fps=None, with_times=False, logger=None, dtype="uint8")]
    num_frames = len(frames)
    frame_size = frames[0].shape
    
    timer.display_proctime()
    
    video_meta = {
                    "num_frames": num_frames,
                    "fps": video_fps,
                    "duration": video_duration,
                    "frame_size": frame_size
                }
    print(f"loaded frame duration: {video_duration} secs, frame length: {num_frames}, fps: {video_fps}, resolution: {frame_size}")
    return frames, video_meta

def convertcolor_hexstr2tuple(color: str, color_format: str = "rgb") -> Tuple[int, int, int]:
    """
    Convert a hex color string to an RGB or BGR tuple.

    Args:
        color (str): Hex color string (e.g., "#ff0000").
        color_format (str): "rgb" or "bgr".

    Returns:
        Tuple[int, int, int]: Color tuple.
    """
    if isinstance(color, str):
        hex2rgb = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        if color_format == "bgr":
            hex2rgb = hex2rgb[::-1]
        color = hex2rgb
    return color

def load_image(img_path, img_size=None) -> np.ndarray:
    # Load image file
    print("loading image file...")
    img = cv2.imread(rf"{img_path}") # BGR

    if img_size is not None:
        # Resize image to input size
        print(f"input image size: {img.shape}")
        img = image_resize(img, img_size, mode="cv", ref="wh")
    
    print(f"input image resize: {img.shape}")
    return img
    
def image_resize(x: np.ndarray, target_resolution: Tuple[int, int], mode: str = "cv", ref: str = "w") -> np.ndarray:
    """
    image_resize
    """
    if mode == "pil": # rgb
        ispil = not isinstance(x, np.ndarray)
        if not ispil:
            x = PIL.Image.fromarray(x)
        x_input_size = (x.width, x.height)
        w, h = target_resolution
        if ref == "w":
            alpha_size_scale = w/x_input_size[0]
            h = int(x_input_size[1]*alpha_size_scale)
        elif ref == "h":
            alpha_size_scale = h/x_input_size[1]
            w = int(x_input_size[0]*alpha_size_scale)
        else:
            pass
        anony_base_size = (w, h)
        x = x.resize(anony_base_size, PIL.Image.ANTIALIAS)
        if ispil: 
            return x
        return np.asarray(x)
    elif mode == "cv": #bgr
        x_input_size = x.shape[:-1][::-1]
        w, h = target_resolution
        if ref == "w":
            alpha_size_scale = w/x_input_size[0]
            h = int(x_input_size[1]*alpha_size_scale)
        elif ref == "h":
            alpha_size_scale = h/x_input_size[1]
            w = int(x_input_size[0]*alpha_size_scale)
        else:
            pass
        anony_base_size = (w, h)
        x = cv2.resize(x, anony_base_size)
        return x
    else:
        raise Exception(f"Undefined mode {mode}. choose from cv or pil.")
           
def resize_with_pad(image: np.ndarray, new_shape: Tuple[int, int], padding_color: Tuple[int] = (255, 255, 255)) -> np.ndarray:
    """
    Resize image with padding while maintaining the aspect ratio.

    Args:
        image (np.ndarray): Input image.
        new_shape (Tuple[int, int]): Target (width, height): Expected (width, height) of new image. 
        padding_color (Tuple[int, int, int]): Padding color (BGR): Tuple in BGR of padding color.

    Returns:
        np.ndarray: Resized image with padding.
    """
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape)) / max(original_shape)
    new_size = tuple([int(x * ratio) for x in original_shape])
    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    print(ratio, new_size, delta_w, delta_h)
    print(top, bottom, left, right)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return image

def add_pad(image: np.ndarray, 
            new_shape: Tuple[int, int], 
            padding_color: Tuple[int] = (255, 255, 255)
        ) -> np.ndarray:
    """
    Add padding to an image to reach the target shape.

    Args:
        image (np.ndarray): Input image.
        new_shape (Tuple[int, int]): Target (width, height): Expected (width, height) of new image.
        padding_color (Tuple[int, int, int]): Padding color (BGR).

    Returns:
        np.ndarray: Image with added padding.
    """
    original_shape = (image.shape[1], image.shape[0])
    delta_w = new_shape[0] - original_shape[0]
    delta_h = new_shape[1] - original_shape[1]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    print(delta_w, delta_h)
    print(top, bottom, left, right)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return image

def remove_pad(image: np.ndarray, 
                new_shape: Tuple[int, int]
            ) -> np.ndarray:
    """
    Remove padding from an image to reach the target shape.

    Args:
        image (np.ndarray): Input image.
        new_shape (Tuple[int, int]): Target (width, height): Expected (width, height) of new image.

    Returns:
        np.ndarray: Image with padding removed.
    """
    original_shape = (image.shape[1], image.shape[0])
    delta_w = original_shape[0] - new_shape[0] 
    delta_h = original_shape[1] - new_shape[1]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    print(original_shape, delta_w, delta_h)
    print(top, bottom, left, right)
    image = image[bottom:original_shape[1]-top, left:original_shape[0]-right]
    return image

def save_figure(fig,
                filename: str,
                filedirpath: str,
                isshow: bool = False,
                issave: bool = True,
                dpi: int = 300,
                format: str = ".jpg"
            ) -> None:
    """
    Save a matplotlib figure to disk.

    Args:
        filename (str): Name of the file.
        fig: Matplotlib figure object.
        filedirpath (str): Directory to save the figure.
        isshow (bool): Whether to display the figure.
        issave (bool): Whether to save the figure.
        dpi (int): Dots per inch for saved figure.
        format (str): File format (e.g., ".jpg").
    """
    if issave:
        filepath = rf"{filedirpath}/{filename}.{format}"
        print("saving to ", filedirpath)
        fig.savefig(Path(filepath), dpi=dpi, bbox_inches='tight')
        if isshow:
            plt.show(fig)
        else:
            plt.close()

    if isshow:
        plt.show(fig)

def save_video(frame_list: list, fps: int, filename: str, filedirpath: str) -> None:
    """
    Save a list of frames as a video file.
    Args:
        frame_list (list): List of frames (BGR).
        fps (int): Frames per second.
        filename (str): Name of the video file.
        filedirpath (str): Directory to save the video file.
    """
    filepath = rf'{filedirpath}/{filename}.mp4'
    print("saving video to ", filepath)
    mpyVidObj = mpy.ImageSequenceClip(frame_list, fps=fps)
    mpyVidObj.write_videofile(filepath, fps=fps, codec="libx264")
    mpyVidObj.close()

def save_json(filename: Optional[str] = None, data: Dict[Any, Any] = {}, filepath: Optional[str] = None, sort_keys: bool = True, verbose=0):
    """save dict as json 

    Args:
        filename (_type_, optional): _description_. Defaults to None.
        data (dict, optional): _description_. Defaults to {}.
        filepath (_type_, optional): _description_. Defaults to None.
        verbose (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    class MyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return super(MyEncoder, self).default(obj)

    if filename is not None:
        assert filepath is not None
        filepath = Path(filepath) / filename
        
    filepath = Path(filepath) if isinstance(filepath, str) else filepath
    if verbose: print('saving ', filepath, end='\r')
    with open(filepath, 'w') as fhandle:
        fhandle.write(json.dumps(data, indent=4,
                                 sort_keys=sort_keys, cls=MyEncoder))
        fhandle.close()

def load_json(filename: Optional[str] = None, filepath: Optional[str] = None, verbose:  bool = False):
    if filename is not None:
        assert filepath is not None
        filepath = Path(filepath) / filename

    filepath = Path(filepath) if isinstance(filepath, str) else filepath
    if verbose: print('loading ', filepath, end='\r')
    with open(filepath, 'r') as fhandle:
        data_dict = json.load(fhandle)
        fhandle.close()
    # print(data_dict)
    return data_dict

def multiprocessor_speedup(func, *args, input_data: List[Any], num_workers: int = 4, keep_order: bool = True, **kwargs):
    
    with ThreadPoolExecutor(max_workers=min(num_workers, 16)) as pool:
        # Submit tasks to the thread pool
        futures = [pool.submit(func, x, *args, **kwargs) for x in input_data]
        results = []
        if keep_order:
            for future in futures:                
                try:
                    results.append(future.result()) # .result() blocks until this specific task completes
                except ValueError as ex:
                    print(f"ThreadPoolExecutor has failed: {ex}.")
                    pool.shutdown(wait=False, cancel_futures=True)   
        else:
            for future in as_completed(futures):                
                try:
                    results.append(future.result())
                except ValueError as ex:
                    print(f"ThreadPoolExecutor has failed: {ex}.")
                    pool.shutdown(wait=False, cancel_futures=True)  

    return results