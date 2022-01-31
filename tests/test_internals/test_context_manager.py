import shutil
import unittest
import tempfile
import numpy as np
from pynwb import NWBHDF5IO
import os
from pynwb import NWBFile
from datetime import datetime
from dateutil.tz import tzlocal
from pynwb.image import ImageSeries
from nwb_conversion_tools.datainterfaces.behavior.movie.movie_utils import *
import cv2


def create_movies(test_dir):
    movie_file1 = os.path.join(test_dir, "test1.avi")
    movie_file2 = os.path.join(test_dir, "test2.avi")
    (nf, nx, ny) = (10, 640, 480)
    writer1 = cv2.VideoWriter(
        filename=movie_file1,
        apiPreference=None,
        fourcc=cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        fps=25,
        frameSize=(ny, nx),
        params=None,
    )
    writer2 = cv2.VideoWriter(
        filename=movie_file2,
        apiPreference=None,
        fourcc=cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        fps=25,
        frameSize=(ny, nx),
        params=None,
    )

    for k in range(nf):
        writer1.write(np.random.randint(0, 255, (nx, ny, 3)).astype("uint8"))
        writer2.write(np.random.randint(0, 255, (nx, ny, 3)).astype("uint8"))
    writer1.release()
    writer2.release()
    return [movie_file1, movie_file2]

def write_npy_data_here(data:str, suffix, filepath):
    npy_file = os.path.join(filepath, f"array_{suffix}_data.npy")
    with open(npy_file, "wb") as f:
        np.save(f, data)
    print(f"written npy file at {npy_file}")

def write_npy_frames0_here(video_context_ob, suffix, filepath):
    npy_file = os.path.join(filepath, f"array_{suffix}_frames0.npy")
    file = open(npy_file, "wb")
    for data in video_context_ob:
        np.save(file, data)
    file.close()
    print(f"written npy file at {npy_file}")

def write_npy_frames1_here(video_context_ob, suffix, filepath):
    npy_file = os.path.join(filepath, f"array_{suffix}_frames1.npy")
    frames = []
    for data in video_context_ob:
        frames.append(data)
    with open(npy_file, "wb") as f:
        np.save(f, np.array(frames))
    print(f"written npy file at {npy_file}")


class TestVideoContext(unittest.TestCase):
    """
    Unit tests to single out the segmentation fault source.
    Number of tests are a permutation of
    (creating simple nwb, using nwbconverter, using fileio),
    (videocapturecontext, using open cv directly, and simple videocapture context).
    """
    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp()
        self.movie_files = create_movies(self.test_dir)

    def test_read_here_write_data(self):
        with VideoCaptureContext(self.movie_files[0]) as vc:
            fps = vc.get_movie_fps()
        write_npy_data(fps, "r_here_w_there", self.test_dir)
        write_npy_data_here(fps, "r_here_w_here", self.test_dir)

    def test_read_here_write_frames0(self):
        with VideoCaptureContext(self.movie_files[0]) as vc:
            fps = vc.get_movie_fps()
        write_npy_frames0(vc, "r_here_w_there", self.test_dir)
        write_npy_frames0_here(vc, "r_here_w_here", self.test_dir)

    def test_read_here_write_frames1(self):
        with VideoCaptureContext(self.movie_files[0]) as vc:
            fps = vc.get_movie_fps()
        write_npy_frames1(vc, "r_here_w_there", self.test_dir)
        write_npy_frames1_here(vc, "r_here_w_here", self.test_dir)

    def test_read_there_write_data(self):
        fps = write_npy_data_video(self.movie_files[0])
        write_npy_data_here(fps, "r_there_w_there_data", os.path.dirname(self.movie_files[0]))

    def test_read_there_write_frames0(self):
        vc = write_npy_frames0_video(self.movie_files[0])
        write_npy_frames0_here(vc, "r_there_w_there_frames0", os.path.dirname(self.movie_files[0]))

    def test_read_there_write_frames1(self):
        vc = write_npy_frames1_video(self.movie_files[0])
        write_npy_frames1_here(vc, "r_there_w_there_frames1", os.path.dirname(self.movie_files[0]))