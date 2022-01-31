import os
import tempfile
import unittest
from datetime import datetime

import cv2
import numpy as np
from dateutil.tz import tzlocal
from pynwb import NWBFile
from pynwb import NWBHDF5IO
from pynwb.image import ImageSeries

from nwb_conversion_tools import NWBConverter, MovieInterface
from nwb_conversion_tools.datainterfaces.behavior.movie.customdatainterface import \
    custom_data_interface
from nwb_conversion_tools.datainterfaces.behavior.movie.movie_utils import \
    VideoCaptureContext, VideoCaptureContextSimple


class TestVideoContext(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp()
        self.movie_files = self.create_movies()
        self.nwbfile_path = os.path.join(self.test_dir, "movie_test.nwb")
        self.nwbfile = self.create_nwb()

    def create_movies(self):
        movie_file1 = os.path.join(self.test_dir, "test1.avi")
        movie_file2 = os.path.join(self.test_dir, "test2.avi")
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

    def create_nwb(self):
        return NWBFile("session description",
                       'EXAMPLE_ID',
                       datetime.now(tzlocal()))

    def create_image_series(self, ext_file, rate):
        return ImageSeries(name="name", description="desc", unit="n.a.",
                           starting_time=0.0, rate=rate, format="external",
                           external_file=ext_file)

    def nwb_file_round_trip(self):
        with NWBHDF5IO(self.nwbfile_path, "w") as io:
            io.write(self.nwbfile)
        with NWBHDF5IO(self.nwbfile_path, "r") as io:
            nwbfile = io.read()
        print(f"nwbfile written at {self.nwbfile_path}")

    def use_full_converter(self, context):
        class MovieTestNWBConverter(NWBConverter):
            data_interface_classes = dict(Movie=MovieInterface)

        source_data = dict(Movie=dict(file_paths=self.movie_files))
        converter = MovieTestNWBConverter(source_data)

        starting_times = [np.float(np.random.randint(200)) for i in range(len(self.movie_files))]
        conversion_opts = dict(Movie=dict(starting_times=starting_times,
                                          external_mode=True,
                                          context_manager=context))
        converter.run_conversion(
            save_to_file=False, overwrite=True, conversion_options=conversion_opts, nwbfile=self.nwbfile
        )
        self.nwb_file_round_trip()

    def use_data_interface(self, context):
        starting_times = [np.float(np.random.randint(200)) for i in range(len(self.movie_files))]
        args = dict(starting_times=starting_times,
                    external_mode=True,
                    context_manager=context)
        mi = MovieInterface(file_paths=self.movie_files)
        mi.run_conversion(self.nwbfile, mi.get_metadata(), **args)
    # using the full context manager.
    def test_normal_context_fullconverter(self):  # seg fault
        self.use_full_converter("normal")

    def test_normal_context_datainterface(self):  # seg fault
        self.use_data_interface("normal")
        self.nwb_file_round_trip()

    def test_normal_context_customdatainterface(self):  # seg fault
        custom_data_interface(self.nwbfile, self.movie_files, "normal", write=False)
        self.nwb_file_round_trip()

    def test_normal_context_nwb(self):
        with VideoCaptureContext(self.movie_files[0]) as vc:
            fps = vc.get_movie_fps()
        self.nwbfile.add_acquisition(self.create_image_series(self.movie_files, fps))
        self.nwb_file_round_trip()
    # using cv2 directly
    def test_cv2_fullconverter(self):
        self.use_full_converter("cv2")

    def test_cv2_datainterface(self):
        self.use_data_interface("cv2")
        self.nwb_file_round_trip()

    def test_cv2_customdatainterface(self):
        custom_data_interface(self.nwbfile, self.movie_files, "cv2")
        self.nwb_file_round_trip()

    def test_cv2_nwb(self):
        vc = cv2.VideoCapture(self.movie_files[0])
        fps = vc.get(cv2.CAP_PROP_FPS)
        vc.release()
        self.nwbfile.add_acquisition(self.create_image_series(self.movie_files, fps))
        self.nwb_file_round_trip()
    # using simple cv2.VideoWriter inherited class
    def test_simple_context_fullconverter(self):  # seg fault
        self.use_full_converter("simple")

    def test_simple_context_datainterface(self):  # seg fault
        self.use_data_interface("simple")
        self.nwb_file_round_trip()

    def test_simple_context_customdatainterface(self):  # seg fault
        custom_data_interface(self.nwbfile, self.movie_files, "simple", write=False)
        self.nwb_file_round_trip()

    def test_simple_context_nwb(self):
        vc = VideoCaptureContextSimple(self.movie_files[0])
        fps = vc.get(cv2.CAP_PROP_FPS)
        vc.release()
        self.nwbfile.add_acquisition(self.create_image_series(self.movie_files, fps))
        self.nwb_file_round_trip()
    # testing hard coded values:
    def test_hard_code_fps_fullconverter(self):
        self.use_full_converter("")

    def test_hard_code_fps_datainterface(self):
        self.use_data_interface("")
        self.nwb_file_round_trip()

    def test_hard_code_fps_converter(self):
        self.nwbfile.add_acquisition(self.create_image_series(self.movie_files, 25.0))
        self.nwb_file_round_trip()
