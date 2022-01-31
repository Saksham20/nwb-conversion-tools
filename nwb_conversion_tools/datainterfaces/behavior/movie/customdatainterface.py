from .movie_utils import VideoCaptureContext, VideoCaptureContextSimple
from pynwb.image import ImageSeries
import cv2
from pynwb import NWBHDF5IO
import tempfile
import os


def write_nwb(nwbfile, suffix):
    nwbfile_loc = os.path.join(tempfile.mkdtemp(), f"custominterface_{suffix}.nwb")
    with NWBHDF5IO(nwbfile_loc, "w") as io:
        io.write(nwbfile)
    print(f"nwbfile written at {nwbfile_loc}")

def custom_data_interface(nwbfile, movie_files, context_manager, write=True):
    """
    This is a method that takes a nwbfile and writes ImageSeries with external flag on to
    nwbfile's acquisition.
    """
    if context_manager == "normal":
        with VideoCaptureContext(str(movie_files[0])) as vc:
            fps = vc.get_movie_fps()
        del vc
    elif context_manager == "simple":
        vc = VideoCaptureContextSimple(str(movie_files[0]))
        fps = vc.get(cv2.CAP_PROP_FPS)
        vc.release()
        del vc
    elif context_manager == "cv2":
        vc = cv2.VideoCapture(str(movie_files[0]))
        fps = vc.get(cv2.CAP_PROP_FPS)
        vc.release()
        del vc
    else:
        fps = 25.0
    nwbfile.add_acquisition(
        ImageSeries(name="name", description="desc", unit="n.a.",
                    starting_time=0.0, rate=fps, format="external",
                    external_file=movie_files))
    if write:
        write_nwb(nwbfile, context_manager)

