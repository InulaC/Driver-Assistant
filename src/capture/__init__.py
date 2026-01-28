"""Frame capture module for the Vehicle Safety Alert System."""

from src.capture.adapter import CameraAdapter, FrameSource
from src.capture.frame import Frame
from src.capture.opencv_camera import OpenCVCameraAdapter
from src.capture.video_file import VideoFileAdapter
from src.capture.factory import create_camera_adapter

__all__ = [
    "CameraAdapter",
    "FrameSource",
    "Frame",
    "OpenCVCameraAdapter",
    "VideoFileAdapter",
    "create_camera_adapter",
]
