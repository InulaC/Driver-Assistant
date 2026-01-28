"""
CSI Camera adapter for Raspberry Pi.

Uses picamera2 library for CSI camera interface.
"""

import time
import logging
from typing import Optional

import numpy as np

from src.capture.adapter import CameraAdapter, CaptureConfig
from src.capture.frame import Frame, FrameSource
from src.utils.platform import is_raspberry_pi

logger = logging.getLogger(__name__)


class CSICameraAdapter(CameraAdapter):
    """
    Camera adapter for Raspberry Pi CSI camera using picamera2.
    
    Only available on Raspberry Pi platforms.
    """
    
    def __init__(self, config: CaptureConfig):
        """
        Initialize the CSI camera adapter.
        
        Args:
            config: Capture configuration
        """
        super().__init__(config)
        self._camera = None
    
    def initialize(self) -> bool:
        """
        Initialize the CSI camera using picamera2.
        
        Returns:
            True if initialization successful
        """
        if not is_raspberry_pi():
            logger.error("CSI camera only available on Raspberry Pi")
            return False
        
        try:
            from picamera2 import Picamera2
            
            self._camera = Picamera2()
            
            width, height = self._config.resolution
            
            # Configure for video capture
            config = self._camera.create_video_configuration(
                main={"size": (width, height), "format": "BGR888"},
                buffer_count=2,
            )
            self._camera.configure(config)
            
            # Start the camera
            self._camera.start()
            
            # Allow camera to warm up
            time.sleep(0.5)
            
            logger.info(f"CSI camera initialized: {width}x{height}")
            
            self._is_initialized = True
            self.reset_frame_count()
            return True
            
        except ImportError:
            logger.error("picamera2 library not installed")
            return False
        except Exception as e:
            logger.error(f"CSI camera initialization error: {e}")
            return False
    
    def capture(self) -> Optional[Frame]:
        """
        Capture a frame from the CSI camera.
        
        Returns:
            Frame object if successful, None if capture failed
        """
        if not self._is_initialized or self._camera is None:
            logger.warning("CSI camera not initialized")
            return None
        
        try:
            # Capture frame as numpy array
            frame_data = self._camera.capture_array()
            timestamp = time.monotonic()
            
            if frame_data is None:
                logger.warning("CSI frame capture failed")
                return None
            
            # picamera2 returns BGR888 as configured
            # Ensure uint8 dtype
            if frame_data.dtype != np.uint8:
                frame_data = frame_data.astype(np.uint8)
            
            # Verify shape
            if len(frame_data.shape) != 3 or frame_data.shape[2] != 3:
                logger.error(f"Unexpected frame shape: {frame_data.shape}")
                return None
            
            frame = Frame(
                data=frame_data,
                timestamp=timestamp,
                sequence=self._increment_frame_count(),
                source=FrameSource.CSI,
            )
            
            return frame
            
        except Exception as e:
            logger.error(f"CSI capture error: {e}")
            return None
    
    def release(self) -> None:
        """Release the CSI camera."""
        if self._camera is not None:
            try:
                self._camera.stop()
                self._camera.close()
            except Exception as e:
                logger.warning(f"Error releasing CSI camera: {e}")
            self._camera = None
        
        self._is_initialized = False
        logger.info("CSI camera released")
    
    def is_healthy(self) -> bool:
        """
        Check if the CSI camera is functioning properly.
        
        Returns:
            True if camera is healthy
        """
        if not self._is_initialized or self._camera is None:
            return False
        
        try:
            # Try to check camera status
            return self._camera.is_open
        except Exception:
            return False
