#!/usr/bin/env python3
"""
Vehicle Safety Alert System - Entry Point

This is the main entry point for the Driver Assistant system.
Run this file directly or use: python -m src.main

Usage:
    python driver_assistant.py --help
    python driver_assistant.py --source webcam --display
    python driver_assistant.py --source video --video-path videos/test.mp4 --display
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.main import main

if __name__ == "__main__":
    sys.exit(main())
