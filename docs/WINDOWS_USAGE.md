# Windows Usage Guide

## Quick Start

### 1. Activate Virtual Environment
```powershell
cd C:\Users\cinul\Desktop\Driver-Assistant
.\venv\Scripts\activate
```

### 2. Run with Video File
```powershell
python -m src.main --source video --video-path videos/laneTest2.mp4 --display
```

### 3. Run with Webcam
```powershell
python -m src.main --source webcam --display
```

---

## Sample Commands

```powershell
# Run with a test video
python -m src.main --source video --video-path videos/laneTest2.mp4 --display

# Run with webcam (camera index 0)
python -m src.main --source webcam --display

# Run headless (no display window)
python -m src.main --source video --video-path videos/test.mp4 --headless

# Run with custom config
python -m src.main --source webcam --display --config my_config.yaml
```

---

## Command Options

| Option | Description |
|--------|-------------|
| `--source` | `video`, `webcam`, or `csi` (Pi only) |
| `--video-path` | Path to video file (required for video source) |
| `--display` | Show visual output window |
| `--headless` | Run without display |
| `--config` | Custom config file (default: config.yaml) |

---

## Run Tests
```powershell
.\venv\Scripts\python.exe -m pytest tests/ -v
```

---

## Controls
- Press **Q** or **ESC** to quit when display is active

---

## Notes
- GPIO/LiDAR features use stubs on Windows (no hardware)
- Audio alerts work via Windows beep
- Full hardware features require Raspberry Pi
