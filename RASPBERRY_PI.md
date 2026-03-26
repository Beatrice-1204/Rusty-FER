# Raspberry Pi 5 Setup

This branch can now read frames directly from the Raspberry Pi Camera Module 3 Wide by using the `Picamera2` backend instead of `cv2.VideoCapture(0)`.

## Required packages

On Raspberry Pi OS, install the runtime dependencies:

```bash
pip install -r requirements-pi.txt
```

If `picamera2` is not available in your virtual environment, install the Raspberry Pi camera stack first and verify the camera works outside Python:

```bash
libcamera-hello
```

## Run

Use these environment variables on the Pi:

```bash
RUSTY_CAMERA_BACKEND=picamera2
RUSTY_CAMERA_WIDTH=640
RUSTY_CAMERA_HEIGHT=480
python src/app.py
```

Single-line version:

```bash
RUSTY_CAMERA_BACKEND=picamera2 RUSTY_CAMERA_WIDTH=640 RUSTY_CAMERA_HEIGHT=480 python src/app.py
```

## Notes

- `RUSTY_CAMERA_BACKEND=auto` will also select `Picamera2` automatically on Linux when the package is installed.
- `640x480` is a practical starting point for Raspberry Pi 5 because it keeps the YuNet + FER+ ONNX pipeline responsive.
- If you want lower latency, test `512x384`.
- If you want more face detail, test `800x600`, but expect lower FPS.
