# Bride and Groom Crowd Shots Video Processor

This project processes video files from multiple input folders to identify and classify ceremony videos using a machine learning model.

## Project Structure

```
bride_and_groom_crowd_shots/
├── main.py                 # Main entry point
├── parameters.json         # Configuration file
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── models/                # ML model files
└── utils/
    ├── __init__.py
    ├── ceremony_detector.py  # Video classification logic
    └── video_validator.py     # Video validation and filtering
```

## Features

- **Video Validation**: Filters videos by duration (minimum 10 minutes)
- **Recursive Directory Scanning**: Searches through nested subdirectories
- **Ceremony Detection**: Uses ML model to classify videos as ceremony, dance, or other
- **Batch Processing**: Processes multiple input folders sequentially
- **Comprehensive Logging**: Detailed progress and results reporting

## Configuration

Edit `parameters.json` to configure:
- Input folders to scan
- Output path for processed videos
- Export settings (resolution, frame rate, duration)

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place your ML model weights in `models/weights.h5`

3. Run the processor:
   ```bash
   python main.py
   ```

## Video Processing Pipeline

1. **Parameter Loading**: Parses configuration from `parameters.json`
2. **Directory Scanning**: Recursively scans each input folder for video files
3. **Video Validation**: Filters videos by duration (≥10 minutes)
4. **Ceremony Detection**: Analyzes each valid video using the ML model
5. **Results Collection**: Returns list of videos classified as ceremony

## Supported Video Formats

- MP4, AVI, MOV, MKV, WMV, FLV, WebM, M4V

## Requirements

- Python 3.7+
- TensorFlow 2.0+
- MoviePy
- Pillow
- NumPy
