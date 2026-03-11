# Crowd Behaviour Analysis

A comprehensive computer vision system for analyzing crowd behavior using pose estimation, activity recognition, and crowd group detection. This project identifies individual activities (sitting, standing, walking, running) and detects crowd formations in real-time.

## Features

- **Activity Detection**: Recognizes human poses and classifies activities:
  - Sitting
  - Standing
  - Walking
  - Running

- **Multi-Object Tracking**: Uses DeepSort algorithm to track individuals across frames

- **Crowd Group Detection**: Identifies and analyzes group formations within crowds

- **Real-time Heatmap**: Visualizes crowd density and movement patterns

- **Multiple Input Sources**:
  - Laptop/built-in webcam
  - Iriun virtual camera
  - Video files
  - Uploaded video files

- **Interactive Web UI**: Built with Streamlit for easy interaction

## Project Structure

```
├── camera.py          # Camera and video input handling
├── model.py           # Activity and crowd detection models
├── ui.py             # Streamlit web interface
├── requirements.txt   # Python dependencies
└── README.md         # Project documentation
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/SuprakashMaji/Crowd-Behaviour-Analysis.git
cd Crowd-Behaviour-Analysis
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

Start the Streamlit web interface:
```bash
streamlit run ui.py
```

The application will open in your default browser at `http://localhost:8501`

### Using Different Video Sources

1. **Laptop Webcam**: Select "Laptop Camera" from the sidebar
2. **Iriun Camera**: Select "Iriun Camera" from the sidebar
3. **Video File**: Upload or select a video file
4. **Live Stream**: Use the webcam option for real-time analysis

## Dependencies

- **numpy** (1.26.4): Numerical computing library
- **opencv-python** (4.10.0.84): Computer vision library
- **streamlit** (1.31.1): Web app framework
- **ultralytics** (8.1.0): YOLO implementation
- **deep-sort-realtime** (1.3.2): DeepSort tracking algorithm

## How It Works

### Activity Detection
1. Captures video frames from the selected source
2. Uses YOLO pose estimation to detect human keypoints
3. Calculates angles between joints (hip, knee, ankle)
4. Classifies activity based on pose metrics:
   - **Standing**: Knee angle above threshold
   - **Sitting**: Low vertical position
   - **Walking/Running**: Movement speed and motion patterns

### Crowd Group Detection
1. Detects all individuals in the frame
2. Analyzes spatial proximity and movement patterns
3. Groups nearby individuals into crowd clusters
4. Generates heatmap showing crowd density

### Multi-Object Tracking
- Maintains consistent IDs across frames
- Tracks individuals even during brief occlusions
- Provides motion history for each person

## Configuration

Adjust model parameters in `model.py`:
- `TH_STAND`: Standing threshold (default: 0.35)
- `TH_RUN`: Running threshold (default: 1.6)
- `fps`: Frame rate adjustment
- `model_name`: YOLO model version

## Performance Tips

- Use GPU acceleration if available (NVIDIA CUDA)
- For webcam: Lower resolution for faster processing
- For video files: Pre-compress to reduce file size
- Use lighter YOLO models (yolov8n-pose.pt) for faster inference

## Supported Video Formats

- MP4, AVI, MOV, FLV, MKV
- Webcam streams
- Virtual cameras (Iriun, OBS Virtual Camera)

## Key Components

### camera.py
- `open_laptop_camera()`: Access built-in webcam
- `open_iriun_camera()`: Access Iriun virtual camera
- `open_video_file()`: Open video files
- `save_uploaded_to_temp()`: Handle Streamlit file uploads

### model.py
- `ActivityModel`: Pose-based activity recognition
- `CrowdGroupModel`: Crowd detection and grouping
- Keypoint-based angle calculations
- Real-time frame processing

### ui.py
- Streamlit interface
- Multi-column layout
- Real-time visualization
- Session state management

## Limitations

- Performance depends on lighting conditions
- Occluded people may be tracked incorrectly
- Very dense crowds may reduce accuracy
- Requires adequate GPU memory for smooth performance

## Future Enhancements

- [ ] Anomaly detection in crowd behavior
- [ ] Emotion recognition from pose
- [ ] Fall detection
- [ ] Crowd evacuation prediction
- [ ] Database logging of events
- [ ] REST API for integration

## License

This project is open source and available for educational and research purposes.

## Authors

- Suprakash Maji
- B.Tech Project Contributors

## Contact & Support

For issues, questions, or contributions, please visit the GitHub repository:
https://github.com/SuprakashMaji/Crowd-Behaviour-Analysis

## Acknowledgments

- YOLOv8 by Ultralytics
- DeepSort by nwojke
- Streamlit for the web framework
