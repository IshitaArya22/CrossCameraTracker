# Cross-Camera Player Mapping System

## Overview

This is a computer vision application for tracking and mapping soccer players across multiple camera views (broadcast and tactical cameras). The system uses deep learning models for player detection, multi-object tracking, feature extraction, and cross-camera player association to maintain consistent player identities across different viewpoints.

The application is built using Streamlit for the web interface and combines several computer vision techniques including YOLO object detection, Kalman filter-based tracking, and feature-based player re-identification.

## System Architecture

### Frontend Architecture
- **Streamlit Web Interface**: Single-page application with file upload capabilities
- **Real-time Configuration**: Sidebar controls for adjusting detection parameters
- **Dual Video Processing**: Side-by-side upload interface for broadcast and tactical camera feeds

### Backend Architecture
- **Modular Design**: Separate modules for detection, tracking, feature extraction, and player mapping
- **Pipeline Architecture**: Sequential processing through detection → tracking → feature extraction → mapping
- **OpenCV-based Video Processing**: Frame-by-frame analysis with efficient memory management

### Core Processing Pipeline
1. **Video Input**: Dual camera feed processing
2. **Player Detection**: YOLO-based person detection
3. **Multi-Object Tracking**: Kalman filter-based tracking with Hungarian algorithm
4. **Feature Extraction**: Multi-modal feature vectors (color, texture, shape)
5. **Cross-Camera Mapping**: Similarity-based player association

## Key Components

### 1. Player Detection (`detection.py`)
- **Technology**: YOLOv8 pre-trained model
- **Purpose**: Detect players in video frames using bounding boxes
- **Key Features**:
  - Configurable confidence thresholds
  - Person class filtering (COCO dataset class 0)
  - Bounding box validation

### 2. Player Tracking (`tracking.py`)
- **Technology**: Deep SORT-inspired multi-object tracking
- **Purpose**: Maintain consistent player IDs across frames
- **Key Features**:
  - Kalman filter for motion prediction
  - Hungarian algorithm for detection-to-track association
  - Track lifecycle management (creation, update, deletion)

### 3. Feature Extraction (`feature_extraction.py`)
- **Technology**: Multi-modal computer vision features
- **Purpose**: Extract distinctive visual features for player re-identification
- **Key Features**:
  - Color histograms (HSV, LAB color spaces)
  - Texture analysis
  - Shape descriptors
  - Feature normalization and concatenation

### 4. Player Mapping (`player_mapping.py`)
- **Technology**: Cosine similarity with Hungarian algorithm optimization
- **Purpose**: Associate players across different camera views
- **Key Features**:
  - Feature similarity computation
  - Optimal assignment solving
  - Configurable similarity thresholds

### 5. Video Processing (`utils.py`)
- **Technology**: OpenCV video handling utilities
- **Purpose**: Video I/O operations and frame extraction
- **Key Features**:
  - Video metadata extraction
  - Frame sampling and processing
  - Annotation and output generation

## Data Flow

1. **Input Stage**: User uploads broadcast and tactical camera videos through Streamlit interface
2. **Detection Stage**: YOLO processes each frame to detect player bounding boxes
3. **Tracking Stage**: Multi-object tracker assigns and maintains player IDs within each camera view
4. **Feature Extraction Stage**: Visual features are extracted from player crops for each detected player
5. **Mapping Stage**: Cross-camera association using feature similarity and optimal assignment
6. **Output Stage**: Annotated videos with consistent player mappings across views

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework
- **OpenCV**: Computer vision and video processing
- **Ultralytics**: YOLOv8 object detection model
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **SciPy**: Scientific computing (optimization algorithms)
- **scikit-learn**: Machine learning utilities (similarity metrics, normalization)

### Model Dependencies
- **YOLOv8**: Pre-trained COCO model for person detection
- **COCO Dataset**: Class definitions for object detection

## Deployment Strategy

### Development Environment
- **Platform**: Compatible Python environment
- **Dependencies**: Install via pip (requirements managed in code imports)
- **Video Storage**: Temporary file handling for uploaded videos

### Production Considerations
- **Memory Management**: Frame-by-frame processing to handle large video files
- **Performance Optimization**: Configurable frame sampling and processing parameters
- **Scalability**: Modular architecture allows for component replacement and optimization

### File Structure
```
├── app.py                 # Main Streamlit application
├── detection.py          # YOLO-based player detection
├── tracking.py           # Multi-object tracking implementation
├── feature_extraction.py # Visual feature extraction
├── player_mapping.py     # Cross-camera player association
├── utils.py              # Video processing utilities
└── attached_assets/      # Documentation and reference materials
```
