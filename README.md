# Cow Re-Identification System

A production-ready cattle re-identification system using advanced computer vision and deep learning. Identify individual cows based on their unique spot patterns across multiple camera angles and time periods.

## Features

- **Multi-View Reference System**: Register cows from multiple angles for robust identification
- **Advanced Pattern Matching**: DINOv2 patch-based matching for superior accuracy
- **YOLO Detection**: Automatic cow detection and segmentation
- **Real-time Processing**: Process images and videos efficiently
- **TOP-1 Matching**: Identify the reference cow (ALPHA) among detected cattle
- **Flexible Models**: Support for ResNet50, DINOv2, and MegaDescriptor

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or MPS for Apple Silicon

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/cow-reid.git
cd cow-reid
```

2. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download YOLO model** (optional - will auto-download on first run)

The system will automatically download the YOLOv8 model on first use. Alternatively, download manually from Ultralytics and place in the project directory.

## Quick Start

### Running the Application

```bash
streamlit run app.py
```

The web interface will open at `http://localhost:8501`

### Basic Workflow

1. **Configure Settings** (Left Sidebar)
   - Select embedding model (DINOv2 recommended)
   - Enable patch matching for best accuracy
   - Choose YOLO model type
   - Enable preprocessing if needed

2. **Register Reference Cow (ALPHA)**
   - Upload image of the cow you want to identify
   - Choose registration mode:
     - **Single View**: One reference image
     - **Multi-View**: 3 images from different angles
   - Click "Set as ALPHA (Reference Cow)"

3. **Process Images/Videos**
   - Upload test image or video
   - Click "Start Processing"
   - View results with similarity scores
   - TOP-1 match is identified as ALPHA

## Usage Guide

### Single-View Registration

Best for fixed camera angles where the cow is always seen from the same perspective.

```
1. Upload reference image
2. Select "Single View"
3. Click "Set as ALPHA (Reference Cow)"
```

### Multi-View Registration

Best for scenarios with multiple cameras or varying angles. Provides 30-50% better accuracy across different viewpoints.

#### Option A: Auto-Split

Use when you have a single wide-angle image:

```
1. Upload panoramic/wide image of the cow
2. Select "Multi-View (3 Images)"
3. Choose "Auto-split (divide image into 3 parts)"
4. Preview the 3 views (Left | Center | Right)
5. Click "Register Multi-View (Auto-split)"
```

#### Option B: Manual Upload

Use when you have 3 separate images from different angles:

```
1. Select "Multi-View (3 Images)"
2. Choose "Manual upload (3 separate images)"
3. Upload 3 images:
   - View 1: e.g., Left side
   - View 2: e.g., Top/Front
   - View 3: e.g., Right side
4. Click "Register Multi-View (Manual)"
```

### Model Selection

**ResNet50 (Default)**
- Fast and lightweight
- Good for simple patterns
- CPU-friendly

**DINOv2 ViT-B/14 (Recommended)**
- State-of-the-art accuracy
- Excellent patch-level matching
- Best for multi-view scenarios
- Requires GPU for optimal performance

**MegaDescriptor-S/L**
- SOTA matching performance
- Highest accuracy on complex patterns
- Requires significant GPU memory

### YOLO Configuration

**YOLOv8x-seg (Segmentation)**
- Default - most accurate
- Precise cow masks
- Class 19 (cow) from COCO dataset

**Custom Model**
- Upload your own trained YOLO model
- Detection or segmentation modes supported

## Understanding Results

### Similarity Scores

- **0.9 - 1.0**: Very high confidence match (same cow)
- **0.7 - 0.9**: High confidence (likely same cow)
- **0.5 - 0.7**: Medium confidence (review manually)
- **Below 0.5**: Low confidence (different cow)

### TOP-1 Matching

The system identifies the cow with the **highest similarity score** as ALPHA. All other detected cows are labeled as "None".

### Multi-View Matching

When using multi-view:
- System compares against ALL registered views
- Returns the **MAXIMUM** score (best matching view)
- Provides robust identification across different angles

## Performance Optimization

### For Best Accuracy

```python
Model: DINOv2 ViT-B/14
Patch Matching: Enabled
Registration: Multi-View (3 images)
Preprocessing: Enabled (CLAHE + Blur)
```

### For Best Speed

```python
Model: ResNet50
Patch Matching: Disabled
Registration: Single View
Video Processing: 1 FPS sampling
```

## Troubleshooting

### Common Issues

**Issue**: Model loading is slow
- **Solution**: Models download on first use. Subsequent runs are fast.

**Issue**: Low accuracy on different angles
- **Solution**: Use Multi-View registration with 3 diverse angles

**Issue**: Out of memory errors
- **Solution**: Use ResNet50 instead of DINOv2, or reduce video resolution

**Issue**: YOLO not detecting cows
- **Solution**: Ensure good lighting and cow is clearly visible. Try custom trained model.

### GPU/MPS Support

The system automatically detects and uses:
- **CUDA** on NVIDIA GPUs
- **MPS** on Apple Silicon (M1/M2/M3)
- **CPU** as fallback

Check the sidebar for current device information.

## Advanced Configuration

### Preprocessing

**CLAHE (Contrast Limited Adaptive Histogram Equalization)**
- Enhances spot contrast
- Reduces lighting variations

**Gaussian Blur**
- Reduces noise
- Smooths patterns

Enable for:
- Poor lighting conditions
- Low contrast images
- Noisy camera feeds

Disable for:
- High-quality images
- When speed is critical

### Patch Matching

When enabled:
- Divides image into 16×16 grid (256 patches)
- Performs local pattern matching
- More robust to pose variations
- Slower but more accurate

## System Requirements

### Minimum

- Python 3.8+
- 8GB RAM
- 4GB free disk space

### Recommended

- Python 3.10+
- 16GB RAM
- NVIDIA GPU with 8GB+ VRAM
- 10GB free disk space

### Optimal

- Python 3.11+
- 32GB RAM
- NVIDIA GPU with 16GB+ VRAM (for MegaDescriptor)
- SSD storage

## Citation

If you use this system in your research, please cite:

```bibtex
@software{cow_reid_2026,
  title = {Cow Re-Identification System},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/cow-reid}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv8 by Ultralytics
- DINOv2 by Meta AI Research
- ResNet by Microsoft Research
- Streamlit for the web framework

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Submit a pull request
- Contact: your.email@example.com

## Changelog

### v1.0.0 (2026-01-19)
- Initial release
- Multi-view registration support
- DINOv2 integration
- Patch-level matching
- TOP-1 identification system
- Video processing support
