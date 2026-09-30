# Project Structure & File Organization

## Complete Lunar Image Registration System
### ISRO SIH 2026 - Problem ID: 26166

---

## Directory Structure

```
lunar-image-registration/
│
├── Core Pipeline
│   ├── pipeline.py              # Main orchestration & workflow
│   ├── preprocessing.py         # Image preprocessing & normalization
│   ├── feature_extraction.py    # Feature detection & matching
│   ├── registration.py          # Geometric transformation & alignment
│   └── validation.py            # Quality metrics & assessment
│
├── Configuration & Setup
│   ├── requirements.txt         # Python dependencies
│   ├── config.json             # Configuration parameters
│   └── setup.py                # Installation script (optional)
│
├── Examples & Testing
│   ├── demo.py                 # Comprehensive demo with synthetic data
│   ├── example_usage.py        # 10 complete usage examples
│   └── test_images/            # Generated test results (created during runs)
│
├── Documentation
│   ├── README.md               # Complete project documentation
│   ├── QUICKSTART.md          # 5-minute quick start guide
│   ├── PROJECT_STRUCTURE.md   # This file
│   └── INSTALLATION.md         # Detailed installation instructions
│
└── Output (Generated)
    ├── registered_images/      # Output aligned images
    ├── results/                # Detailed results & metrics
    └── visualizations/         # Result visualizations
```

---

## File Descriptions

### Core Pipeline Files

#### `pipeline.py` (Main Entry Point)
- **Lines**: ~600
- **Purpose**: Orchestrates complete registration workflow
- **Key Classes**:
  - `LunarImageRegistrationPipeline`: Main pipeline class
  - Methods for single pair, series, and batch registration

**Key Functions**:
```python
register_pair(source, reference, output)  # Register two images
register_series(images_dict, ref_key)      # Register multiple to reference
batch_register(src_dir, ref_dir)           # Batch process directories
visualize_results(result)                  # Visualize and save
save_results(result, output_dir)           # Save comprehensive results
```

#### `preprocessing.py` (Image Preparation)
- **Lines**: ~350
- **Purpose**: Handle image preprocessing, normalization, enhancement
- **Key Classes**:
  - `LunarImagePreprocessor`: Main preprocessing class
  - `ImageNormalizer`: Multi-modal normalization

**Key Functions**:
```python
normalize_intensity(image)           # Normalize to 0-255
enhance_contrast(image)              # CLAHE enhancement
geometric_correction(image)          # Handle viewpoint variations
remove_illumination_bias(image)      # Handle sun angle effects
create_image_pyramid(image, levels)  # Multi-scale representation
preprocess_pipeline(image)           # Complete preprocessing
```

#### `feature_extraction.py` (Landmark Detection)
- **Lines**: ~450
- **Purpose**: Extract and match image features
- **Key Classes**:
  - `FeatureExtractor`: SIFT/ORB/AKAZE detector
  - `FeatureMatcher`: Brute force and FLANN matching
  - `LunarLandmarkDetector`: Crater and landmark detection

**Key Functions**:
```python
detect_and_compute(image)           # SIFT/ORB keypoints & descriptors
detect_craters(image)               # Hough circle crater detection
extract_multiscale_features(image)  # Features at multiple scales
match_features(desc1, desc2)        # Feature descriptor matching
filter_matches_ransac()             # RANSAC outlier filtering
```

#### `registration.py` (Geometric Alignment)
- **Lines**: ~400
- **Purpose**: Compute and apply geometric transformations
- **Key Classes**:
  - `ImageRegistration`: Transformation computation
  - `MultiModalRegistration`: Multi-sensor alignment

**Key Functions**:
```python
compute_transformation(src_pts, dst_pts)  # Compute transform matrix
apply_transformation(image, T)            # Warp image
refine_alignment(src, dst, T)             # Intensity-based refinement
extract_transformation_params(T)          # Get rotation/scale/translation
```

#### `validation.py` (Quality Assessment)
- **Lines**: ~350
- **Purpose**: Compute quality metrics and validation
- **Key Classes**:
  - `RegistrationMetrics`: Metric computation
  - `RegistrationValidator`: Validation and scoring
  - `RegistrationReport`: Report generation

**Key Functions**:
```python
compute_all_metrics(reference, registered)     # 6 quality metrics
compute_registration_quality_score(metrics)    # Overall score (0-100)
validate_point_accuracy(kp_src, kp_dst, T)    # Keypoint accuracy
visualize_registration_result()                # Create visualization
```

### Configuration Files

#### `config.json`
- **Size**: ~150 lines
- **Purpose**: All configurable parameters
- **Sections**:
  - `preprocessing`: Contrast, illumination handling
  - `feature_extraction`: SIFT/ORB parameters, crater detection
  - `feature_matching`: Matcher type, ratio test, RANSAC
  - `registration`: Transformation type, refinement settings
  - `validation`: Metrics selection and weighting
  - `output`: File saving options

**Example**:
```json
{
  "feature_extraction": {
    "method": "sift",
    "crater_detection": { "enabled": true }
  },
  "registration": {
    "refine_alignment": true,
    "max_iterations": 100
  }
}
```

#### `requirements.txt`
- **Purpose**: Python package dependencies
- **Key Packages**:
  - `opencv-python`: Image processing
  - `numpy`: Numerical computing
  - `scipy`: Scientific algorithms
  - `scikit-image`: Advanced imaging
  - `matplotlib`: Visualization

### Testing & Examples

#### `demo.py`
- **Lines**: ~400
- **Purpose**: Complete working demonstrations
- **Includes**:
  1. Single pair registration
  2. Multi-modal (OHRC/TMC/IIRS) registration
  3. Batch processing
  4. Configuration file usage
  5. Synthetic lunar image generation

**Run with**: `python demo.py`

#### `example_usage.py`
- **Lines**: ~350
- **Purpose**: 10 practical usage examples
- **Examples**:
  1. Basic registration
  2. With visualization
  3. Custom configuration
  4. Batch processing
  5. Examining results
  6. Preprocessing only
  7. Feature extraction only
  8. Configuration loading
  9. Saving detailed results
  10. Comparing configurations

### Documentation Files

#### `README.md`
- **Length**: ~1500 lines
- **Sections**:
  - Project overview and architecture
  - Complete installation guide
  - API documentation for each module
  - Configuration parameter guide
  - Example workflows
  - Performance optimization
  - Troubleshooting guide
  - ISRO SIH 2026 details
  - Future enhancements
  - References

#### `QUICKSTART.md`
- **Length**: ~400 lines
- **Quick Reference**:
  - 60-second setup
  - 5-minute first run
  - Common tasks with code
  - Configuration presets
  - Results interpretation
  - Troubleshooting table

#### `PROJECT_STRUCTURE.md` (This File)
- **Length**: ~300 lines
- **Contents**:
  - Complete file listing
  - Individual file descriptions
  - Class and function listings
  - Usage examples per module
  - Lines of code metrics

---

## Statistics

### Code Statistics
| File | Lines | Purpose |
|------|-------|---------|
| pipeline.py | ~600 | Main orchestration |
| preprocessing.py | ~350 | Image preparation |
| feature_extraction.py | ~450 | Landmark detection |
| registration.py | ~400 | Alignment & transformation |
| validation.py | ~350 | Quality assessment |
| demo.py | ~400 | Complete demonstrations |
| example_usage.py | ~350 | Usage examples |
| **Total** | **~3000** | **Production-ready code** |

### Documentation
| File | Content |
|------|---------|
| README.md | 1500+ lines comprehensive guide |
| QUICKSTART.md | 400 lines quick reference |
| config.json | 150 lines configuration |
| PROJECT_STRUCTURE.md | 300 lines (this file) |

### Total Project Size
- **Python Code**: ~3000 lines
- **Configuration**: ~150 lines
- **Documentation**: ~2200 lines
- **Total**: ~5350 lines

---

## Quick Module Reference

### Module Hierarchy

```
┌─────────────────────────────────────┐
│   pipeline.py (Main Orchestrator)   │
├─────────────────────────────────────┤
│  ├─ preprocessing.py                │
│  │  ├─ LunarImagePreprocessor      │
│  │  └─ ImageNormalizer             │
│  │                                   │
│  ├─ feature_extraction.py           │
│  │  ├─ FeatureExtractor            │
│  │  ├─ FeatureMatcher              │
│  │  └─ LunarLandmarkDetector       │
│  │                                   │
│  ├─ registration.py                 │
│  │  ├─ ImageRegistration           │
│  │  └─ MultiModalRegistration      │
│  │                                   │
│  └─ validation.py                   │
│     ├─ RegistrationMetrics         │
│     ├─ RegistrationValidator       │
│     └─ RegistrationReport          │
└─────────────────────────────────────┘
```

---

## Input/Output Formats

### Inputs
- **Image Formats**: PNG, JPG, TIFF (8-bit or 16-bit grayscale)
- **Metadata**: JSON configuration files
- **Datasets**: Directories of organized image pairs

### Outputs
```
output_directory/
├── registered.png                # Main aligned image
├── transformation.npy            # 3x3 transformation matrix
├── metrics.json                  # Quality scores
├── metadata.json                 # Registration parameters
└── visualization.png             # Result visualization
```

---

## Getting Started Checklist

- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Verify installation: `python -c "import cv2; import numpy"`
- [ ] Run demo: `python demo.py`
- [ ] Read QUICKSTART.md
- [ ] Try example_usage.py
- [ ] Register your images
- [ ] Save and visualize results

---

## Development Guide

### Adding New Feature Detector
Edit `feature_extraction.py`:
```python
def add_detector(self, detector_name):
    if detector_name == 'akaze':
        self.detector = cv2.AKAZE_create()
    # ...
```

### Adding New Transformation Type
Edit `registration.py`:
```python
def compute_transformation(self, src_pts, dst_pts, method='custom'):
    if method == 'custom':
        # Implement custom logic
        T = custom_transform_logic(src_pts, dst_pts)
    # ...
```

### Adding New Metric
Edit `validation.py`:
```python
@staticmethod
def new_metric(img1, img2):
    # Implement metric
    return metric_value
```

---

## Performance Characteristics

### Time Complexity
- **Preprocessing**: O(n) where n = image pixels
- **Feature Extraction**: O(n log n) for SIFT
- **Feature Matching**: O(m²) where m = keypoints
- **Registration**: O(k) where k = iteration count
- **Total**: O(n log n) dominated by feature extraction

### Space Complexity
- **Image Storage**: O(n)
- **Keypoints**: O(m) where m << n
- **Descriptors**: O(m × d) where d = descriptor dimension
- **Total**: O(n + m × d)

---

## Deployment Options

### Option 1: Standalone Script
```bash
python pipeline.py
```

### Option 2: Jupyter Notebook
```python
from pipeline import LunarImageRegistrationPipeline
pipeline = LunarImageRegistrationPipeline()
# ... use in notebook
```

### Option 3: Web API (Future)
```python
from flask import Flask
app = Flask(__name__)
@app.route('/register', methods=['POST'])
def register():
    # API endpoint
    pass
```

### Option 4: Batch Processing
```bash
python pipeline.py --batch --source-dir ./input --output-dir ./output
```

---

## Maintenance Notes

### Regular Updates
- Keep dependencies updated: `pip install --upgrade -r requirements.txt`
- Monitor for OpenCV/NumPy security updates
- Test with new Chandrayaan-2 data releases

### Performance Tuning
- Profile with cProfile: `python -m cProfile pipeline.py`
- Identify bottlenecks in feature extraction
- Consider GPU acceleration if needed

### Version Control
- Current Version: 1.0.0
- Last Updated: September 2026
- Status: Production Ready

---

## References & Links

- **ISRO Official**: https://www.isro.gov.in/
- **Chandrayaan-2**: https://www.isro.gov.in/chandrayaan-2/
- **CHMAP Database**: https://chmapbrowse.issdc.gov.in/
- **LROC Images**: https://lroc.im-ldi.com/
- **OpenCV Docs**: https://docs.opencv.org/

---

## Support & Contact

For issues, enhancements, or questions:
1. Review README.md for detailed documentation
2. Check QUICKSTART.md for common tasks
3. Examine example_usage.py for code samples
4. Inspect module docstrings for API details

---

**Last Updated**: September 2026
**Project Status**: ✅ Complete & Production Ready
