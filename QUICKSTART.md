# Quick Start Guide

## 60-Second Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify installation
python -c "import cv2, numpy, scipy; print('Ready!')"
```

## 5-Minute First Run

### Option A: Try the Demo
```bash
# Run complete demo with synthetic images
python demo.py
```

This will:
- Generate synthetic lunar images
- Demonstrate single pair registration
- Show multi-modal alignment
- Perform batch processing

Check `test_images/`, `test_images_multimodal/`, and `test_batch/` directories for results.

### Option B: Register Your Images
```python
from pipeline import LunarImageRegistrationPipeline

# Create pipeline
pipeline = LunarImageRegistrationPipeline()

# Register two images
result = pipeline.register_pair(
    'your_source.png',
    'your_reference.png',
    'output.png'
)

# See results
print(f"Quality Score: {result['quality_score']}")
```

---

## Common Tasks

### Task 1: Register Single Image Pair
```python
from pipeline import LunarImageRegistrationPipeline

pipeline = LunarImageRegistrationPipeline()
result = pipeline.register_pair('source.png', 'reference.png', 'output.png')
pipeline.visualize_results(result, 'visualization.png')
```

### Task 2: Register Multiple Images to Reference
```python
image_dict = {
    'ref': 'reference.png',
    'img1': 'image1.png',
    'img2': 'image2.png'
}

results = pipeline.register_series(image_dict, 'ref', output_dir='./results')
```

### Task 3: Batch Process Directory
```python
results = pipeline.batch_register(
    source_dir='./source_images',
    reference_dir='./reference_images',
    output_dir='./registered'
)
```

### Task 4: Get Detailed Metrics
```python
result = pipeline.register_pair('source.png', 'reference.png')

print(f"RMSE: {result['metrics']['rmse']:.2f}")
print(f"SSIM: {result['metrics']['ssim']:.4f}")
print(f"Correlation: {result['metrics']['correlation']:.4f}")
print(f"Quality Score: {result['quality_score']:.1f}/100")
```

### Task 5: Extract Transformation Parameters
```python
result = pipeline.register_pair('source.png', 'reference.png')

params = result['transformation_params']
print(f"Translation: {params['translation']}")
print(f"Rotation: {params['rotation']:.2f}°")
print(f"Scale: {params['scale']:.4f}")
```

---

## Configuration Presets

### Fast (For Speed)
```python
config = {
    'feature_method': 'orb',
    'enhance_preprocessing': False,
    'refine_alignment': False
}
```

### Balanced (Default)
```python
config = {
    'feature_method': 'sift',
    'enhance_preprocessing': True,
    'refine_alignment': True
}
```

### Accurate (For Best Results)
```python
config = {
    'feature_method': 'sift',
    'enhance_preprocessing': True,
    'remove_illumination_bias': True,
    'refine_alignment': True
}
```

---

## Understanding Results

### Quality Score Interpretation
- **90-100**: Excellent registration, high confidence
- **75-90**: Good registration, suitable for most applications
- **60-75**: Acceptable registration, verify visually
- **Below 60**: Poor registration, check inputs

### Key Metrics
- **Correlation**: How similar the images are (higher = better)
- **SSIM**: Structural similarity (higher = better)
- **RMSE**: Root mean square error (lower = better)
- **Matches**: Number of matched features (more = more reliable)

---

## Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| "Not enough keypoints" | Increase enhance_preprocessing or reduce contrast |
| "Not enough matches" | Verify images have overlap, check quality |
| "Slow processing" | Use ORB instead of SIFT, disable refinement |
| "Poor alignment" | Enable preprocessing, increase iterations |
| "Memory error" | Reduce image size or use batch processing |

---

## Output Files Explained

```
output_directory/
├── registered.png          # Aligned source image
├── transformation.npy      # Transformation matrix (3x3)
├── metrics.json           # Quality metrics
├── metadata.json          # Registration information
└── visualization.png      # Result visualization
```

---

## Next Steps

1. **Review**: Check the README.md for detailed documentation
2. **Experiment**: Try different configurations in config.json
3. **Validate**: Run demo.py to understand all features
4. **Integrate**: Use pipeline in your workflow
5. **Optimize**: Adjust settings based on your data

---

## Real Data: Using Chandrayaan-2 Images

```python
# Download from ISRO links:
# - CHMAP: https://chmapbrowse.issdc.gov.in/
# - LROC: https://lroc.im-ldi.com/images/downloads/

from pipeline import LunarImageRegistrationPipeline

pipeline = LunarImageRegistrationPipeline()

# Register Chandrayaan-2 OHRC to TMC
result = pipeline.register_pair(
    'ch2_ohrc_image.tif',
    'ch2_tmc_image.tif',
    'ch2_registered.tif'
)

# Save comprehensive results
pipeline.save_results(result, './ch2_results')
```

---

## Performance Tips

### For Speed
```python
config = {
    'feature_method': 'orb',          # Fast detector
    'enhance_preprocessing': False,    # Skip enhancement
    'refine_alignment': False          # Skip refinement
}
```

### For Accuracy
```python
config = {
    'feature_method': 'sift',         # Accurate detector
    'enhance_preprocessing': True,     # Full preprocessing
    'remove_illumination_bias': True,  # Remove sun angle effects
    'refine_alignment': True           # Polish alignment
}
```

---

## What's Happening Behind the Scenes?

```
1. Load Images
   ↓
2. Preprocess (normalize, enhance, remove bias)
   ↓
3. Extract Features (SIFT keypoints + descriptors)
   ↓
4. Match Features (FLANN matcher, Lowe's ratio test)
   ↓
5. Filter Outliers (RANSAC)
   ↓
6. Compute Transformation (Homography matrix)
   ↓
7. Refine Alignment (Intensity-based optimization)
   ↓
8. Apply Transformation (Warp source to reference)
   ↓
9. Compute Metrics (Quality scores)
   ↓
10. Save Results (Registered image + metadata)
```

---

## Common Questions

**Q: Can I use this with real Chandrayaan-2 images?**
A: Yes! Download from ISRO's CHMAP or LROC databases and use directly.

**Q: How long does processing take?**
A: Depends on image size and settings. Typically 5-30 seconds per image pair.

**Q: Can I process multiple images in parallel?**
A: Yes, use `batch_register()` method for automatic parallel processing.

**Q: What's the maximum image size?**
A: Limited only by available RAM. For 16GB RAM, ~4000x4000 pixels is comfortable.

**Q: Can I adjust the quality metrics?**
A: Yes, edit config.json to enable/disable specific metrics and adjust weights.

---

## Support

1. **For errors**: Check output messages and troubleshooting section
2. **For configuration**: See config.json documentation in README
3. **For API details**: Check docstrings in each module
4. **For examples**: Run demo.py to see all features

---

**Ready to start? Run: `python demo.py`**
