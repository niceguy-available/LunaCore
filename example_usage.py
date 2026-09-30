"""
Simple example usage of the Lunar Image Registration Pipeline
Run this to see basic functionality
"""

import cv2
import numpy as np
from pathlib import Path
from pipeline import LunarImageRegistrationPipeline


def example_1_basic_registration():
    """Example 1: Basic image pair registration"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Image Pair Registration")
    print("="*70)
    
    # Create pipeline with default settings
    pipeline = LunarImageRegistrationPipeline()
    
    # Register two images (replace with your image paths)
    # result = pipeline.register_pair(
    #     'source_image.png',
    #     'reference_image.png',
    #     'registered_output.png'
    # )
    
    # Check results
    # if result:
    #     print(f"\nRegistration Quality Score: {result['quality_score']:.2f}/100")
    #     print(f"Matches Found: {result['matches']}")
    
    print("\nUncomment the code and replace image paths to use this example")


def example_2_with_visualization():
    """Example 2: Registration with visualization"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Registration with Visualization")
    print("="*70)
    
    pipeline = LunarImageRegistrationPipeline()
    
    # result = pipeline.register_pair(
    #     'source_image.png',
    #     'reference_image.png',
    #     'registered_output.png'
    # )
    
    # if result:
    #     # Visualize and save
    #     pipeline.visualize_results(result, 'registration_result.png')
    #     print("Visualization saved to: registration_result.png")
    
    print("\nUncomment the code and replace image paths to use this example")


def example_3_custom_configuration():
    """Example 3: Using custom configuration"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Custom Configuration")
    print("="*70)
    
    # Custom settings for faster processing
    custom_config = {
        'feature_method': 'orb',           # Faster than SIFT
        'matcher_method': 'bf',            # Brute force matcher
        'transformation_type': 'affine',   # Simpler transformation
        'enhance_preprocessing': True,
        'remove_illumination_bias': False,
        'refine_alignment': False          # Skip refinement for speed
    }
    
    print("\nCustom Configuration:")
    for key, value in custom_config.items():
        print(f"  {key}: {value}")
    
    pipeline = LunarImageRegistrationPipeline(custom_config)
    
    # result = pipeline.register_pair(
    #     'source_image.png',
    #     'reference_image.png'
    # )
    
    print("\nUncomment the code and replace image paths to use this example")


def example_4_batch_processing():
    """Example 4: Batch processing multiple images"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Batch Processing")
    print("="*70)
    
    pipeline = LunarImageRegistrationPipeline()
    
    # results = pipeline.batch_register(
    #     source_dir='./images/source',
    #     reference_dir='./images/reference',
    #     output_dir='./images/registered'
    # )
    
    # print(f"Processed {len(results)} image pairs")
    # for key, result in results.items():
    #     if result and 'quality_score' in result:
    #         print(f"  {key}: Score = {result['quality_score']:.2f}/100")
    
    print("\nUncomment the code and set appropriate directories to use this example")


def example_5_examining_results():
    """Example 5: Detailed result examination"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Examining Registration Results")
    print("="*70)
    
    print("\nAfter registration, access results like this:")
    print("""
    # Quality metrics
    print(f"RMSE: {result['metrics']['rmse']:.2f}")
    print(f"SSIM: {result['metrics']['ssim']:.4f}")
    print(f"Correlation: {result['metrics']['correlation']:.4f}")
    print(f"Mutual Information: {result['metrics']['mutual_info']:.4f}")
    
    # Transformation info
    params = result['transformation_params']
    print(f"Translation: {params['translation']}")
    print(f"Rotation: {params['rotation']:.2f}°")
    print(f"Scale: {params['scale']:.4f}")
    
    # Keypoint statistics
    print(f"Source keypoints: {result['keypoints_src']}")
    print(f"Reference keypoints: {result['keypoints_ref']}")
    print(f"Matches: {result['matches']}")
    
    # Processing info
    print(f"Processing time: {result['processing_time']:.2f} seconds")
    """)


def example_6_preprocessing_only():
    """Example 6: Just preprocessing without registration"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Preprocessing Only")
    print("="*70)
    
    from preprocessing import LunarImagePreprocessor
    
    print("\nPreprocessing without registration:")
    print("""
    preprocessor = LunarImagePreprocessor()
    
    # Load image
    image = cv2.imread('input.png', cv2.IMREAD_GRAYSCALE)
    
    # Full preprocessing
    processed = preprocessor.preprocess_pipeline(
        image,
        enhance=True,
        remove_bias=True
    )
    
    # Save
    cv2.imwrite('preprocessed.png', processed)
    
    # Visualize stages
    fig = preprocessor.visualize_preprocessing(image)
    fig.savefig('preprocessing_stages.png')
    """)


def example_7_feature_extraction_only():
    """Example 7: Just feature extraction"""
    print("\n" + "="*70)
    print("EXAMPLE 7: Feature Extraction Only")
    print("="*70)
    
    print("\nFeature extraction without registration:")
    print("""
    from feature_extraction import FeatureExtractor
    
    extractor = FeatureExtractor(method='sift')
    
    # Extract features
    image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
    keypoints, descriptors = extractor.detect_and_compute(image)
    
    print(f"Found {len(keypoints)} keypoints")
    
    # Draw keypoints
    img_kp = cv2.drawKeypoints(image, keypoints, None)
    cv2.imwrite('keypoints.png', img_kp)
    """)


def example_8_load_configuration():
    """Example 8: Load configuration from file"""
    print("\n" + "="*70)
    print("EXAMPLE 8: Loading Configuration from File")
    print("="*70)
    
    print("\nLoad configuration from config.json:")
    print("""
    import json
    from pipeline import LunarImageRegistrationPipeline
    
    # Load full config
    with open('config.json', 'r') as f:
        full_config = json.load(f)
    
    # Extract pipeline config
    pipeline_config = {
        'feature_method': full_config['feature_extraction']['method'],
        'matcher_method': full_config['feature_matching']['method'],
        'transformation_type': full_config['registration']['transformation_type'],
        'enhance_preprocessing': full_config['preprocessing']['enhance_contrast'],
        'refine_alignment': full_config['registration']['refine_alignment']
    }
    
    # Create pipeline
    pipeline = LunarImageRegistrationPipeline(pipeline_config)
    """)


def example_9_save_detailed_results():
    """Example 9: Save detailed results"""
    print("\n" + "="*70)
    print("EXAMPLE 9: Saving Detailed Results")
    print("="*70)
    
    print("\nSave all registration results:")
    print("""
    # Register
    result = pipeline.register_pair('source.png', 'reference.png')
    
    # Save everything
    pipeline.save_results(result, './registration_results')
    
    # This creates:
    # - registration_results/registered.png (aligned image)
    # - registration_results/transformation.npy (3x3 matrix)
    # - registration_results/metrics.json (quality metrics)
    # - registration_results/metadata.json (registration info)
    """)


def example_10_comparing_configurations():
    """Example 10: Compare different configurations"""
    print("\n" + "="*70)
    print("EXAMPLE 10: Comparing Different Configurations")
    print("="*70)
    
    print("\nCompare fast vs accurate registration:")
    print("""
    # Fast configuration
    fast_config = {
        'feature_method': 'orb',
        'enhance_preprocessing': False,
        'refine_alignment': False
    }
    
    # Accurate configuration
    accurate_config = {
        'feature_method': 'sift',
        'enhance_preprocessing': True,
        'refine_alignment': True
    }
    
    # Test both
    pipeline_fast = LunarImageRegistrationPipeline(fast_config)
    pipeline_accurate = LunarImageRegistrationPipeline(accurate_config)
    
    import time
    
    # Time fast method
    start = time.time()
    result_fast = pipeline_fast.register_pair('source.png', 'reference.png')
    time_fast = time.time() - start
    
    # Time accurate method
    start = time.time()
    result_accurate = pipeline_accurate.register_pair('source.png', 'reference.png')
    time_accurate = time.time() - start
    
    print(f"Fast: {time_fast:.2f}s, Score: {result_fast['quality_score']:.1f}")
    print(f"Accurate: {time_accurate:.2f}s, Score: {result_accurate['quality_score']:.1f}")
    """)


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("LUNAR IMAGE REGISTRATION - EXAMPLE USAGE GUIDE")
    print("="*80)
    
    # List all examples
    examples = [
        example_1_basic_registration,
        example_2_with_visualization,
        example_3_custom_configuration,
        example_4_batch_processing,
        example_5_examining_results,
        example_6_preprocessing_only,
        example_7_feature_extraction_only,
        example_8_load_configuration,
        example_9_save_detailed_results,
        example_10_comparing_configurations
    ]
    
    # Run examples
    for example_func in examples:
        example_func()
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""
1. Replace placeholder image paths with your actual lunar images
2. Adjust configuration parameters in custom_config dictionary
3. Run the pipeline: result = pipeline.register_pair(...)
4. Examine results and visualize: pipeline.visualize_results(result)
5. Save results: pipeline.save_results(result, output_dir)

For more details, see:
- README.md: Comprehensive documentation
- QUICKSTART.md: Quick reference guide
- config.json: All configurable parameters
- demo.py: Complete working examples
    """)


if __name__ == "__main__":
    main()
