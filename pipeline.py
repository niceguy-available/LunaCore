"""
Main registration pipeline - Orchestrates the complete workflow
"""

import cv2
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
import json

from preprocessing import LunarImagePreprocessor, ImageNormalizer
from feature_extraction import FeatureExtractor, FeatureMatcher, LunarLandmarkDetector
from registration import ImageRegistration, MultiModalRegistration
from validation import RegistrationValidator, RegistrationReport


class LunarImageRegistrationPipeline:
    """Complete pipeline for lunar image registration"""
    
    def __init__(self, config=None):
        """
        Initialize pipeline
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()
        self.preprocessor = LunarImagePreprocessor(debug=self.config.get('debug', False))
        self.feature_extractor = FeatureExtractor(method=self.config['feature_method'])
        self.feature_matcher = FeatureMatcher(method=self.config['matcher_method'])
        self.registration = MultiModalRegistration()
        self.validator = RegistrationValidator()
        self.report_generator = RegistrationReport()
        
        self.results = {}
    
    def _default_config(self):
        """Default configuration"""
        return {
            'feature_method': 'sift',
            'matcher_method': 'flann',
            'transformation_type': 'homography',
            'enhance_preprocessing': True,
            'remove_illumination_bias': True,
            'refine_alignment': True,
            'debug': False,
            'save_intermediates': False
        }
    
    def register_pair(self, source_path, reference_path, output_path=None):
        """
        Register a single image pair
        
        Args:
            source_path: Path to source image
            reference_path: Path to reference image
            output_path: Path to save registered image
        
        Returns:
            result_dict: Dictionary containing results
        """
        print(f"\n{'='*70}")
        print(f"REGISTERING: {Path(source_path).name} -> {Path(reference_path).name}")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        # Load images
        print("\n[1/7] Loading images...")
        src_img = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
        ref_img = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
        
        if src_img is None or ref_img is None:
            print("ERROR: Could not load images")
            return None
        
        print(f"  Source: {src_img.shape}, Reference: {ref_img.shape}")
        
        # Preprocessing
        print("\n[2/7] Preprocessing images...")
        src_preprocessed = self.preprocessor.preprocess_pipeline(
            src_img,
            enhance=self.config['enhance_preprocessing'],
            remove_bias=self.config['remove_illumination_bias']
        )
        ref_preprocessed = self.preprocessor.preprocess_pipeline(
            ref_img,
            enhance=self.config['enhance_preprocessing'],
            remove_bias=self.config['remove_illumination_bias']
        )
        
        # Feature extraction
        print("\n[3/7] Extracting features...")
        src_kp, src_desc = self.feature_extractor.detect_and_compute(src_preprocessed)
        ref_kp, ref_desc = self.feature_extractor.detect_and_compute(ref_preprocessed)
        
        print(f"  Source keypoints: {len(src_kp)}")
        print(f"  Reference keypoints: {len(ref_kp)}")
        
        if len(src_kp) < 4 or len(ref_kp) < 4:
            print("WARNING: Not enough keypoints for registration")
            return None
        
        # Feature matching
        print("\n[4/7] Matching features...")
        matches, src_pts, dst_pts = self.feature_matcher.filter_matches_ransac(
            src_kp, src_desc, ref_kp, ref_desc
        )
        
        print(f"  Matches found: {len(matches)}")
        
        if len(matches) < 4:
            print("ERROR: Not enough matches for registration")
            return None
        
        # Compute transformation
        print("\n[5/7] Computing transformation...")
        T = self.registration.registration.compute_transformation(
            src_pts, dst_pts,
            method=self.config['transformation_type']
        )
        
        if T is None:
            print("ERROR: Could not compute transformation")
            return None
        
        # Refine alignment
        if self.config['refine_alignment']:
            print("\n[6/7] Refining alignment (this may take a moment)...")
            T, refinement_error = self.registration.registration.refine_alignment(
                src_preprocessed, ref_preprocessed, T
            )
            print(f"  Refinement error: {refinement_error:.4f}")
        
        # Apply transformation
        print("\n[7/7] Applying transformation...")
        registered = self.registration.registration.apply_transformation(
            src_img, T, ref_img.shape
        )
        
        # Validation
        print("\nValidating registration...")
        metrics_dict = self.validator.compute_all_metrics(ref_img, registered)
        quality_score = self.validator.compute_registration_quality_score(metrics_dict)
        
        # Extract transformation parameters
        params = self.registration.registration.extract_transformation_params(T)
        
        # Prepare results
        result_dict = {
            'source_image': src_img,
            'reference_image': ref_img,
            'registered_image': registered,
            'transformation_matrix': T.tolist(),
            'transformation_params': params,
            'keypoints_src': len(src_kp),
            'keypoints_ref': len(ref_kp),
            'matches': len(matches),
            'metrics': metrics_dict,
            'quality_score': quality_score,
            'processing_time': time.time() - start_time
        }
        
        # Save results
        if output_path:
            cv2.imwrite(str(output_path), registered)
            print(f"\nRegistered image saved to: {output_path}")
        
        # Print report
        report = self.report_generator.generate_report(ref_img, registered, 
                                                       metrics_dict=metrics_dict)
        self.report_generator.print_report(report)
        
        print(f"Processing time: {result_dict['processing_time']:.2f} seconds")
        
        return result_dict
    
    def register_series(self, image_dict, reference_key, output_dir=None):
        """
        Register multiple images to a reference
        
        Args:
            image_dict: Dictionary of image paths {key: path}
            reference_key: Key of reference image
            output_dir: Directory to save results
        
        Returns:
            results_dict: Dictionary of results for each image
        """
        results_dict = {}
        reference_path = image_dict[reference_key]
        
        print(f"\n{'='*70}")
        print(f"REGISTERING IMAGE SERIES TO: {reference_key}")
        print(f"{'='*70}")
        
        for key, img_path in image_dict.items():
            if key == reference_key:
                print(f"\nSkipping reference image: {key}")
                results_dict[key] = {'status': 'reference'}
                continue
            
            output_path = None
            if output_dir:
                output_path = Path(output_dir) / f"{key}_registered.png"
            
            result = self.register_pair(img_path, reference_path, output_path)
            results_dict[key] = result if result else {'status': 'failed'}
        
        return results_dict
    
    def batch_register(self, source_dir, reference_dir, output_dir=None):
        """
        Batch register all images in source_dir to images in reference_dir
        
        Args:
            source_dir: Directory containing source images
            reference_dir: Directory containing reference images
            output_dir: Directory to save results
        """
        source_dir = Path(source_dir)
        reference_dir = Path(reference_dir)
        output_dir = Path(output_dir) if output_dir else source_dir / 'registered'
        output_dir.mkdir(exist_ok=True, parents=True)
        
        source_images = sorted(source_dir.glob('*.png')) + sorted(source_dir.glob('*.jpg'))
        reference_images = sorted(reference_dir.glob('*.png')) + sorted(reference_dir.glob('*.jpg'))
        
        results = {}
        
        for src_img in tqdm(source_images, desc="Registering images"):
            for ref_img in reference_images:
                key = f"{src_img.stem}_to_{ref_img.stem}"
                output_path = output_dir / f"{key}_registered.png"
                
                result = self.register_pair(str(src_img), str(ref_img), str(output_path))
                results[key] = result
        
        return results
    
    def visualize_results(self, result_dict, save_path=None):
        """Visualize registration results"""
        if result_dict is None or 'registered_image' not in result_dict:
            print("Invalid result dictionary")
            return
        
        fig = self.validator.visualize_registration_result(
            result_dict['reference_image'],
            result_dict['registered_image'],
            result_dict['metrics']
        )
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        return fig
    
    def save_results(self, result_dict, output_prefix):
        """Save all results to disk"""
        output_prefix = Path(output_prefix)
        output_prefix.mkdir(exist_ok=True, parents=True)
        
        # Save registered image
        if 'registered_image' in result_dict:
            cv2.imwrite(str(output_prefix / 'registered.png'), 
                       result_dict['registered_image'])
        
        # Save transformation matrix
        T = np.array(result_dict['transformation_matrix'])
        np.save(str(output_prefix / 'transformation.npy'), T)
        
        # Save metrics
        metrics = result_dict['metrics']
        with open(str(output_prefix / 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save metadata
        metadata = {
            'keypoints_src': result_dict['keypoints_src'],
            'keypoints_ref': result_dict['keypoints_ref'],
            'matches': result_dict['matches'],
            'quality_score': result_dict['quality_score'],
            'processing_time': result_dict['processing_time'],
            'transformation_params': result_dict['transformation_params']
        }
        with open(str(output_prefix / 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Results saved to: {output_prefix}")


def main():
    """Example usage"""
    # Create pipeline
    config = {
        'feature_method': 'sift',
        'matcher_method': 'flann',
        'transformation_type': 'homography',
        'enhance_preprocessing': True,
        'refine_alignment': True,
        'debug': False
    }
    
    pipeline = LunarImageRegistrationPipeline(config)
    
    # Example: Register a single pair
    # result = pipeline.register_pair('source.png', 'reference.png', 'output.png')
    # pipeline.visualize_results(result, 'result_visualization.png')
    
    print("Pipeline ready for use")


if __name__ == "__main__":
    main()
