"""
Preprocessing module for lunar image registration
Handles normalization, geometric corrections, and enhancement
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage import exposure, filters
import matplotlib.pyplot as plt


class LunarImagePreprocessor:
    """Handles preprocessing of lunar images from different sensors"""
    
    def __init__(self, debug=False):
        self.debug = debug
        
    def normalize_intensity(self, image):
        """Normalize image intensity to handle illumination variations"""
        # Convert to float
        img_float = image.astype(np.float32)
        
        # Clip extreme values
        p2, p98 = np.percentile(img_float, (2, 98))
        img_clipped = np.clip(img_float, p2, p98)
        
        # Normalize to 0-1
        img_norm = (img_clipped - p2) / (p98 - p2 + 1e-8)
        
        return np.clip(img_norm * 255, 0, 255).astype(np.uint8)
    
    def enhance_contrast(self, image, clip_limit=2.0, tile_size=8):
        """Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        enhanced = clahe.apply(image)
        return enhanced
    
    def geometric_correction(self, image, correction_type='bilateral'):
        """Apply geometric corrections to handle viewpoint variations"""
        if correction_type == 'bilateral':
            # Bilateral filter preserves edges while smoothing
            corrected = cv2.bilateralFilter(image, 9, 75, 75)
        elif correction_type == 'morphological':
            # Morphological operations for feature enhancement
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            corrected = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        else:
            corrected = image
            
        return corrected
    
    def create_image_pyramid(self, image, levels=4):
        """Create multi-scale pyramid for scale invariance"""
        pyramid = [image]
        current = image.copy()
        
        for i in range(levels - 1):
            current = cv2.pyrDown(current)
            pyramid.append(current)
            
        return pyramid
    
    def remove_illumination_bias(self, image, kernel_size=31):
        """Remove illumination bias using morphological operations"""
        # Create background estimate
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        # Subtract background to enhance local features
        img_float = image.astype(np.float32)
        bg_float = background.astype(np.float32)
        
        result = (img_float - bg_float + 128).astype(np.uint8)
        return np.clip(result, 0, 255)
    
    def preprocess_pipeline(self, image, enhance=True, remove_bias=True, 
                           create_pyramid=False, levels=4):
        """Complete preprocessing pipeline"""
        # Step 1: Normalize intensity
        processed = self.normalize_intensity(image)
        
        # Step 2: Remove illumination bias
        if remove_bias:
            processed = self.remove_illumination_bias(processed)
        
        # Step 3: Enhance contrast
        if enhance:
            processed = self.enhance_contrast(processed)
        
        # Step 4: Geometric correction
        processed = self.geometric_correction(processed)
        
        # Step 5: Create pyramid if needed
        if create_pyramid:
            pyramid = self.create_image_pyramid(processed, levels)
            return processed, pyramid
        
        return processed
    
    def visualize_preprocessing(self, image, title="Preprocessing Steps"):
        """Visualize preprocessing stages"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # Original
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        # Normalized
        norm = self.normalize_intensity(image)
        axes[0, 1].imshow(norm, cmap='gray')
        axes[0, 1].set_title('Normalized')
        axes[0, 1].axis('off')
        
        # Bias removed
        bias_removed = self.remove_illumination_bias(norm)
        axes[0, 2].imshow(bias_removed, cmap='gray')
        axes[0, 2].set_title('Bias Removed')
        axes[0, 2].axis('off')
        
        # Contrast enhanced
        enhanced = self.enhance_contrast(bias_removed)
        axes[1, 0].imshow(enhanced, cmap='gray')
        axes[1, 0].set_title('Contrast Enhanced')
        axes[1, 0].axis('off')
        
        # Geometrically corrected
        corrected = self.geometric_correction(enhanced)
        axes[1, 1].imshow(corrected, cmap='gray')
        axes[1, 1].set_title('Geometric Corrected')
        axes[1, 1].axis('off')
        
        # Final result
        final = self.preprocess_pipeline(image)
        axes[1, 2].imshow(final, cmap='gray')
        axes[1, 2].set_title('Final Preprocessed')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        return fig


class ImageNormalizer:
    """Handles multi-modal image normalization"""
    
    @staticmethod
    def normalize_multimodal(images_dict):
        """Normalize images from different sensors to same scale"""
        normalized = {}
        
        # Get global min/max across all images
        all_values = []
        for img in images_dict.values():
            all_values.extend(img.flatten())
        
        global_min = np.min(all_values)
        global_max = np.max(all_values)
        
        # Normalize each image
        for key, img in images_dict.items():
            norm_img = (img.astype(np.float32) - global_min) / (global_max - global_min + 1e-8)
            normalized[key] = (norm_img * 255).astype(np.uint8)
        
        return normalized
    
    @staticmethod
    def equalize_histograms(image1, image2):
        """Equalize histograms between two images"""
        # Match histogram of image2 to image1
        matched = exposure.match_histograms(image2.astype(np.float32), 
                                           image1.astype(np.float32))
        return matched.astype(np.uint8)
