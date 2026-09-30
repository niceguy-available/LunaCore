"""
Image registration and geometric transformation module
Handles alignment and transformation computation
"""

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import affine_transform
import matplotlib.pyplot as plt


class ImageRegistration:
    """Handle image registration and alignment"""
    
    def __init__(self, transformation_type='homography'):
        """
        Initialize registration
        
        Args:
            transformation_type: 'affine', 'homography', or 'rigid'
        """
        self.transformation_type = transformation_type
        self.transformation_matrix = None
    
    def compute_transformation(self, src_pts, dst_pts, method='homography'):
        """Compute geometric transformation between point sets"""
        src_pts = np.asarray(src_pts, dtype=np.float32)
        dst_pts = np.asarray(dst_pts, dtype=np.float32)
        
        if len(src_pts) < 4:
            print(f"Warning: Need at least 4 points, got {len(src_pts)}")
            return None
        
        if method == 'homography':
            # 8 degrees of freedom
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            return H
        
        elif method == 'affine':
            # 6 degrees of freedom
            if len(src_pts) >= 3:
                H = cv2.getAffineTransform(src_pts[:3], dst_pts[:3])
                # Convert to 3x3 for consistency
                H = np.vstack([H, [0, 0, 1]])
                return H
        
        elif method == 'rigid':
            # 4 degrees of freedom (translation, rotation, scale)
            return self._compute_rigid_transform(src_pts, dst_pts)
        
        elif method == 'similarity':
            # 4 degrees of freedom
            return self._compute_similarity_transform(src_pts, dst_pts)
        
        return None
    
    def _compute_rigid_transform(self, src_pts, dst_pts):
        """Compute rigid transformation (rotation + translation)"""
        # Center points
        src_center = np.mean(src_pts, axis=0)
        dst_center = np.mean(dst_pts, axis=0)
        
        src_centered = src_pts - src_center
        dst_centered = dst_pts - dst_center
        
        # SVD for rotation
        H = src_centered.T @ dst_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Ensure proper rotation (det = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Translation
        t = dst_center - R @ src_center
        
        # Build transformation matrix
        T = np.eye(3)
        T[:2, :2] = R[:2, :2]
        T[:2, 2] = t
        
        return T
    
    def _compute_similarity_transform(self, src_pts, dst_pts):
        """Compute similarity transformation (rotation + scale + translation)"""
        # Center points
        src_center = np.mean(src_pts, axis=0)
        dst_center = np.mean(dst_pts, axis=0)
        
        src_centered = src_pts - src_center
        dst_centered = dst_pts - dst_center
        
        # Compute scale
        src_scale = np.sqrt(np.sum(src_centered ** 2))
        dst_scale = np.sqrt(np.sum(dst_centered ** 2))
        scale = dst_scale / (src_scale + 1e-8)
        
        # Normalized points
        src_norm = src_centered / (src_scale + 1e-8)
        dst_norm = dst_centered / (dst_scale + 1e-8)
        
        # SVD for rotation
        H = src_norm.T @ dst_norm
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Build transformation matrix
        T = np.eye(3)
        T[:2, :2] = scale * R[:2, :2]
        T[:2, 2] = dst_center - scale * R[:2, :2] @ src_center
        
        return T
    
    def apply_transformation(self, image, transformation_matrix, output_shape=None):
        """Apply geometric transformation to image"""
        if transformation_matrix is None:
            return image
        
        if output_shape is None:
            output_shape = image.shape
        
        # Use warp perspective for homography
        if transformation_matrix.shape == (3, 3):
            result = cv2.warpPerspective(image, transformation_matrix, 
                                        (output_shape[1], output_shape[0]),
                                        flags=cv2.INTER_LINEAR)
        else:
            result = cv2.warpAffine(image, transformation_matrix[:2],
                                   (output_shape[1], output_shape[0]),
                                   flags=cv2.INTER_LINEAR)
        
        return result
    
    def refine_alignment(self, src_img, dst_img, initial_transform=None, 
                        max_iterations=100):
        """Refine alignment using intensity-based optimization"""
        if initial_transform is None:
            initial_transform = np.eye(3)
        
        def alignment_error(params):
            """Compute alignment error for given transformation parameters"""
            # Unpack parameters (4 for similarity, 6 for affine)
            if len(params) == 4:
                # Similarity: [tx, ty, angle, scale]
                tx, ty, angle, scale = params
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                T = np.array([[scale * cos_a, -scale * sin_a, tx],
                             [scale * sin_a, scale * cos_a, ty],
                             [0, 0, 1]])
            else:
                # Affine: flatten 2x3 matrix
                T = np.vstack([params.reshape(2, 3), [0, 0, 1]])
            
            # Apply transformation
            warped = cv2.warpPerspective(src_img, T, 
                                        (dst_img.shape[1], dst_img.shape[0]))
            
            # Compute SSD (Sum of Squared Differences)
            error = np.sum((warped.astype(np.float32) - dst_img.astype(np.float32)) ** 2)
            return error
        
        # Initial parameters
        if initial_transform is not None:
            tx = initial_transform[0, 2]
            ty = initial_transform[1, 2]
            angle = np.arctan2(initial_transform[1, 0], initial_transform[0, 0])
            scale = initial_transform[0, 0] / np.cos(angle)
            initial_params = np.array([tx, ty, angle, scale])
        else:
            initial_params = np.array([0, 0, 0, 1])
        
        # Optimize
        result = minimize(alignment_error, initial_params, method='Powell',
                         options={'maxiter': max_iterations})
        
        # Extract optimized transformation
        params = result.x
        tx, ty, angle, scale = params
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        T_refined = np.array([[scale * cos_a, -scale * sin_a, tx],
                             [scale * sin_a, scale * cos_a, ty],
                             [0, 0, 1]])
        
        return T_refined, result.fun
    
    def extract_transformation_params(self, T):
        """Extract transformation parameters (translation, rotation, scale)"""
        tx = T[0, 2]
        ty = T[1, 2]
        
        # Extract rotation and scale
        a = T[0, 0]
        b = T[0, 1]
        c = T[1, 0]
        d = T[1, 1]
        
        scale = np.sqrt(a**2 + b**2)
        angle = np.arctan2(c, a)
        
        return {
            'translation': (tx, ty),
            'rotation': np.degrees(angle),
            'scale': scale
        }
    
    def compose_transformations(self, T1, T2):
        """Compose two transformation matrices"""
        return T2 @ T1
    
    def invert_transformation(self, T):
        """Invert a transformation matrix"""
        try:
            return np.linalg.inv(T)
        except:
            return None


class MultiModalRegistration:
    """Handle registration of multi-modal lunar images"""
    
    def __init__(self):
        self.registration = ImageRegistration()
    
    def register_images(self, src_img, dst_img, src_pts, dst_pts, 
                       method='homography', refine=True):
        """
        Register source image to destination image
        
        Args:
            src_img: Source image
            dst_img: Destination/reference image
            src_pts: Source keypoint coordinates
            dst_pts: Destination keypoint coordinates
            method: Transformation method ('homography', 'affine', 'rigid')
            refine: Whether to refine with intensity-based optimization
        
        Returns:
            registered_img: Aligned source image
            transformation: Transformation matrix
            error: Registration error
        """
        # Compute initial transformation
        T = self.registration.compute_transformation(src_pts, dst_pts, method)
        
        if T is None:
            print("Failed to compute transformation")
            return src_img, None, float('inf')
        
        # Refine transformation
        if refine:
            T, error = self.registration.refine_alignment(src_img, dst_img, T)
        else:
            # Compute alignment error
            warped = self.registration.apply_transformation(src_img, T, dst_img.shape)
            error = np.sum((warped.astype(np.float32) - dst_img.astype(np.float32)) ** 2)
        
        # Apply transformation
        registered = self.registration.apply_transformation(src_img, T, dst_img.shape)
        
        return registered, T, error
    
    def align_multimodal_series(self, images_dict, reference_key, method='homography'):
        """
        Align multiple modal images to a reference image
        
        Args:
            images_dict: Dictionary of images from different sensors
            reference_key: Key of reference image
            method: Transformation method
        
        Returns:
            aligned_dict: Dictionary of aligned images
            transforms: Dictionary of transformation matrices
        """
        reference = images_dict[reference_key]
        aligned_dict = {reference_key: reference}
        transforms = {reference_key: np.eye(3)}
        
        for key, img in images_dict.items():
            if key == reference_key:
                continue
            
            print(f"Aligning {key} to {reference_key}...")
            
            # This would typically use feature matching
            # For now, we just apply a basic transformation
            T = np.eye(3)  # Placeholder
            aligned = img
            
            aligned_dict[key] = aligned
            transforms[key] = T
        
        return aligned_dict, transforms
