"""
Validation and quality metrics for image registration
Quantitative and qualitative assessment
"""

import cv2
import numpy as np
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


class RegistrationMetrics:
    """Compute metrics to evaluate registration quality"""
    
    @staticmethod
    def root_mean_square_error(img1, img2):
        """Compute RMSE between two images"""
        if img1.shape != img2.shape:
            return float('inf')
        
        mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
        rmse = np.sqrt(mse)
        return rmse
    
    @staticmethod
    def mean_absolute_error(img1, img2):
        """Compute MAE between two images"""
        if img1.shape != img2.shape:
            return float('inf')
        
        mae = np.mean(np.abs(img1.astype(np.float32) - img2.astype(np.float32)))
        return mae
    
    @staticmethod
    def correlation_coefficient(img1, img2):
        """Compute normalized cross-correlation"""
        if img1.shape != img2.shape:
            return 0.0
        
        # Flatten images
        flat1 = img1.flatten().astype(np.float32)
        flat2 = img2.flatten().astype(np.float32)
        
        # Compute correlation
        corr = np.corrcoef(flat1, flat2)[0, 1]
        return corr if not np.isnan(corr) else 0.0
    
    @staticmethod
    def structural_similarity(img1, img2, data_range=255):
        """Compute SSIM between two images"""
        if img1.shape != img2.shape:
            return 0.0
        
        ssim_val = ssim(img1, img2, data_range=data_range)
        return ssim_val
    
    @staticmethod
    def mutual_information(img1, img2, bins=256):
        """Compute mutual information between images"""
        if img1.shape != img2.shape:
            return 0.0
        
        # Compute histograms
        h1 = np.histogram(img1.flatten(), bins=bins)[0]
        h2 = np.histogram(img2.flatten(), bins=bins)[0]
        h12 = np.histogram2d(img1.flatten(), img2.flatten(), bins=bins)[0]
        
        # Normalize
        h1 = h1 / np.sum(h1)
        h2 = h2 / np.sum(h2)
        h12 = h12 / np.sum(h12)
        
        # Compute MI
        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if h12[i, j] > 0:
                    mi += h12[i, j] * np.log(h12[i, j] / (h1[i] * h2[j] + 1e-8))
        
        return mi
    
    @staticmethod
    def gradient_correlation(img1, img2):
        """Compute correlation of image gradients"""
        # Compute gradients
        gx1 = cv2.Sobel(img1, cv2.CV_32F, 1, 0, ksize=3)
        gy1 = cv2.Sobel(img1, cv2.CV_32F, 0, 1, ksize=3)
        
        gx2 = cv2.Sobel(img2, cv2.CV_32F, 1, 0, ksize=3)
        gy2 = cv2.Sobel(img2, cv2.CV_32F, 0, 1, ksize=3)
        
        # Compute correlation
        gx_corr = np.corrcoef(gx1.flatten(), gx2.flatten())[0, 1]
        gy_corr = np.corrcoef(gy1.flatten(), gy2.flatten())[0, 1]
        
        gx_corr = gx_corr if not np.isnan(gx_corr) else 0.0
        gy_corr = gy_corr if not np.isnan(gy_corr) else 0.0
        
        return (gx_corr + gy_corr) / 2


class RegistrationValidator:
    """Validate registration results"""
    
    def __init__(self):
        self.metrics = RegistrationMetrics()
    
    def compute_all_metrics(self, reference, registered, gt_transform=None):
        """Compute all quality metrics"""
        metrics_dict = {
            'rmse': self.metrics.root_mean_square_error(reference, registered),
            'mae': self.metrics.mean_absolute_error(reference, registered),
            'correlation': self.metrics.correlation_coefficient(reference, registered),
            'ssim': self.metrics.structural_similarity(reference, registered),
            'mutual_info': self.metrics.mutual_information(reference, registered),
            'gradient_corr': self.metrics.gradient_correlation(reference, registered)
        }
        
        return metrics_dict
    
    def validate_point_accuracy(self, kp_src, kp_dst, transformation):
        """Validate registration using keypoint accuracy"""
        if transformation is None or len(kp_src) == 0:
            return float('inf')
        
        # Transform source points
        kp_src_h = np.hstack([kp_src, np.ones((len(kp_src), 1))])
        transformed = (transformation @ kp_src_h.T).T
        transformed = transformed[:, :2]
        
        # Compute error
        errors = np.linalg.norm(transformed - kp_dst, axis=1)
        
        return {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'max_error': np.max(errors),
            'min_error': np.min(errors),
            'median_error': np.median(errors)
        }
    
    def compute_registration_quality_score(self, metrics_dict, weights=None):
        """
        Compute overall registration quality score (0-100)
        
        Higher is better
        """
        if weights is None:
            weights = {
                'correlation': 0.3,
                'ssim': 0.3,
                'gradient_corr': 0.2,
                'mutual_info': 0.2
            }
        
        score = 0.0
        
        # Correlation: 0 to 1 normalized to 0-100
        score += weights.get('correlation', 0) * max(0, metrics_dict['correlation']) * 100
        
        # SSIM: -1 to 1, normalized to 0-100
        score += weights.get('ssim', 0) * ((metrics_dict['ssim'] + 1) / 2) * 100
        
        # Gradient correlation
        score += weights.get('gradient_corr', 0) * max(0, metrics_dict['gradient_corr']) * 100
        
        # Mutual information (normalized)
        mi_norm = np.exp(-metrics_dict['mutual_info'] / 100) if metrics_dict['mutual_info'] > 0 else 0
        score += weights.get('mutual_info', 0) * mi_norm * 100
        
        return np.clip(score, 0, 100)
    
    def visualize_registration_result(self, reference, registered, 
                                     metrics_dict=None, title="Registration Result"):
        """Visualize registration results"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(title, fontsize=16)
        
        # Reference image
        axes[0, 0].imshow(reference, cmap='gray')
        axes[0, 0].set_title('Reference Image')
        axes[0, 0].axis('off')
        
        # Registered image
        axes[0, 1].imshow(registered, cmap='gray')
        axes[0, 1].set_title('Registered Image')
        axes[0, 1].axis('off')
        
        # Overlay (checkerboard pattern)
        overlay = self._create_overlay(reference, registered, mode='checkerboard')
        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title('Overlay (Checkerboard)')
        axes[1, 0].axis('off')
        
        # Difference map
        if reference.shape == registered.shape:
            diff = cv2.absdiff(reference, registered)
            axes[1, 1].imshow(diff, cmap='hot')
            axes[1, 1].set_title('Absolute Difference')
            axes[1, 1].axis('off')
        
        # Add metrics text
        if metrics_dict:
            metrics_text = self._format_metrics(metrics_dict)
            fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=10, 
                    family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def _create_overlay(self, img1, img2, mode='checkerboard'):
        """Create overlay of two images"""
        if img1.shape != img2.shape:
            return None
        
        h, w = img1.shape
        
        if mode == 'checkerboard':
            # Checkerboard pattern
            overlay = np.zeros((h, w, 3), dtype=np.uint8)
            overlay[::2, ::2] = cv2.cvtColor(img1[::2, ::2], cv2.COLOR_GRAY2BGR)
            overlay[1::2, 1::2] = cv2.cvtColor(img2[1::2, 1::2], cv2.COLOR_GRAY2BGR)
        
        elif mode == 'blend':
            # 50-50 blend
            overlay = cv2.addWeighted(cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR), 0.5,
                                     cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR), 0.5, 0)
        
        else:  # mode == 'difference'
            diff = cv2.absdiff(img1, img2)
            overlay = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
        
        return overlay.astype(np.uint8)
    
    def _format_metrics(self, metrics_dict):
        """Format metrics for display"""
        text = "Registration Metrics:\n"
        text += f"  RMSE: {metrics_dict.get('rmse', 0):.2f}\n"
        text += f"  MAE: {metrics_dict.get('mae', 0):.2f}\n"
        text += f"  Correlation: {metrics_dict.get('correlation', 0):.4f}\n"
        text += f"  SSIM: {metrics_dict.get('ssim', 0):.4f}\n"
        text += f"  MI: {metrics_dict.get('mutual_info', 0):.4f}\n"
        text += f"  Gradient Corr: {metrics_dict.get('gradient_corr', 0):.4f}"
        return text


class RegistrationReport:
    """Generate comprehensive registration report"""
    
    def __init__(self):
        self.validator = RegistrationValidator()
    
    def generate_report(self, reference, registered, kp_error=None, metrics_dict=None):
        """Generate detailed registration report"""
        report = {}
        
        # Image statistics
        report['image_stats'] = {
            'reference_mean': np.mean(reference),
            'reference_std': np.std(reference),
            'registered_mean': np.mean(registered),
            'registered_std': np.std(registered),
            'reference_range': (np.min(reference), np.max(reference)),
            'registered_range': (np.min(registered), np.max(registered))
        }
        
        # Quality metrics
        if metrics_dict is None:
            metrics_dict = self.validator.compute_all_metrics(reference, registered)
        report['metrics'] = metrics_dict
        
        # Overall score
        report['quality_score'] = self.validator.compute_registration_quality_score(metrics_dict)
        
        # Point accuracy
        if kp_error:
            report['point_accuracy'] = kp_error
        
        return report
    
    def print_report(self, report):
        """Print formatted report"""
        print("\n" + "="*60)
        print("REGISTRATION REPORT")
        print("="*60)
        
        print("\nImage Statistics:")
        for key, val in report['image_stats'].items():
            if isinstance(val, tuple):
                print(f"  {key}: {val[0]:.2f} - {val[1]:.2f}")
            else:
                print(f"  {key}: {val:.2f}")
        
        print("\nRegistration Metrics:")
        for key, val in report['metrics'].items():
            print(f"  {key}: {val:.4f}")
        
        print(f"\nOverall Quality Score: {report['quality_score']:.2f}/100")
        
        if 'point_accuracy' in report:
            print("\nPoint Accuracy:")
            for key, val in report['point_accuracy'].items():
                print(f"  {key}: {val:.2f} pixels")
        
        print("\n" + "="*60 + "\n")
