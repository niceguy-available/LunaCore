"""
Feature extraction and matching module
Handles SIFT, ORB, SURF features and lunar-specific landmarks
"""

import cv2
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


class FeatureExtractor:
    """Extract scale and rotation invariant features"""
    
    def __init__(self, method='sift'):
        """
        Initialize feature extractor
        
        Args:
            method: 'sift', 'orb', or 'akaze'
        """
        self.method = method
        self._init_detector()
    
    def _init_detector(self):
        """Initialize appropriate feature detector"""
        if self.method == 'sift':
            self.detector = cv2.SIFT_create()
        elif self.method == 'orb':
            self.detector = cv2.ORB_create(nfeatures=10000)
        elif self.method == 'akaze':
            self.detector = cv2.AKAZE_create()
        else:
            self.detector = cv2.SIFT_create()
    
    def detect_and_compute(self, image):
        """Detect keypoints and compute descriptors"""
        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        return keypoints, descriptors
    
    def detect_craters(self, image, threshold=0.7):
        """Detect lunar craters as landmarks using Hough circles"""
        # Blur image for better circle detection
        blurred = cv2.GaussianBlur(image, (9, 9), 2)
        
        # Detect circles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=50
        )
        
        if circles is None:
            return [], []
        
        circles = np.uint16(np.around(circles))
        keypoints = []
        descriptors = []
        
        for circle in circles[0, :]:
            x, y, r = circle
            # Create keypoint at crater center
            kp = cv2.KeyPoint(x, y, r)
            keypoints.append(kp)
            
            # Create simple descriptor (crater properties)
            region = image[max(0, y-r):min(image.shape[0], y+r),
                          max(0, x-r):min(image.shape[1], x+r)]
            if region.size > 0:
                descriptor = self._compute_crater_descriptor(region)
                descriptors.append(descriptor)
        
        return keypoints, np.array(descriptors) if descriptors else None
    
    def _compute_crater_descriptor(self, region):
        """Compute descriptor for crater region"""
        if region.size == 0:
            return np.zeros(8)
        
        # Extract features from crater region
        features = np.array([
            np.mean(region),
            np.std(region),
            np.min(region),
            np.max(region),
            np.percentile(region, 25),
            np.percentile(region, 50),
            np.percentile(region, 75),
            region.size
        ])
        
        return features
    
    def extract_multiscale_features(self, image, scales=[0.5, 1.0, 1.5, 2.0]):
        """Extract features at multiple scales"""
        all_keypoints = []
        all_descriptors = []
        
        h, w = image.shape
        
        for scale in scales:
            # Resize image
            scaled = cv2.resize(image, (int(w * scale), int(h * scale)))
            
            # Detect features
            kp, desc = self.detect_and_compute(scaled)
            
            # Scale keypoints back to original size
            for k in kp:
                k.pt = (k.pt[0] / scale, k.pt[1] / scale)
                k.size = k.size / scale
                all_keypoints.append(k)
            
            if desc is not None:
                all_descriptors.extend(desc)
        
        return all_keypoints, np.array(all_descriptors) if all_descriptors else None


class FeatureMatcher:
    """Match features between images"""
    
    def __init__(self, method='bf'):
        """
        Initialize feature matcher
        
        Args:
            method: 'bf' (brute force) or 'flann'
        """
        self.method = method
    
    def match_features(self, desc1, desc2, k=2):
        """Match descriptors between two images using Lowe's ratio test"""
        if desc1 is None or desc2 is None:
            return []
        
        if self.method == 'flann':
            matcher = self._create_flann_matcher(desc1)
        else:
            matcher = self._create_bf_matcher(desc1)
        
        try:
            matches = matcher.knnMatch(desc1, desc2, k=k)
        except:
            return []
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def _create_bf_matcher(self, desc1):
        """Create brute force matcher"""
        if len(desc1) > 0 and desc1.dtype == np.uint8:
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    def _create_flann_matcher(self, desc1):
        """Create FLANN matcher"""
        if len(desc1) > 0 and desc1.dtype == np.uint8:
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=10, key_size=20, multi_probe_level=2)
        else:
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        
        search_params = dict(checks=50)
        return cv2.FlannBasedMatcher(index_params, search_params)
    
    def match_keypoints(self, kp1, desc1, kp2, desc2, ratio=0.75):
        """Match keypoints between two images"""
        matches = self.match_features(desc1, desc2, k=2)
        
        # Convert matches to coordinate pairs
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        return matches, src_pts, dst_pts
    
    def filter_matches_ransac(self, kp1, desc1, kp2, desc2, threshold=5.0):
        """Match features and filter outliers using RANSAC"""
        matches, src_pts, dst_pts = self.match_keypoints(kp1, desc1, kp2, desc2)
        
        if len(matches) < 4:
            return matches, None, None
        
        # RANSAC to find homography
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, threshold)
        
        # Filter inliers
        inlier_matches = [matches[i] for i in range(len(matches)) if mask[i]]
        inlier_src = src_pts[mask.flatten() == 1]
        inlier_dst = dst_pts[mask.flatten() == 1]
        
        return inlier_matches, inlier_src, inlier_dst
    
    def visualize_matches(self, img1, kp1, img2, kp2, matches, max_matches=50):
        """Visualize matched features"""
        # Limit matches for visualization
        matches = matches[:max_matches]
        
        # Draw matches
        result = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, 
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        return result
    
    def visualize_inliers(self, img1, kp1, img2, kp2, inlier_matches):
        """Visualize inlier matches"""
        result = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return result


class LunarLandmarkDetector:
    """Detect and match lunar landmarks (craters, peaks, etc.)"""
    
    def __init__(self):
        self.crater_detector = FeatureExtractor('sift')
    
    def detect_landmarks(self, image):
        """Detect lunar landmarks"""
        # Method 1: Craters using Hough circles
        craters_kp, craters_desc = self.crater_detector.detect_craters(image)
        
        # Method 2: General features using SIFT
        sift_kp, sift_desc = self.crater_detector.detect_and_compute(image)
        
        # Combine detections
        combined_kp = craters_kp + sift_kp
        combined_desc = np.vstack([craters_desc, sift_desc]) if craters_desc is not None else sift_desc
        
        return combined_kp, combined_desc
    
    def match_landmarks(self, img1, img2):
        """Match landmarks between two lunar images"""
        kp1, desc1 = self.detect_landmarks(img1)
        kp2, desc2 = self.detect_landmarks(img2)
        
        matcher = FeatureMatcher('bf')
        matches, src_pts, dst_pts = matcher.match_keypoints(kp1, desc1, kp2, desc2)
        
        return kp1, kp2, matches, src_pts, dst_pts
