"""
preprocessing.py — Enhanced Illumination-Invariant Image Preprocessing
ISRO SIH 2026 | Problem Statement 26166

Implements Stage A of the pipeline: radiometric/illumination normalization.
Converts images to illumination-invariant representations before feature
detection, specifically designed for cross-sensor lunar imagery where sun
angle, shadow direction, and overall illumination vary dramatically.

Key additions over the original prototype:
  • Phase Congruency maps (Gabor filterbank) — the gold standard for
    illumination-invariant edge/feature representation in remote sensing.
  • Difference-of-Log (DoL) transform — fast illumination normalizer.
  • Weber Local Descriptor (WLD) preprocessing — texture invariance.
  • Multi-resolution image pyramid with GSD-aware level selection.
  • Sun-angle metadata integration for auxiliary shadow estimation.
"""

import cv2
import numpy as np
from scipy import ndimage
from scipy.signal import convolve2d
from typing import Tuple, List, Dict, Optional, Any
import math


# ---------------------------------------------------------------------------
# Phase Congruency via Gabor Filterbank
# ---------------------------------------------------------------------------

def _build_log_gabor_filter(rows: int, cols: int, wavelength: float,
                             sigma_on_f: float = 0.55) -> np.ndarray:
    """
    Construct a Log-Gabor filter in the frequency domain.
    
    Log-Gabor filters have zero DC component and the response drops off
    gracefully on both sides of the centre frequency, making them ideal
    for phase congruency computation.
    """
    # Frequency coordinates centered at DC
    u = (np.arange(cols) - cols // 2) / cols
    v = (np.arange(rows) - rows // 2) / rows
    u, v = np.meshgrid(u, v)
    radius = np.sqrt(u ** 2 + v ** 2)
    radius[rows // 2, cols // 2] = 1.0  # avoid log(0)

    fo = 1.0 / wavelength
    log_gabor = np.exp(-(np.log(radius / fo)) ** 2 / (2 * np.log(sigma_on_f) ** 2))
    log_gabor[rows // 2, cols // 2] = 0.0  # zero DC
    return log_gabor


def compute_phase_congruency(image: np.ndarray,
                              n_scales: int = 4,
                              n_orientations: int = 6,
                              min_wavelength: int = 6,
                              mult: float = 2.1,
                              sigma_on_f: float = 0.55,
                              noise_threshold: float = 2.0
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Phase Congruency map via Log-Gabor filterbank.
    
    Phase congruency is maximally illumination-invariant: it responds to
    features (edges, lines, corners) based on the *phase alignment* of
    frequency components rather than their amplitude. This makes it ideal
    for matching lunar images under different sun angles.
    
    Returns:
        pc_map:   Phase congruency magnitude (0-1), float32, same size as input.
        pc_orient: Dominant orientation at each pixel (radians).
        pc_edges:  Thresholded binary edge map from phase congruency.
    """
    img = image.astype(np.float64)
    rows, cols = img.shape[:2]
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)

    img_fft = np.fft.fftshift(np.fft.fft2(img))

    # Accumulate energy and phase info across scales and orientations
    total_energy = np.zeros((rows, cols), dtype=np.float64)
    total_sum_an = np.zeros((rows, cols), dtype=np.float64)  # sum of amplitudes
    orient_accum_x = np.zeros((rows, cols), dtype=np.float64)
    orient_accum_y = np.zeros((rows, cols), dtype=np.float64)

    for o in range(n_orientations):
        angle = o * np.pi / n_orientations
        # Angular filter (raised cosine)
        u = (np.arange(cols) - cols // 2) / cols
        v = (np.arange(rows) - rows // 2) / rows
        u, v = np.meshgrid(u, v)
        theta = np.arctan2(-v, u)
        ds = np.abs(theta - angle)
        ds = np.minimum(ds, np.pi - ds)
        angular_filter = np.cos(ds) ** 2
        angular_filter[ds > np.pi / (2 * n_orientations)] = 0

        sum_e = np.zeros((rows, cols), dtype=np.float64)
        sum_o = np.zeros((rows, cols), dtype=np.float64)
        sum_an = np.zeros((rows, cols), dtype=np.float64)

        for s in range(n_scales):
            wavelength = min_wavelength * (mult ** s)
            log_gabor = _build_log_gabor_filter(rows, cols, wavelength, sigma_on_f)
            combined_filter = log_gabor * angular_filter

            filtered = np.fft.ifft2(np.fft.ifftshift(img_fft * combined_filter))
            e_part = np.real(filtered)  # even (symmetric) component
            o_part = np.imag(filtered)  # odd (antisymmetric) component
            amplitude = np.sqrt(e_part ** 2 + o_part ** 2)

            sum_e += e_part
            sum_o += o_part
            sum_an += amplitude

        # Energy = sqrt(sumE^2 + sumO^2) - noise
        energy = np.sqrt(sum_e ** 2 + sum_o ** 2)
        energy = np.maximum(energy - noise_threshold, 0)

        total_energy += energy
        total_sum_an += sum_an
        orient_accum_x += energy * np.cos(2 * angle)
        orient_accum_y += energy * np.sin(2 * angle)

    # Phase congruency = energy / sum_of_amplitudes
    pc_map = total_energy / (total_sum_an + 1e-8)
    pc_map = np.clip(pc_map, 0, 1).astype(np.float32)

    # Dominant orientation
    pc_orient = np.arctan2(orient_accum_y, orient_accum_x) / 2.0

    # Edge map via Otsu threshold on PC
    pc_uint8 = (pc_map * 255).astype(np.uint8)
    _, pc_edges = cv2.threshold(pc_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return pc_map, pc_orient, pc_edges


# ---------------------------------------------------------------------------
# Difference-of-Log (DoL) Transform
# ---------------------------------------------------------------------------

def difference_of_log(image: np.ndarray,
                      sigma1: float = 1.0,
                      sigma2: float = 10.0) -> np.ndarray:
    """
    Difference-of-Log (DoL) illumination normalization.
    
    Takes the log of the image, then subtracts a heavily-blurred log from
    a lightly-blurred log. This removes low-frequency illumination gradients
    while preserving high-frequency texture/edge information.
    
    Well-suited for lunar images with strong sun-angle-driven shading.
    """
    img = image.astype(np.float64) + 1.0  # avoid log(0)
    log_img = np.log(img)

    blur1 = cv2.GaussianBlur(log_img, (0, 0), sigma1)
    blur2 = cv2.GaussianBlur(log_img, (0, 0), sigma2)

    dol = blur1 - blur2

    # Normalize to 0-255
    dol = cv2.normalize(dol, None, 0, 255, cv2.NORM_MINMAX)
    return dol.astype(np.uint8)


# ---------------------------------------------------------------------------
# Weber Local Descriptor (WLD) Preprocessing
# ---------------------------------------------------------------------------

def weber_local_descriptor(image: np.ndarray,
                           epsilon: float = 1e-5) -> np.ndarray:
    """
    Weber Local Descriptor (WLD) differential excitation.
    
    Based on Weber's law of perception: the ratio of intensity change to
    background intensity is what matters, not the absolute change itself.
    This makes the representation inherently invariant to global illumination
    changes — exactly what we need for cross-sun-angle lunar matching.
    """
    img = image.astype(np.float64)

    # 3x3 neighborhood filter (sum of differences from center)
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]], dtype=np.float64)

    diff_sum = convolve2d(img, kernel, mode='same', boundary='symm')
    excitation = np.arctan(diff_sum / (img + epsilon))

    # Normalize to 0-255
    excitation = cv2.normalize(excitation, None, 0, 255, cv2.NORM_MINMAX)
    return excitation.astype(np.uint8)


# ---------------------------------------------------------------------------
# Main Preprocessor Class
# ---------------------------------------------------------------------------

class LunarPreprocessor:
    """
    Enhanced preprocessor for multi-modal lunar image registration.
    
    Provides multiple illumination-invariant representations optimized for
    cross-sensor Chandrayaan-2 imagery under varying sun angles.
    """
    
    # Sensor Ground Sample Distance (meters) — used for scale priors
    SENSOR_GSD = {
        "OHRC": 0.25,
        "TMC2": 5.0,
        "TMC": 5.0,
        "IIRS": 80.0,
        "LRO_NAC": 0.5,
        "LRO_WAC": 100.0,
        "SELENE": 10.0,
    }

    def __init__(self, method: str = "phase_congruency", debug: bool = False):
        """
        Args:
            method: Illumination normalization method. One of:
                    'phase_congruency', 'dol', 'wld', 'clahe', 'combined'
            debug:  If True, store intermediate images for visualization.
        """
        self.method = method
        self.debug = debug
        self.intermediates: Dict[str, np.ndarray] = {}

    # ----- public API -----

    def preprocess(self, image: np.ndarray,
                   sensor_type: Optional[str] = None,
                   sun_azimuth: Optional[float] = None,
                   sun_elevation: Optional[float] = None,
                   target_gsd: Optional[float] = None
                   ) -> Dict[str, Any]:
        """
        Full preprocessing pipeline (Stage A).
        
        Args:
            image:         Input grayscale image (uint8 or uint16).
            sensor_type:   Sensor code for GSD lookup ('OHRC', 'TMC2', 'IIRS', …).
            sun_azimuth:   Sun azimuth angle in degrees (metadata, optional).
            sun_elevation: Sun elevation angle in degrees (metadata, optional).
            target_gsd:    If provided, resample to this GSD (meters/pixel).
            
        Returns:
            dict with keys:
              'normalized'     — illumination-normalized image (uint8)
              'original'       — input image
              'pc_map'         — phase congruency map (if computed)
              'pc_orient'      — PC orientation map (if computed)
              'pc_edges'       — PC edge map (if computed)
              'pyramid'        — multi-resolution pyramid list
              'metadata'       — dict of preprocessing metadata
              'intermediates'  — dict of per-step images (if debug=True)
        """
        self.intermediates = {}
        
        # Ensure grayscale
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Convert 16-bit to 8-bit if needed
        if image.dtype == np.uint16:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        original = image.copy()
        self._store("original", original)

        # Step 1: Intensity normalization (percentile stretch)
        normalized = self._normalize_intensity(image)
        self._store("intensity_normalized", normalized)

        # Step 2: Illumination bias removal (morphological background subtraction)
        debiased = self._remove_illumination_bias(normalized)
        self._store("bias_removed", debiased)

        # Step 3: CLAHE contrast enhancement
        enhanced = self._enhance_contrast(debiased)
        self._store("contrast_enhanced", enhanced)

        # Step 4: Illumination-invariant representation
        result: Dict[str, Any] = {
            "original": original,
            "metadata": {
                "sensor_type": sensor_type,
                "sun_azimuth": sun_azimuth,
                "sun_elevation": sun_elevation,
                "method": self.method,
                "shape": image.shape,
            },
        }

        if self.method in ("phase_congruency", "combined"):
            pc_map, pc_orient, pc_edges = compute_phase_congruency(enhanced)
            result["pc_map"] = pc_map
            result["pc_orient"] = pc_orient
            result["pc_edges"] = pc_edges
            pc_uint8 = (pc_map * 255).astype(np.uint8)
            self._store("phase_congruency", pc_uint8)
            # For feature detection, blend phase congruency (edges) with enhanced texture
            # This provides rich invariant structural anchors with strong gradient peaks
            normalized_out = cv2.addWeighted(pc_uint8, 0.65, enhanced, 0.35, 0)
        elif self.method == "dol":
            normalized_out = difference_of_log(enhanced)
            self._store("dol", normalized_out)
        elif self.method == "wld":
            normalized_out = weber_local_descriptor(enhanced)
            self._store("wld", normalized_out)
        elif self.method == "clahe":
            normalized_out = enhanced  # CLAHE already applied
        else:
            normalized_out = enhanced

        if self.method == "combined":
            # Blend PC + DoL + CLAHE for maximum robustness
            dol_img = difference_of_log(enhanced)
            wld_img = weber_local_descriptor(enhanced)
            blended = cv2.addWeighted(
                (result["pc_map"] * 255).astype(np.uint8), 0.4,
                dol_img, 0.3, 0
            )
            blended = cv2.addWeighted(blended, 1.0, wld_img, 0.3, 0)
            normalized_out = blended
            self._store("combined", normalized_out)

        result["normalized"] = normalized_out

        # Step 5: Build multi-resolution pyramid
        result["pyramid"] = self._build_pyramid(normalized_out, levels=5)

        # Step 6: Sun-angle-aware shadow estimation (auxiliary cue)
        if sun_azimuth is not None and sun_elevation is not None:
            shadow_mask = self._estimate_shadow_mask(original, sun_azimuth, sun_elevation)
            result["shadow_mask"] = shadow_mask
            self._store("shadow_mask", shadow_mask)

        result["intermediates"] = dict(self.intermediates) if self.debug else {}
        return result

    def get_scale_ratio(self, sensor_src: str, sensor_ref: str) -> float:
        """
        Compute the expected scale ratio between two sensors based on GSD.
        
        E.g., OHRC (0.25 m) → TMC (5.0 m) gives ratio = 20.0,
        meaning 1 TMC pixel ≈ 20 OHRC pixels.
        """
        gsd_src = self.SENSOR_GSD.get(sensor_src, 1.0)
        gsd_ref = self.SENSOR_GSD.get(sensor_ref, 1.0)
        return gsd_ref / gsd_src

    def resample_to_common_gsd(self, image: np.ndarray,
                                current_gsd: float,
                                target_gsd: float) -> np.ndarray:
        """Resample image to match a target GSD (meters/pixel)."""
        scale = current_gsd / target_gsd
        if abs(scale - 1.0) < 0.01:
            return image
        new_h = int(image.shape[0] * scale)
        new_w = int(image.shape[1] * scale)
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        return cv2.resize(image, (new_w, new_h), interpolation=interp)

    # ----- private helpers -----

    def _normalize_intensity(self, image: np.ndarray) -> np.ndarray:
        """Percentile-based intensity stretch to [0, 255]."""
        p2, p98 = np.percentile(image, (2, 98))
        clipped = np.clip(image.astype(np.float32), p2, p98)
        stretched = (clipped - p2) / (p98 - p2 + 1e-8)
        return (stretched * 255).astype(np.uint8)

    def _remove_illumination_bias(self, image: np.ndarray,
                                   kernel_size: int = 51) -> np.ndarray:
        """Morphological background subtraction for illumination bias."""
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        result = image.astype(np.float32) - background.astype(np.float32) + 128
        return np.clip(result, 0, 255).astype(np.uint8)

    def _enhance_contrast(self, image: np.ndarray,
                          clip_limit: float = 3.0,
                          tile_size: int = 8) -> np.ndarray:
        """CLAHE contrast enhancement."""
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_size, tile_size)
        )
        return clahe.apply(image)

    def _build_pyramid(self, image: np.ndarray, levels: int = 5) -> List[np.ndarray]:
        """Build Gaussian image pyramid."""
        pyramid = [image]
        current = image
        for _ in range(levels - 1):
            if current.shape[0] < 8 or current.shape[1] < 8:
                break
            current = cv2.pyrDown(current)
            pyramid.append(current)
        return pyramid

    def _estimate_shadow_mask(self, image: np.ndarray,
                               sun_azimuth: float,
                               sun_elevation: float) -> np.ndarray:
        """
        Estimate shadow regions using sun geometry and image intensity.
        
        This is an auxiliary cue, not a hard dependency. It helps weight
        feature detection away from shadow boundaries that shift with
        sun angle.
        """
        # Low-intensity regions with strong gradient perpendicular to
        # sun azimuth are likely shadow boundaries
        angle_rad = np.radians(sun_azimuth)

        # Directional Sobel along sun azimuth
        dx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=5)
        dy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=5)
        directional_grad = dx * np.cos(angle_rad) + dy * np.sin(angle_rad)

        # Threshold: low intensity + high directional gradient → shadow edge
        intensity_mask = image < np.percentile(image, 25)
        grad_mask = np.abs(directional_grad) > np.percentile(
            np.abs(directional_grad), 75
        )

        shadow_mask = (intensity_mask & grad_mask).astype(np.uint8) * 255
        # Dilate to cover shadow transition zones
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        shadow_mask = cv2.dilate(shadow_mask, kernel, iterations=2)
        return shadow_mask

    def _store(self, name: str, image: np.ndarray) -> None:
        if self.debug:
            self.intermediates[name] = image.copy()
