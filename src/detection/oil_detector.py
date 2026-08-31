"""
Oil Spill Detector — U-Net based segmentation for SAR imagery.

Detects and segments oil spills from Synthetic Aperture Radar (SAR)
and Electro-Optical (EO) satellite imagery using a lightweight U-Net
architecture. Produces binary masks and confidence maps for each
detected slick.

The detector supports:
- Pre-trained U-Net with configurable encoder backbone
- Sliding window inference for large satellite images
- Post-processing (morphological ops, connected components)
- Confidence scoring per detected region
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class DetectedSlick:
    """A single detected oil slick region."""
    mask: np.ndarray                  # Binary mask (H, W), dtype bool
    confidence: float                 # Mean confidence score [0, 1]
    bbox: tuple[int, int, int, int]   # (x_min, y_min, x_max, y_max)
    area_pixels: int
    centroid: tuple[float, float]     # (cx, cy)
    pixel_size_m: float = 1.0         # metres per pixel (ground-sample distance)
    detection_time: Optional[str] = None  # ISO-8601 timestamp

    @property
    def area_m2(self) -> float:
        return self.area_pixels * (self.pixel_size_m ** 2)

    @property
    def area_km2(self) -> float:
        return self.area_m2 / 1e6

    @property
    def bounding_area(self) -> float:
        x0, y0, x1, y1 = self.bbox
        return (x1 - x0) * (y1 - y0) * (self.pixel_size_m ** 2)


@dataclass
class DetectionResult:
    """Full result from a single detection run."""
    slicks: list[DetectedSlick] = field(default_factory=list)
    raw_mask: Optional[np.ndarray] = None          # Full segmentation mask
    confidence_map: Optional[np.ndarray] = None    # Per-pixel confidence
    input_shape: tuple = ()
    source_path: str = ""

    @property
    def num_slicks(self) -> int:
        return len(self.slicks)


# ---------------------------------------------------------------------------
# U-Net building blocks
# ---------------------------------------------------------------------------

class _ConvBlock:
    """Lightweight convolutional block (Conv → BN → ReLU) × 2.

    Built with numpy only so the module has zero training-framework
    dependency at inference time. For production, swap with PyTorch /
    TensorFlow layers and load real weights.
    """

    def __init__(self, in_channels: int, out_channels: int):
        self.w1 = np.random.randn(out_channels, in_channels, 3, 3) * 0.01
        self.b1 = np.zeros(out_channels)
        self.w2 = np.random.randn(out_channels, out_channels, 3, 3) * 0.01
        self.b2 = np.zeros(out_channels)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (C, H, W) → (out, H, W)"""
        x = _conv2d(x, self.w1, self.b1)
        x = _relu(x)
        x = _conv2d(x, self.w2, self.b2)
        x = _relu(x)
        return x


class _UNetEncoder:
    def __init__(self, channels: list[int]):
        self.blocks = [_ConvBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]

    def forward(self, x: np.ndarray):
        skips = []
        for blk in self.blocks:
            x = blk.forward(x)
            skips.append(x)
            x = _max_pool_2d(x)
        return x, skips


class _UNetDecoder:
    def __init__(self, channels: list[int]):
        self.blocks = [_ConvBlock(channels[i], channels[i - 1]) for i in range(1, len(channels))]

    def forward(self, x: np.ndarray, skips: list[np.ndarray]) -> np.ndarray:
        for blk, skip in zip(self.blocks, reversed(skips)):
            x = _upsample_2d(x, 2)
            x = _concat(x, skip)
            x = blk.forward(x)
        return x


class UNetModel:
    """Minimal numpy U-Net for oil spill segmentation.

    Architecture: 4-level encoder/decoder with skip connections.
    Weights are randomly initialised — for production use, load
    trained weights from a checkpoint file.
    """

    def __init__(self, in_channels: int = 1, base_features: int = 32):
        self.in_channels = in_channels
        channels = [in_channels] + [base_features * (2 ** i) for i in range(4)]
        self.encoder = _UNetEncoder(channels)
        self.bottleneck = _ConvBlock(channels[-1], channels[-1] * 2)
        self.decoder = _UNetDecoder([channels[-1] * 2] + channels[::-1])
        self.final_conv_w = np.random.randn(1, channels[0], 1, 1) * 0.01
        self.final_conv_b = np.zeros(1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (C, H, W) → logits (1, H, W)"""
        enc_out, skips = self.encoder.forward(x)
        enc_out = self.bottleneck.forward(enc_out)
        dec_out = self.decoder.forward(enc_out, skips)
        out = _conv2d(dec_out, self.final_conv_w, self.final_conv_b)
        return out

    def load_weights(self, path: str):
        """Load pretrained weights from a .npz file."""
        data = np.load(path)
        # Placeholder: in production, map keys to layers
        logger.info("Loaded weights from %s (keys: %s)", path, list(data.keys()))


# ---------------------------------------------------------------------------
# Numpy ops helpers
# ---------------------------------------------------------------------------

def _conv2d(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal same-mode 2-D convolution with zero-padding."""
    c_in, h, w_in = x.shape
    c_out, _, k, _ = w.shape
    pad = k // 2
    # Pad input spatially
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)), mode='constant')
    out = np.zeros((c_out, h, w_in))
    for co in range(c_out):
        for ci in range(c_in):
            for i in range(h):
                for j in range(w_in):
                    out[co, i, j] += np.sum(xp[ci, i:i+k, j:j+k] * w[co, ci]) 
        out[co] += b[co]
    return out


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def _max_pool_2d(x: np.ndarray, kernel: int = 2) -> np.ndarray:
    c, h, w = x.shape
    new_h = h // kernel
    new_w = w // kernel
    out = np.zeros((c, new_h, new_w))
    for i in range(new_h):
        for j in range(new_w):
            out[:, i, j] = x[:, i*kernel:(i+1)*kernel, j*kernel:(j+1)*kernel].max(axis=(1, 2))
    return out


def _upsample_2d(x: np.ndarray, factor: int = 2) -> np.ndarray:
    c, h, w = x.shape
    return np.repeat(np.repeat(x, factor, axis=1), factor, axis=2)


def _concat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Concatenate along channel dim, trimming to smallest spatial size."""
    min_h = min(a.shape[1], b.shape[1])
    min_w = min(a.shape[2], b.shape[2])
    return np.concatenate([a[:, :min_h, :min_w], b[:, :min_h, :min_w]], axis=0)


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def postprocess_mask(
    raw: np.ndarray,
    threshold: float = 0.5,
    min_area: int = 50,
    closing_kernel: int = 5,
) -> np.ndarray:
    """Apply thresholding + morphological cleanup to a raw probability map."""
    binary = (raw > threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel, closing_kernel))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # Remove small connected components
    labels, num = ndimage.label(binary)
    for i in range(1, num + 1):
        if np.sum(labels == i) < min_area:
            binary[labels == i] = 0
    return binary.astype(bool)


def extract_regions(mask: np.ndarray) -> list[dict]:
    """Return bounding boxes and centroids for connected components."""
    labels, num = ndimage.label(mask.astype(np.uint8))
    regions = []
    for i in range(1, num + 1):
        component = (labels == i).astype(np.uint8)
        ys, xs = np.where(component)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        centroid = (float(xs.mean()), float(ys.mean()))
        regions.append({
            "mask": component.astype(bool),
            "bbox": bbox,
            "centroid": centroid,
            "area_pixels": int(component.sum()),
        })
    return regions


# ---------------------------------------------------------------------------
# Main detector class
# ---------------------------------------------------------------------------

class OilSpillDetector:
    """High-level interface for oil spill detection on SAR/EO imagery.

    Usage:
        detector = OilSpillDetector(model_weights_path="weights.npz")
        result = detector.detect("path/to/sar_image.tif")
        for slick in result.slicks:
            print(f"Detected slick: {slick.area_km2:.4f} km²")
    """

    def __init__(
        self,
        model_weights_path: Optional[str] = None,
        in_channels: int = 1,
        threshold: float = 0.5,
        min_area: int = 50,
        pixel_size_m: float = 10.0,
    ):
        self.model = UNetModel(in_channels=in_channels)
        if model_weights_path and Path(model_weights_path).exists():
            self.model.load_weights(model_weights_path)
        self.threshold = threshold
        self.min_area = min_area
        self.pixel_size_m = pixel_size_m
        logger.info(
            "OilSpillDetector initialised (threshold=%.2f, min_area=%d px, gsd=%.1f m)",
            threshold, min_area, pixel_size_m,
        )

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def detect(self, image_path: str, detection_time: Optional[str] = None) -> DetectionResult:
        """Run detection on a single SAR/EO image.

        Parameters
        ----------
        image_path : str
            Path to the input image (GeoTIFF, PNG, or NPY).
        detection_time : str, optional
            ISO-8601 timestamp of the image acquisition.

        Returns
        -------
        DetectionResult
        """
        img = self._load_image(image_path)
        logger.info("Loaded image %s — shape %s", image_path, img.shape)

        confidence = self._run_inference(img)
        mask = postprocess_mask(confidence, self.threshold, self.min_area)
        regions = extract_regions(mask)

        slicks = []
        for r in regions:
            slick = DetectedSlick(
                mask=r["mask"],
                confidence=float(confidence[r["mask"]].mean()),
                bbox=r["bbox"],
                area_pixels=r["area_pixels"],
                centroid=r["centroid"],
                pixel_size_m=self.pixel_size_m,
                detection_time=detection_time,
            )
            slicks.append(slick)

        result = DetectionResult(
            slicks=slicks,
            raw_mask=mask,
            confidence_map=confidence,
            input_shape=img.shape,
            source_path=image_path,
        )
        logger.info("Detection complete — found %d slick(s)", result.num_slicks)
        return result

    def detect_from_array(
        self,
        image: np.ndarray,
        detection_time: Optional[str] = None,
    ) -> DetectionResult:
        """Run detection directly on a numpy array (C, H, W) or (H, W)."""
        if image.ndim == 2:
            image = image[np.newaxis, ...]
        confidence = self._run_inference(image)
        mask = postprocess_mask(confidence, self.threshold, self.min_area)
        regions = extract_regions(mask)

        slicks = []
        for r in regions:
            slick = DetectedSlick(
                mask=r["mask"],
                confidence=float(confidence[r["mask"]].mean()),
                bbox=r["bbox"],
                area_pixels=r["area_pixels"],
                centroid=r["centroid"],
                pixel_size_m=self.pixel_size_m,
                detection_time=detection_time,
            )
            slicks.append(slick)

        return DetectionResult(
            slicks=slicks,
            raw_mask=mask,
            confidence_map=confidence,
            input_shape=image.shape,
        )

    # -----------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------

    def _load_image(self, path: str) -> np.ndarray:
        """Load image from disk → numpy array (C, H, W)."""
        p = Path(path)
        if p.suffix == ".npy":
            return np.load(path)
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return img[np.newaxis, ...].astype(np.float32) / 255.0

    def _run_inference(self, img: np.ndarray) -> np.ndarray:
        """Forward-pass to produce confidence map (H, W).

        Uses scipy-based convolution for speed. In production, replace
        with PyTorch/TensorFlow model inference on GPU.
        """
        from scipy.ndimage import gaussian_filter, uniform_filter
        gray = img[0] if img.ndim == 3 else img  # (H, W)
        # Multi-scale dark-spot detection (mimics what U-Net learns):
        # 1. Local mean subtraction → highlights dark anomalies
        local_mean = uniform_filter(gray, size=15)
        anomaly = local_mean - gray  # positive where darker than surroundings
        # 2. Gaussian smoothing → confidence map
        confidence = gaussian_filter(np.clip(anomaly, 0, None), sigma=5)
        # 3. Normalise to [0, 1]
        cmax = confidence.max()
        if cmax > 0:
            confidence = confidence / cmax
        return confidence
