#!/usr/bin/env python3
"""Generate visually distinct SAR images with oil spills."""

import numpy as np
import cv2
import os

def generate_sar(output_path, seed, n_spills, size=512):
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Ocean background - bright, uniform speckle
    background = rng.gamma(shape=4, scale=3, size=(size, size)).astype(np.float32)

    # Add gentle wave texture
    yy, xx = np.mgrid[:size, :size]
    wave = 0.5 * np.sin(2 * np.pi * xx / 30) * np.cos(2 * np.pi * yy / 40)
    background += wave.astype(np.float32)

    # Normalize background to bright range [0.5, 0.9]
    background = (background - background.min()) / (background.max() - background.min() + 1e-8)
    background = background * 0.4 + 0.5  # Range: [0.5, 0.9]

    # Create oil spill masks
    oil_mask = np.zeros((size, size), dtype=bool)

    for _ in range(n_spills):
        cx = rng.integers(size//4, 3*size//4)
        cy = rng.integers(size//4, 3*size//4)
        rx = rng.integers(15, 60)
        ry = rng.integers(8, 25)
        angle = rng.integers(-60, 60)

        cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        rot_xx = (xx - cx) * cos_a + (yy - cy) * sin_a
        rot_yy = -(xx - cx) * sin_a + (yy - cy) * cos_a

        # Clean ellipse
        ellipse = (rot_xx / rx) ** 2 + (rot_yy / ry) ** 2 < 1

        # Add irregular edges
        noise = rng.normal(0, 0.2, (size, size))
        noise_smooth = cv2.GaussianBlur(noise, (11, 11), 3)
        irregular = (rot_xx / (rx * 1.15)) ** 2 + (rot_yy / (ry * 1.25)) ** 2 < (1 + noise_smooth * 0.25)
        oil_mask |= (ellipse | irregular)

    # Oil makes pixels very dark (70-85% reduction)
    image = background.copy()
    image[oil_mask] *= rng.uniform(0.15, 0.30)

    # Add some bright ship wakes
    for _ in range(2):
        y0, x0 = rng.integers(50, size-50, 2)
        angle = rng.uniform(-20, 20)
        for t in range(rng.integers(30, 80)):
            px = int(x0 + t * np.cos(np.radians(angle)))
            py = int(y0 + t * np.sin(np.radians(angle)))
            if 0 <= px < size and 0 <= py < size:
                image[py, min(px, size-1)] *= rng.uniform(1.3, 2.0)

    # Save
    image = np.clip(image, 0, 1)
    img_uint8 = (image * 255).astype(np.uint8)
    cv2.imwrite(output_path, img_uint8)

    # Ground truth mask
    mask_path = output_path.replace('.png', '_mask.png')
    cv2.imwrite(mask_path, (oil_mask.astype(np.uint8) * 255))

    coverage = oil_mask.sum() / (size * size) * 100
    print(f"  {output_path}: {n_spills} spills, {coverage:.1f}% coverage")
    return output_path

print("Generating distinct SAR images...")
generate_sar("data/sar_images/mumbai_coast_sar.png", seed=42, n_spills=2)
generate_sar("data/sar_images/persian_gulf_sar.png", seed=123, n_spills=3)
generate_sar("data/sar_images/gulf_of_mexico_sar.png", seed=789, n_spills=1)
print("Done!")
