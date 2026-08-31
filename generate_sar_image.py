#!/usr/bin/env python3
"""Generate realistic synthetic SAR images with oil spills for demo upload."""

import numpy as np
import cv2
import os

def generate_sar_with_oil_spill(
    output_path="data/sar_images/sample_sar_oilspill.png",
    size=512,
    seed=42,
    n_spills=3,
):
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1. Ocean background — gamma-distributed speckle (characteristic of SAR)
    background = rng.gamma(shape=2, scale=3, size=(size, size)).astype(np.float32)

    # 2. Add ocean surface texture (wave patterns)
    yy, xx = np.mgrid[:size, :size]
    wave1 = 0.3 * np.sin(2 * np.pi * xx / 40 + 0.5 * yy / 30)
    wave2 = 0.2 * np.sin(2 * np.pi * yy / 25 - 0.3 * xx / 35)
    wave3 = 0.15 * np.cos(2 * np.pi * (xx + yy) / 50)
    background += (wave1 + wave2 + wave3).astype(np.float32)

    # 3. Add range-dependent brightness gradient (typical of SAR)
    gradient = np.linspace(0.8, 1.2, size).reshape(-1, 1) * np.ones((1, size))
    background *= gradient

    # 4. Create oil spill patches (dark regions — oil dampens capillary waves)
    oil_mask = np.zeros((size, size), dtype=bool)

    # Unique spill shapes per seed so each image looks different
    spill_params = [
        {"cx": 100 + rng.integers(0, 100), "cy": 100 + rng.integers(0, 100),
         "rx": 30 + rng.integers(0, 60), "ry": 10 + rng.integers(0, 20),
         "angle": rng.integers(-45, 45)},
        {"cx": 250 + rng.integers(0, 100), "cy": 80 + rng.integers(0, 120),
         "rx": 20 + rng.integers(0, 50), "ry": 8 + rng.integers(0, 15),
         "angle": rng.integers(-60, 60)},
        {"cx": 150 + rng.integers(0, 150), "cy": 250 + rng.integers(0, 100),
         "rx": 25 + rng.integers(0, 55), "ry": 10 + rng.integers(0, 20),
         "angle": rng.integers(-30, 30)},
    ]

    for i, sp in enumerate(spill_params[:n_spills]):
        # Rotated ellipse mask
        cos_a = np.cos(np.radians(sp["angle"]))
        sin_a = np.sin(np.radians(sp["angle"]))
        rx, ry = sp["rx"], sp["ry"]
        rot_xx = (xx - sp["cx"]) * cos_a + (yy - sp["cy"]) * sin_a
        rot_yy = -(xx - sp["cx"]) * sin_a + (yy - sp["cy"]) * cos_a
        ellipse = (rot_xx / rx) ** 2 + (rot_yy / ry) ** 2 < 1

        # Add irregular edges (fractal-like)
        noise = rng.normal(0, 0.3, (size, size))
        noise_smooth = cv2.GaussianBlur(noise, (15, 15), 3)
        irregular = ellipse | ((rot_xx / (rx * 1.1)) ** 2 + (rot_yy / (ry * 1.3)) ** 2 < 1 + noise_smooth * 0.3)

        oil_mask |= irregular

    # 5. Apply oil dampening (reduce backscatter by 70-85%)
    dampening = np.ones((size, size), dtype=np.float32)
    dampening[oil_mask] = rng.uniform(0.12, 0.30, oil_mask.sum())

    # Add some gradient within the oil (thicker in center, thinner at edges)
    for sp in spill_params[:n_spills]:
        dist = np.sqrt(((xx - sp["cx"]) / sp["rx"]) ** 2 + ((yy - sp["cy"]) / sp["ry"]) ** 2)
        edge_fade = np.clip(1.0 - dist, 0.3, 1.0)
        dampening *= np.where(oil_mask, edge_fade, 1.0)

    # 6. Combine
    image = background * dampening

    # 7. Add some ship wake lines (thin bright lines)
    n_wakes = 2
    for _ in range(n_wakes):
        y0 = rng.integers(50, size - 50)
        x0 = rng.integers(50, size - 50)
        angle = rng.uniform(-30, 30)
        length = rng.integers(40, 100)
        for t in range(length):
            px = int(x0 + t * np.cos(np.radians(angle)))
            py = int(y0 + t * np.sin(np.radians(angle)))
            if 0 <= px < size and 0 <= py < size:
                image[py, px] *= rng.uniform(1.5, 2.5)

    # 8. Normalize to 0-255 and save
    image = np.clip(image, 0, None)
    image = (image / image.max() * 255).astype(np.uint8)

    cv2.imwrite(output_path, image)

    # Also save as NPY for direct array loading
    npy_path = output_path.replace(".png", ".npy")
    np.save(npy_path, image.astype(np.float32) / 255.0)

    print(f"SAR image saved: {output_path} ({size}x{size})")
    print(f"NPY array saved: {npy_path}")
    print(f"Oil spill coverage: {oil_mask.sum() / (size*size) * 100:.1f}%")

    # Save a version with ground truth mask for validation
    mask_path = output_path.replace(".png", "_mask.png")
    cv2.imwrite(mask_path, (oil_mask.astype(np.uint8) * 255))
    print(f"Ground truth mask: {mask_path}")

    return output_path


if __name__ == "__main__":
    # Generate multiple samples at different locations
    samples = [
        {"output_path": "data/sar_images/mumbai_coast_sar.png", "seed": 42, "n_spills": 2, "size": 512},
        {"output_path": "data/sar_images/persian_gulf_sar.png", "seed": 123, "n_spills": 3, "size": 512},
        {"output_path": "data/sar_images/gulf_of_mexico_sar.png", "seed": 456, "n_spills": 1, "size": 512},
    ]
    for s in samples:
        print(f"\nGenerating: {s['output_path']}")
        generate_sar_with_oil_spill(**s)
