import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib import cm

# ============================================================
# 1. LOAD REAL INFRARED IMAGES (grayscale IR)
# ============================================================
camA = cv2.imread("camA_ir.png", cv2.IMREAD_ANYDEPTH)
camC = cv2.imread("camC_ir.png", cv2.IMREAD_ANYDEPTH)

# Normalize if 16-bit IR
def normalize(img):
    img = img.astype("float32")
    img -= img.min()
    img /= (img.max() + 1e-6)
    return img

camA = normalize(camA)
camC = normalize(camC)

# Load top-down venue image
topdown = cv2.imread("topdown.png")
h_top, w_top = topdown.shape[:2]


# ============================================================
# 2. HUMAN ISOLATION IN REAL IR
#    (thermal thresholding + morphological shaping)
# ============================================================
def ir_people_mask(img):
    """
    Segment people from real IR by using heat signatures.
    Higher intensities = hotter = people.
    """

    # Adaptive threshold (robust for varying lighting)
    thr = cv2.adaptiveThreshold(
        (img * 255).astype("uint8"),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51,
        -5
    )

    # Clean noise
    kernel = np.ones((5,5), np.uint8)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)

    return thr


maskA = ir_people_mask(camA)
maskC = ir_people_mask(camC)


# ============================================================
# 3. MAKE TRUE DENSITY MAPS (Gaussian kernel per pixel)
# ============================================================
def make_density(mask):
    """
    Convert binary mask to density map with spatial smoothing.
    """
    density = mask.astype("float32") / 255.0
    # Spread the density (simulates real model output)
    density = gaussian_filter(density, sigma=8)
    return density

densityA = make_density(maskA)
densityC = make_density(maskC)


# ============================================================
# 4. LOAD HOMOGRAPHIES + WARP TO TOP-DOWN COORDINATES
# ============================================================
H_A = np.load("H_A.npy")
H_C = np.load("H_C.npy")

warpedA = cv2.warpPerspective(densityA, H_A, (w_top, h_top))
warpedC = cv2.warpPerspective(densityC, H_C, (w_top, h_top))


# ============================================================
# 5. CREATE MASTER DENSITY MAP
# ============================================================
master_density = warpedA + warpedC
master_density = np.clip(master_density, 0, None)

np.save("master_density.npy", master_density)


# ============================================================
# 6. GENERATE VISUAL HEATMAPS
# ============================================================
def heat_overlay(base_img, density):
    d = density / (density.max() + 1e-6)
    heat = (cm.inferno(d)[:, :, :3] * 255).astype("uint8")
    heat = cv2.cvtColor(heat, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(base_img, 0.55, heat, 0.75, 0)


# Individual camera heatmaps
cv2.imwrite("heat_camA.png", heat_overlay(topdown, warpedA))
cv2.imwrite("heat_camC.png", heat_overlay(topdown, warpedC))

# Final master heatmap
cv2.imwrite("heat_master.png", heat_overlay(topdown, master_density))

print("✓ Generated: heat_camA.png, heat_camC.png, heat_master.png")