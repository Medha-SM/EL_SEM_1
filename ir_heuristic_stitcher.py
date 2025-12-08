"""
IR-ONLY DENSITY STITCHER

This version assumes:
  • ALL input images are IR (colorized thermal)
  • No RGB detection
  • No RGB segmentation options
  • Uses ir_to_density() exclusively
  • Homographies or placements may be provided

Usage:
python final_stitcher_ir_only.py --images camA.png camC.png --Hs H_A.npy H_C.npy
python final_stitcher_ir_only.py --images camA.png camC.png --placements 50,200 800,200
"""

import argparse
import cv2
import numpy as np
from matplotlib import cm
from PIL import Image
import os
import sys

# ---------- IR → Density ----------
def ir_to_density(img_bgr):
    """Convert colorized IR image to density map."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-9)

    thr = cv2.adaptiveThreshold(
        (gray * 255).astype(np.uint8), 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 51, -5
    )

    kernel = np.ones((5,5), np.uint8)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)

    density = cv2.GaussianBlur(thr.astype(np.float32)/255.0, (0,0), sigmaX=8, sigmaY=8)
    return density

# ---------- Heatmap Coloring ----------
def heatmap_to_color(density, cmap_name='inferno'):
    if density.max() > 0:
        norm = (density - density.min()) / (density.max() - density.min())
    else:
        norm = density
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(norm)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

# ---------- Stitcher ----------
def stitch_densities(densities, H_files=None, placements=None, topdown_image=None):
    # Determine canvas size
    if topdown_image is not None:
        canvas_h, canvas_w = topdown_image.shape[:2]
    else:
        if placements:
            canvas_w = max(x + dens.shape[1] for (x, y), dens in zip(placements, densities)) + 50
            canvas_h = max(y + dens.shape[0] for (x, y), dens in zip(placements, densities)) + 50
        else:
            maxh = max(d.shape[0] for d in densities)
            sumw = sum(d.shape[1] for d in densities)
            canvas_h = maxh + 100
            canvas_w = int(sumw * 0.75) + 200

    master = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    # Warp using homographies if provided
    if H_files:
        for dens, Hf in zip(densities, H_files):
            H = np.load(Hf)
            warped = cv2.warpPerspective(dens, H, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR)
            master += warped
    else:
        for dens, (x_off, y_off) in zip(densities, placements):
            h,w = dens.shape
            x1, y1 = int(x_off), int(y_off)
            x2, y2 = x1 + w, y1 + h
            master[y1:y2, x1:x2] += dens

    return master

# ---------- MAIN ----------
def main():
    parser = argparse.ArgumentParser(description="IR-only multi-camera heatmap stitcher.")
    parser.add_argument("--images", nargs="+", required=True, help="Input IR camera images.")
    parser.add_argument("--topdown", default=None, help="Optional background image.")
    parser.add_argument("--Hs", nargs="*", default=None, help="Homography .npy files.")
    parser.add_argument("--placements", nargs="*", default=None, help="Manual placements x,y.")
    parser.add_argument("--out-prefix", default="master_ir", help="Output prefix.")
    args = parser.parse_args()

    densities = []
    for img_path in args.images:
        img = cv2.imread(img_path)
        if img is None:
            print("ERROR loading:", img_path)
            sys.exit(1)
        dens = ir_to_density(img)
        densities.append(dens)

    # Load topdown if provided
    topdown_img = cv2.imread(args.topdown) if args.topdown else None

    # Parse placements
    placements = None
    if args.placements:
        placements = [tuple(map(int, p.split(","))) for p in args.placements]

    # Stitch
    master = stitch_densities(
        densities,
        H_files=args.Hs,
        placements=placements,
        topdown_image=topdown_img
    )

    np.save(f"{args.out_prefix}_density.npy", master.astype(np.float32))

    # Visualization
    heat = heatmap_to_color(master)

    if topdown_img is not None:
        base = cv2.resize(topdown_img, (heat.shape[1], heat.shape[0]))
        overlay = cv2.addWeighted(base, 0.6, heat, 0.6, 0)
        cv2.imwrite(f"{args.out_prefix}_heatmap.png", overlay)
    else:
        cv2.imwrite(f"{args.out_prefix}_heatmap.png", heat)

    print("Saved:", f"{args.out_prefix}_heatmap.png", f"{args.out_prefix}_density.npy")

if __name__ == "__main__":
    main()
