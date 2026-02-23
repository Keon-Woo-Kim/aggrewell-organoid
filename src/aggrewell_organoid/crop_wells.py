"""Crop wells from AggreWell plate images using RANSAC grid fitting."""

import json
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from scipy.spatial.distance import cdist

PADDING = 10
INLIER_THRESHOLD = 50
RANSAC_ITERS = 3000


def generate_grid(x0, y0, dx, dy, theta, rows, cols):
    """Generate grid centers from parameters."""
    centers = []
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    for row in range(rows):
        for col in range(cols):
            x = x0 + col * dx * cos_t - row * dy * sin_t
            y = y0 + col * dx * sin_t + row * dy * cos_t
            centers.append((x, y))
    return np.array(centers)


def fit_grid_from_3_points(pts, assignments):
    """Fit grid params from 3 detected points with known (row, col) assignments."""
    r = np.array([a[0] for a in assignments])
    c = np.array([a[1] for a in assignments])
    xs, ys = pts[:, 0], pts[:, 1]

    A_x = np.column_stack([np.ones(3), c])
    A_y = np.column_stack([np.ones(3), r])
    try:
        x0, dx = np.linalg.lstsq(A_x, xs, rcond=None)[0]
        y0, dy = np.linalg.lstsq(A_y, ys, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None

    if dx < 100 or dx > 800 or dy < 100 or dy > 800:
        return None
    if abs(dx - dy) > 100:
        return None

    x_res = xs - (x0 + c * dx)
    if np.std(r) > 0:
        theta = -np.mean(x_res / (r * dy + 1e-10))
        theta = np.clip(theta, -0.1, 0.1)
    else:
        theta = 0.0

    return (x0, y0, dx, dy, theta)


def ransac_grid_fit(det_centers, img_w, img_h, rows, cols):
    """RANSAC to fit grid. Penalizes grids with wells outside image."""
    n = len(det_centers)
    if n < 3:
        return None, 0

    best_params = None
    best_score = -999

    # Assign approximate (row, col) by position clustering
    ys = det_centers[:, 1]
    y_sorted_vals = np.sort(ys)
    row_breaks = np.where(np.diff(y_sorted_vals) > 80)[0]
    if len(row_breaks) < rows - 1:
        row_breaks = np.where(np.diff(y_sorted_vals) > 50)[0]
    row_thresholds = [(y_sorted_vals[i] + y_sorted_vals[i + 1]) / 2 for i in row_breaks]
    row_thresholds = [-np.inf] + row_thresholds + [np.inf]

    det_rc = []
    for i in range(n):
        row = sum(1 for t in row_thresholds[1:] if det_centers[i, 1] > t)
        det_rc.append(row)

    det_assignments = [None] * n
    for row in range(max(det_rc) + 1):
        row_indices = [i for i in range(n) if det_rc[i] == row]
        row_pts = [(det_centers[i, 0], i) for i in row_indices]
        row_pts.sort()
        for col, (_, idx) in enumerate(row_pts):
            det_assignments[idx] = (row, col)

    # RANSAC
    indices = list(range(n))
    for _ in range(RANSAC_ITERS):
        sample = np.random.choice(indices, 3, replace=False)
        pts = det_centers[sample]
        assigns = [det_assignments[s] for s in sample]

        if any(a is None for a in assigns):
            continue
        if len(set(assigns)) < 3:
            continue

        params = fit_grid_from_3_points(pts, assigns)
        if params is None:
            continue

        grid = generate_grid(*params, rows=rows, cols=cols)

        dists = cdist(det_centers, grid).min(axis=1)
        inliers = int(np.sum(dists < INLIER_THRESHOLD))

        margin = 100
        outside = int(np.sum(
            (grid[:, 0] < margin) | (grid[:, 0] > img_w - margin) |
            (grid[:, 1] < margin) | (grid[:, 1] > img_h - margin)
        ))

        score = inliers - outside * 3

        if score > best_score:
            best_score = score
            best_params = params

    # Refine with all inliers
    if best_params is not None:
        grid = generate_grid(*best_params, rows=rows, cols=cols)
        dists = cdist(det_centers, grid)
        nearest_idx = dists.argmin(axis=1)
        nearest_dist = dists.min(axis=1)
        inlier_mask = nearest_dist < INLIER_THRESHOLD

        inlier_pts = det_centers[inlier_mask]
        inlier_grid_rc = []
        for idx in nearest_idx[inlier_mask]:
            r, c = divmod(idx, cols)
            inlier_grid_rc.append((r, c))

        if len(inlier_pts) >= 3:
            rows_arr = np.array([rc[0] for rc in inlier_grid_rc])
            cols_arr = np.array([rc[1] for rc in inlier_grid_rc])
            xs = inlier_pts[:, 0]
            ys = inlier_pts[:, 1]

            A = np.column_stack([np.ones(len(xs)), cols_arr, -rows_arr])
            res_x = np.linalg.lstsq(A, xs, rcond=None)[0]
            A2 = np.column_stack([np.ones(len(ys)), rows_arr, cols_arr])
            res_y = np.linalg.lstsq(A2, ys, rcond=None)[0]

            x0, dx, dy_theta = res_x
            y0, dy, dx_theta = res_y

            theta1 = dy_theta / dy if abs(dy) > 1 else 0
            theta2 = dx_theta / dx if abs(dx) > 1 else 0
            theta = (theta1 + theta2) / 2

            best_params = (x0, y0, dx, dy, theta)

    # Count final inliers
    if best_params is not None:
        grid = generate_grid(*best_params, rows=rows, cols=cols)
        dists = cdist(det_centers, grid).min(axis=1)
        n_inliers = int(np.sum(dists < INLIER_THRESHOLD))
    else:
        n_inliers = 0

    return best_params, n_inliers


def detect_wells_grid(model, img_path, img, h_img, w_img, rows, cols):
    """Detect wells using YOLOv8 + RANSAC grid fitting."""
    results = model(str(img_path), imgsz=1280, conf=0.005, iou=0.5, verbose=False)
    r = results[0]
    if len(r.boxes) == 0:
        return None

    xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    order = np.argsort(-confs)[:30]
    candidates = xyxy[order]

    # Loose size filter (70%-130% of median)
    widths = candidates[:, 2] - candidates[:, 0]
    heights = candidates[:, 3] - candidates[:, 1]
    med_w, med_h = np.median(widths), np.median(heights)
    size_ok = ((widths > 0.7 * med_w) & (widths < 1.3 * med_w) &
               (heights > 0.7 * med_h) & (heights < 1.3 * med_h))
    candidates = candidates[size_ok]
    widths = widths[size_ok]
    heights = heights[size_ok]

    det_centers = np.column_stack([
        (candidates[:, 0] + candidates[:, 2]) / 2,
        (candidates[:, 1] + candidates[:, 3]) / 2,
    ])

    params, n_inliers = ransac_grid_fit(det_centers, w_img, h_img, rows, cols)
    if params is None:
        return None

    grid_centers = generate_grid(*params, rows=rows, cols=cols)

    # Well size from inlier detections
    dists_all = cdist(det_centers, grid_centers).min(axis=1)
    inlier_mask = dists_all < INLIER_THRESHOLD
    reasonable = inlier_mask & (widths > 0.85 * med_w) & (widths < 1.15 * med_w)
    well_w = np.median(widths[reasonable]) if np.any(reasonable) else med_w
    well_h = np.median(heights[reasonable]) if np.any(reasonable) else med_h

    wells = []
    for i, (gx, gy) in enumerate(grid_centers):
        row, col = divmod(i, cols)
        x1 = int(gx - well_w / 2)
        y1 = int(gy - well_h / 2)
        x2 = int(gx + well_w / 2)
        y2 = int(gy + well_h / 2)
        wells.append((row, col, x1, y1, x2, y2))

    return wells


def run(image_dir, model_path, crop_dir, preview_dir, rows, cols):
    """Crop wells from all images. Returns crop_dir path."""
    image_dir = Path(image_dir)
    crop_dir = Path(crop_dir)
    preview_dir = Path(preview_dir)
    crop_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.tiff", "*.tif"):
        images.extend(image_dir.glob(ext))
    images = sorted(images)
    print(f"Processing {len(images)} images\n")

    total_crops = 0
    crop_coords = {}

    for img_path in images:
        fname = img_path.name
        img = cv2.imread(str(img_path))
        h_img, w_img = img.shape[:2]

        wells = detect_wells_grid(model, img_path, img, h_img, w_img, rows, cols)
        if wells is None:
            print(f"  {fname}: grid-fit failed, skipping")
            continue

        vis = img.copy()
        img_coords = {}
        for row, col, x1, y1, x2, y2 in wells:
            x1p = max(0, x1 - PADDING)
            y1p = max(0, y1 - PADDING)
            x2p = min(w_img, x2 + PADDING)
            y2p = min(h_img, y2 + PADDING)

            crop = img[y1p:y2p, x1p:x2p]
            crop_name = f"{img_path.stem}_r{row}c{col}"
            cv2.imwrite(str(crop_dir / f"{crop_name}.jpg"), crop)
            total_crops += 1

            img_coords[f"r{row}c{col}"] = {"x1p": x1p, "y1p": y1p, "x2p": x2p, "y2p": y2p}

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"r{row}c{col}", (x1 + 5, y1 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        crop_coords[img_path.stem] = img_coords
        cv2.imwrite(str(preview_dir / f"{img_path.stem}_grid.jpg"), vis)
        print(f"  {fname}: {len(wells)} wells cropped")

    # Save crop coordinates
    with open(crop_dir / "crop_coordinates.json", "w") as f:
        json.dump(crop_coords, f)

    print(f"\n{'='*50}")
    print(f"Total crops: {total_crops}")
    print(f"Crops: {crop_dir}")
    print(f"Preview: {preview_dir}")

    return crop_dir
