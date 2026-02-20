"""Run organoid segmentation on well crops and overlay results on original images."""

import json
import csv
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict
import openpyxl

AREA_SCALE = 1000


def run(image_dir, crop_dir, model_path, output_dir, data_name, exclude_well):
    """Run organoid inference on all crops. Exports CSV, XLSX, and annotated images."""
    image_dir = Path(image_dir)
    crop_dir = Path(crop_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))

    # Load crop coordinates
    with open(crop_dir / "crop_coordinates.json") as f:
        crop_coords = json.load(f)

    # Group crops by source image
    crops = sorted(crop_dir.glob("*.jpg"))
    print(f"Found {len(crops)} well crops")

    by_image = defaultdict(list)
    for crop_path in crops:
        stem = crop_path.stem
        parts = stem.rsplit("_r", 1)
        img_name = parts[0]
        rc = parts[1]
        row = int(rc.split("c")[0])
        col = int(rc.split("c")[1])
        by_image[img_name].append((row, col, crop_path))

    print(f"Source images: {len(by_image)}")

    total_organoids = 0
    all_records = []

    for img_name in sorted(by_image.keys()):
        wells = by_image[img_name]
        wells.sort(key=lambda x: (x[0], x[1]))

        # Load original plate image (find by stem, any supported extension)
        orig_path = None
        for ext in (".jpg", ".jpeg", ".png", ".tiff", ".tif"):
            candidate = image_dir / f"{img_name}{ext}"
            if candidate.exists():
                orig_path = candidate
                break
        orig_img = cv2.imread(str(orig_path))
        vis = orig_img.copy()
        coords = crop_coords.get(img_name, {})

        img_organoids = 0

        for row, col, crop_path in wells:
            well_key = f"r{row}c{col}"
            crop = cv2.imread(str(crop_path))
            crop_h, crop_w = crop.shape[:2]
            crop_area_px = crop_h * crop_w
            is_excluded = exclude_well is not None and (row, col) == exclude_well

            if well_key not in coords:
                continue
            c = coords[well_key]
            x1p, y1p, x2p, y2p = c["x1p"], c["y1p"], c["x2p"], c["y2p"]

            # Draw well boundary (blue) on the crop
            cv2.rectangle(crop, (0, 0), (crop_w - 1, crop_h - 1), (255, 0, 0), 2)

            if is_excluded:
                pass
            else:
                # Run inference on native crop (no resize/distortion)
                results = model(str(crop_path), imgsz=480, conf=0.5, verbose=False)
                r = results[0]
                n_det = len(r.boxes) if r.boxes is not None else 0

                if n_det > 0 and r.masks is not None:
                    for i in range(n_det):
                        mask = r.masks.data[i].cpu().numpy()
                        mask_resized = cv2.resize(mask, (crop_w, crop_h))
                        mask_bool = mask_resized > 0.5

                        # Calculate organoid area
                        mask_px = int(np.sum(mask_bool))
                        area = mask_px / crop_area_px * AREA_SCALE

                        # Semi-transparent green overlay
                        overlay = crop.copy()
                        overlay[mask_bool] = [0, 255, 0]
                        crop = cv2.addWeighted(crop, 0.7, overlay, 0.3, 0)

                        # Draw contour
                        contours, _ = cv2.findContours(
                            mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        cv2.drawContours(crop, contours, -1, (0, 255, 0), 2)

                        # Area label
                        bx1, by1 = r.boxes.xyxy[i][:2].cpu().numpy().astype(int)
                        cv2.putText(crop, f"{area:.1f}", (bx1, max(by1 - 17, 40)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 3)

                        # Record for export
                        conf = float(r.boxes.conf[i].cpu().numpy())
                        all_records.append({
                            "image": img_name,
                            "row": row,
                            "col": col,
                            "well": well_key,
                            "organoid_idx": i,
                            "area": round(area, 2),
                            "area_px": mask_px,
                            "crop_area_px": crop_area_px,
                            "confidence": round(conf, 4),
                        })

                    img_organoids += n_det

            # Paste annotated crop back onto original image
            vis[y1p:y2p, x1p:x2p] = crop

        cv2.putText(vis, f"{img_name}: {img_organoids} organoids",
                    (10, vis.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imwrite(str(output_dir / f"{img_name}_organoids.jpg"), vis)

        total_organoids += img_organoids
        print(f"  {img_name}: {img_organoids} organoids")

    # --- Export data ---
    raw_fields = ["image", "row", "col", "well", "organoid_idx", "area", "area_px", "crop_area_px", "confidence"]

    avg_by_image = defaultdict(list)
    for rec in all_records:
        avg_by_image[rec["image"]].append(rec["area"])

    avg_records = []
    for name in sorted(avg_by_image.keys()):
        areas = avg_by_image[name]
        avg_records.append({
            "image": name,
            "n_organoids": len(areas),
            "mean_area": round(np.mean(areas), 2),
            "std_area": round(np.std(areas), 2),
            "min_area": round(np.min(areas), 2),
            "max_area": round(np.max(areas), 2),
            "median_area": round(np.median(areas), 2),
        })
    avg_fields = ["image", "n_organoids", "mean_area", "std_area", "min_area", "max_area", "median_area"]

    # CSV - averaged
    csv_avg_path = output_dir / f"{data_name}.csv"
    with open(csv_avg_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=avg_fields)
        writer.writeheader()
        writer.writerows(avg_records)

    # CSV - raw
    csv_raw_path = output_dir / f"{data_name}_raw.csv"
    with open(csv_raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows(all_records)

    # XLSX
    xlsx_path = output_dir / f"{data_name}.xlsx"
    wb = openpyxl.Workbook()

    ws_avg = wb.active
    ws_avg.title = "data"
    ws_avg.append(avg_fields)
    for rec in avg_records:
        ws_avg.append([rec[f] for f in avg_fields])

    ws_raw = wb.create_sheet("data_raw")
    ws_raw.append(raw_fields)
    for rec in all_records:
        ws_raw.append([rec[f] for f in raw_fields])

    wb.save(xlsx_path)

    exclude_str = f"r{exclude_well[0]}c{exclude_well[1]}" if exclude_well else "none"
    print(f"\n{'='*50}")
    print(f"Total organoids: {total_organoids} across {len(by_image)} images")
    print(f"Exported: {len(all_records)} organoids ({exclude_str} excluded)")
    print(f"CSV (averaged): {csv_avg_path}")
    print(f"CSV (raw): {csv_raw_path}")
    print(f"XLSX: {xlsx_path}")
    print(f"Images: {output_dir}")
