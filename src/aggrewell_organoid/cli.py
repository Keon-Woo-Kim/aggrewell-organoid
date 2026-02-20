"""CLI entry point for aggrewell-organoid."""

import argparse
from pathlib import Path

from . import crop_wells, infer_organoids


def parse_exclude(value):
    """Parse exclude well string like 'r3c0' into (row, col) tuple, or None."""
    if value.lower() == "none":
        return None
    value = value.strip().lower()
    if not value.startswith("r") or "c" not in value:
        raise argparse.ArgumentTypeError(f"Invalid well format: '{value}'. Use format 'r3c0' or 'none'.")
    try:
        row = int(value[1:value.index("c")])
        col = int(value[value.index("c") + 1:])
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid well format: '{value}'. Use format 'r3c0' or 'none'.")
    return (row, col)


def main():
    parser = argparse.ArgumentParser(
        prog="aggrewell-organoid",
        description="Automated organoid area measurement from AggreWell plate images.",
    )
    parser.add_argument(
        "folder",
        help="Path to folder containing plate images (.jpg/.png/.tiff)",
    )
    parser.add_argument(
        "--rows", type=int, default=4,
        help="Number of well rows (default: 4)",
    )
    parser.add_argument(
        "--cols", type=int, default=6,
        help="Number of well columns (default: 6)",
    )
    parser.add_argument(
        "--exclude", type=str, default="none",
        help="Well to exclude from analysis, e.g. 'r3c0' (default: none, includes all wells)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: results/ next to input folder)",
    )
    args = parser.parse_args()

    image_dir = Path(args.folder).resolve()
    if not image_dir.is_dir():
        parser.error(f"Folder not found: {image_dir}")

    exclude_well = parse_exclude(args.exclude)
    models_dir = Path(__file__).parent / "models"
    data_name = image_dir.name

    if args.output:
        output_dir = Path(args.output).resolve()
    else:
        output_dir = image_dir.parent / "results"

    crop_dir = output_dir / "crops"
    preview_dir = output_dir / "well_preview"

    print(f"Input:   {image_dir}")
    print(f"Output:  {output_dir}")
    print(f"Grid:    {args.rows} rows x {args.cols} cols")
    print(f"Exclude: {args.exclude}")
    print()

    # Step 1: Crop wells
    print("=" * 50)
    print("Step 1: Detecting wells and cropping")
    print("=" * 50)
    crop_wells.run(
        image_dir=image_dir,
        model_path=models_dir / "well_detector.pt",
        crop_dir=crop_dir,
        preview_dir=preview_dir,
        rows=args.rows,
        cols=args.cols,
    )

    # Step 2: Detect organoids
    print()
    print("=" * 50)
    print("Step 2: Detecting organoids")
    print("=" * 50)
    infer_organoids.run(
        image_dir=image_dir,
        crop_dir=crop_dir,
        model_path=models_dir / "organoid_detector.pt",
        output_dir=output_dir,
        data_name=data_name,
        exclude_well=exclude_well,
    )

    print()
    print("Done!")
