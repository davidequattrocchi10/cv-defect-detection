"""Convert the MVTec AD leather dataset into YOLO format.

Produces the following output structure under data/processed/:
    images/train/  — resized defect-free training images
    images/val/    — resized defective images for validation
    images/test/   — resized defective images for final evaluation
    labels/train/  — empty .txt files (background-only YOLO annotations)
    labels/val/    — YOLO bounding-box annotations derived from GT masks
    labels/test/   — YOLO bounding-box annotations derived from GT masks
    dataset.yaml   — YOLOv8 training configuration
"""

import time
from pathlib import Path

import cv2
import numpy as np
import shutil
from sklearn.model_selection import train_test_split

DATA_ROOT = Path("data/raw/leather")
OUTPUT_DIR = Path("data/processed")
IMAGE_SIZE = 640
VAL_SPLIT = 0.2
RANDOM_SEED = 42

# Map human-readable text (leather defects -> data/raw/leather) into integer numbers
CLASS_MAP = {
    "color": 0,
    "cut": 1,
    "fold": 2,
    "glue": 3,
    "poke": 4,
}


def mask_to_yolo_bbox(mask_path: Path, image_size: int = IMAGE_SIZE) -> list[str]:
    """Convert a binary defect mask to YOLO bounding-box annotation strings.

    YOLO format: each line is "<class_id> <cx> <cy> <w> <h>" where all four
    spatial values are normalised to [0, 1] relative to image width and height.
    cx/cy are the bounding-box centre coordinates; w/h are the box dimensions.

    One annotation line is produced per connected contour found in the mask.
    A mask can contain multiple disconnected defect regions, each becoming its
    own bounding box with the class id determined by the parent defect folder.

    Args:
        mask_path: Path to the ground-truth binary mask (.png, values 0 or 255).
        image_size: The spatial size (pixels) of the resized output image.
            Used as denominator when normalising pixel coordinates.

    Returns:
        List of YOLO annotation strings, one per contour. Empty list if the
        mask contains no foreground pixels or no valid contours are found.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    # It looks at the mask image. If a pixel's brightness is below 127, it forces that pixel to 0. Otherwise, it forces it to 255 (pure white).
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    # Derive class id from the mask's parent folder name (e.g. ground_truth/cut/)
    defect_type = mask_path.parent.name
    class_id = CLASS_MAP.get(defect_type, 0)

    # Masks are sourced at original resolution; compute the scale factor so
    # bounding rectangles are correct after the image has been resized.
    orig_h, orig_w = mask.shape
    scale_x = image_size / orig_w
    scale_y = image_size / orig_h

    annotations: list[str] = []
    for contour in contours:
        # draws the smallest possible straight rectangle that completely encloses defect blob 
        px, py, pw, ph = cv2.boundingRect(contour)

        # Scale pixel coords to the output image size, then normalise to [0, 1]
        cx = (px + pw / 2) * scale_x / image_size
        cy = (py + ph / 2) * scale_y / image_size
        w = pw * scale_x / image_size
        h = ph * scale_y / image_size

        # Format that YOLO needs (text string) 
        annotations.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return annotations


def create_directory_structure(output_dir: Path) -> None:
    """Create the YOLO dataset folder hierarchy under output_dir.

    Args:
        output_dir: Root directory for the processed dataset (data/processed/).
    """
    folders = [
        output_dir / "images" / "train",
        output_dir / "images" / "val",
        output_dir / "images" / "test",
        output_dir / "labels" / "train",
        output_dir / "labels" / "val",
        output_dir / "labels" / "test",
    ]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        print(f"Created: {folder}")


def process_good_images(output_dir: Path) -> int:
    """
    This function grabs all the pictures of defect-free leather, resizes them to 640x640, 
    saves them in the new training folder, and creates a completely blank text file for each one.

    Each image gets a corresponding empty label file because YOLO treats images
    without annotation entries as negative (background-only) samples.

    Args:
        output_dir: Root directory of the processed dataset.

    Returns:
        Number of images processed.
    """
    image_paths = sorted((DATA_ROOT / "train" / "good").glob("*.png"))
    images_out = output_dir / "images" / "train"
    labels_out = output_dir / "labels" / "train"

    for i, path in enumerate(image_paths):
        img = cv2.imread(str(path))
        img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        cv2.imwrite(str(images_out / path.name), img_resized)

        # In the labels_out folder, create a txt file and writes an empty string inside it.
        (labels_out / path.with_suffix(".txt").name).write_text("")

        # Progress tracker every 50 loops
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(image_paths)} training images")

    print(f"  Done — {len(image_paths)} training images processed")
    return len(image_paths)


def process_defective_images(output_dir: Path) -> tuple[int, int]:
    """Split defective test images into val/test sets and generate YOLO labels.

    Collects (image_path, mask_path) pairs for every defect category, then
    performs a stratified 80/20 split (by defect type) into val and test sets.

    Args:
        output_dir: Root directory of the processed dataset.

    Returns:
        Tuple of (val_count, test_count).
    """
    # MVTec stores the actual photos in one folder and the black-and-white defect masks in a totally different folder. 
    # This section finds them both and pairs them up (into "pairs" list).
    pairs: list[tuple[Path, Path, str]] = []
    for defect_type in CLASS_MAP:
        defect_dir = DATA_ROOT / "test" / defect_type
        for img_path in sorted(defect_dir.glob("*.png")):
            mask_path = DATA_ROOT / "ground_truth" / defect_type / f"{img_path.stem}_mask.png"
            if mask_path.exists():
                pairs.append((img_path, mask_path, defect_type))

    # Stratify by defect type to keep class balance across val and test
    labels = [p[2] for p in pairs]
    train_pairs, val_pairs = train_test_split(
        pairs,
        test_size=VAL_SPLIT,
        random_state=RANDOM_SEED,
        stratify=labels, # stratify -> allow that 80/20 ratio is maintained across every single defect type
    )

    def _write_split(split_pairs: list[tuple[Path, Path, str]], split: str) -> int:
        """
        Nested function.
        I need to resize images and generate labels twice (validation and test). 
        Instead of copying and pasting the exact same code twice, I wrote a nested function.
        """
        images_out = output_dir / "images" / split
        labels_out = output_dir / "labels" / split
        for img_path, mask_path, defect_type in split_pairs:
            prefixed_stem = f"{defect_type}_{img_path.stem}"
            img = cv2.imread(str(img_path))
            img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            cv2.imwrite(str(images_out / f"{prefixed_stem}.png"), img_resized)

            annotations = mask_to_yolo_bbox(mask_path)
            label_file = labels_out / f"{prefixed_stem}.txt"
            label_file.write_text("\n".join(annotations))
            
        return len(split_pairs)

    # train_test_split naming: train_pairs → test split, val_pairs → val split
    test_count = _write_split(train_pairs, "test")
    val_count = _write_split(val_pairs, "val")

    print(f"  Val images:  {val_count}")
    print(f"  Test images: {test_count}")
    return val_count, test_count


def create_dataset_yaml(output_dir: Path) -> None:
    """Write the YOLOv8 dataset configuration file.

    Args:
        output_dir: Root directory of the processed dataset.
    """
    names = list(CLASS_MAP.keys())
    yaml_content = (
        f"path: {output_dir}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"\n"
        f"nc: {len(CLASS_MAP)}\n"
        f"names: {names}\n"
    )
    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    print(f"  Written: {yaml_path}")


def main() -> None:
    """Run the full preprocessing pipeline and print a summary."""
    start = time.time()

    print("=== Step 1: Creating directory structure ===")
    create_directory_structure(OUTPUT_DIR)

    print("\n=== Step 2: Processing defect-free training images ===")
    train_count = process_good_images(OUTPUT_DIR)

    print("\n=== Step 3: Processing defective images (val + test split) ===")
    val_count, test_count = process_defective_images(OUTPUT_DIR)

    print("\n=== Step 4: Writing dataset.yaml ===")
    create_dataset_yaml(OUTPUT_DIR)

    elapsed = time.time() - start
    print(f"\n=== Done in {elapsed:.1f}s ===")
    print(f"  Train: {train_count} images")
    print(f"  Val:   {val_count} images")
    print(f"  Test:  {test_count} images")
    print(f"  Total: {train_count + val_count + test_count} images")


if __name__ == "__main__":
    main()
