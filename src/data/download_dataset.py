# Dataset: MVTec AD — Leather category
# Source:  https://www.mvtec.com/company/research/datasets/mvtec-ad
# License: CC BY-NC-SA 4.0 — free for non-commercial use
#
# Manual download instructions:
#   1. Visit the URL above and accept the license agreement.
#   2. Download leather.tar.xz.
#   3. Extract it so that the folder structure is:
#        data/raw/leather/
#          train/good/
#          test/good/
#          test/cut/
#          test/fold/
#          test/glue/
#          test/poke/
#          test/color/
#          ground_truth/

import os
from pathlib import Path


EXPECTED_FOLDERS = [
    "train/good",
    "test/good",
    "test/cut",
    "test/fold",
    "test/glue",
    "test/poke",
    "test/color",
    "ground_truth",
]


def inspect_dataset(base_path: str = "data/raw/leather") -> dict[str, int]:
    """Walk the dataset directory and print a summary of .png images per subfolder.

    Args:
        base_path: Path to the extracted leather dataset root.

    Returns:
        Dictionary mapping each subfolder name to its image count.

    Raises:
        FileNotFoundError: If base_path does not exist.
    """
    root = Path(base_path)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{root.resolve()}'.\n"
            "Download leather.tar.xz from https://www.mvtec.com/company/research/datasets/mvtec-ad "
            f"and extract it so that '{root}/' exists."
        )

    counts: dict[str, int] = {}
    for dirpath, _, filenames in os.walk(root):
        png_count = sum(1 for f in filenames if f.lower().endswith(".png"))
        if png_count > 0:
            label = str(Path(dirpath).relative_to(root))
            counts[label] = png_count

    col_w = max((len(k) for k in counts), default=8)
    header = f"{'Folder':<{col_w}}  {'Images':>6}"
    separator = "-" * len(header)

    print(f"\nDataset root: {root.resolve()}")
    print(header)
    print(separator)
    for folder, count in sorted(counts.items()):
        print(f"{folder:<{col_w}}  {count:>6}")
    print(separator)
    print(f"{'Total':<{col_w}}  {sum(counts.values()):>6}")

    return counts


def validate_dataset_structure(base_path: str = "data/raw/leather") -> bool:
    """Check that all expected subfolders are present in the dataset.

    Args:
        base_path: Path to the extracted leather dataset root.

    Returns:
        True if every expected folder exists, False otherwise.
    """
    root = Path(base_path)
    print(f"\nValidating dataset structure at '{root.resolve()}' ...")

    all_present = True
    for folder in EXPECTED_FOLDERS:
        full_path = root / folder
        if full_path.exists():
            print(f"  ✅  {folder}")
        else:
            print(f"  ❌  {folder}  (missing)")
            all_present = False

    if all_present:
        print("\nAll expected folders are present.")
    else:
        print("\nSome folders are missing — check the extraction above.")

    return all_present


if __name__ == "__main__":
    inspect_dataset()
    validate_dataset_structure()
