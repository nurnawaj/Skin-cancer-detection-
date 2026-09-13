"""
prepare_data.py
----------------
Validates the HAM10000 dataset, then produces reproducible
train / validation / test CSV splits (80% / 10% / 10%).

Leakage avoidance:
HAM10000 contains multiple images of the same physical lesion
(grouped by `lesion_id`). If a lesion's images were split across
train and test, the model could "memorize" that lesion and get an
artificially inflated test score. To avoid this we split at the
LESION level (all images of a lesion stay in the same split), then
stratify by that lesion's diagnosis so class proportions remain
balanced across splits.

If `lesion_id` is not present in the metadata, we fall back to a
plain stratified split on individual images and print a warning
documenting this limitation.

Usage:
    python src/prepare_data.py
"""

import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config as cfg
from src.dataset import build_class_mapping
from src.utils import seed_everything, save_json


def validate_metadata(metadata: pd.DataFrame) -> None:
    required_columns = {cfg.IMAGE_ID_COLUMN, cfg.LABEL_COLUMN}
    missing = required_columns - set(metadata.columns)
    if missing:
        raise ValueError(
            f"Metadata CSV is missing required column(s): {missing}. "
            f"Expected at least: {required_columns}"
        )
    if metadata.empty:
        raise ValueError("Metadata CSV is empty. Cannot prepare an empty dataset.")


def check_images_exist(metadata: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose image file cannot be found, warning the user."""
    existing_rows = []
    missing_count = 0
    for _, row in metadata.iterrows():
        image_id = str(row[cfg.IMAGE_ID_COLUMN])
        candidate = os.path.join(cfg.IMAGES_PATH, image_id + cfg.IMAGE_EXTENSION)
        found = os.path.exists(candidate)
        if not found:
            for root, _, files in os.walk(cfg.IMAGES_PATH):
                if image_id + cfg.IMAGE_EXTENSION in files:
                    found = True
                    break
        if found:
            existing_rows.append(row)
        else:
            missing_count += 1

    if missing_count > 0:
        print(
            f"Warning: {missing_count} image(s) referenced in metadata were not "
            f"found in '{cfg.IMAGES_PATH}' and will be excluded."
        )
    return pd.DataFrame(existing_rows).reset_index(drop=True)


def grouped_stratified_split(metadata: pd.DataFrame):
    """
    Split at the lesion (group) level when possible, otherwise fall back
    to a plain per-image stratified split.
    """
    has_groups = cfg.LESION_ID_COLUMN in metadata.columns

    if has_groups:
        print(f"'{cfg.LESION_ID_COLUMN}' column found — using grouped, leakage-free splitting.")
        # One row per lesion, using that lesion's diagnosis label for stratification.
        lesion_labels = (
            metadata.groupby(cfg.LESION_ID_COLUMN)[cfg.LABEL_COLUMN]
            .agg(lambda s: s.value_counts().idxmax())  # majority label per lesion
            .reset_index()
        )

        train_lesions, temp_lesions = train_test_split(
            lesion_labels,
            test_size=(cfg.VAL_SPLIT + cfg.TEST_SPLIT),
            stratify=lesion_labels[cfg.LABEL_COLUMN],
            random_state=cfg.RANDOM_SEED,
        )
        relative_test_size = cfg.TEST_SPLIT / (cfg.VAL_SPLIT + cfg.TEST_SPLIT)
        val_lesions, test_lesions = train_test_split(
            temp_lesions,
            test_size=relative_test_size,
            stratify=temp_lesions[cfg.LABEL_COLUMN],
            random_state=cfg.RANDOM_SEED,
        )

        train_ids = set(train_lesions[cfg.LESION_ID_COLUMN])
        val_ids = set(val_lesions[cfg.LESION_ID_COLUMN])
        test_ids = set(test_lesions[cfg.LESION_ID_COLUMN])

        train_df = metadata[metadata[cfg.LESION_ID_COLUMN].isin(train_ids)].reset_index(drop=True)
        val_df = metadata[metadata[cfg.LESION_ID_COLUMN].isin(val_ids)].reset_index(drop=True)
        test_df = metadata[metadata[cfg.LESION_ID_COLUMN].isin(test_ids)].reset_index(drop=True)
    else:
        print(
            f"Warning: no '{cfg.LESION_ID_COLUMN}' column found in metadata. "
            f"Falling back to per-image stratified splitting. This may allow "
            f"images of the same lesion to appear in different splits."
        )
        train_df, temp_df = train_test_split(
            metadata,
            test_size=(cfg.VAL_SPLIT + cfg.TEST_SPLIT),
            stratify=metadata[cfg.LABEL_COLUMN],
            random_state=cfg.RANDOM_SEED,
        )
        relative_test_size = cfg.TEST_SPLIT / (cfg.VAL_SPLIT + cfg.TEST_SPLIT)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=relative_test_size,
            stratify=temp_df[cfg.LABEL_COLUMN],
            random_state=cfg.RANDOM_SEED,
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df


def main():
    seed_everything(cfg.RANDOM_SEED)

    if not os.path.exists(cfg.METADATA_PATH):
        raise FileNotFoundError(
            f"Metadata file not found at '{cfg.METADATA_PATH}'.\n"
            f"Please download HAM10000 and place 'HAM10000_metadata.csv' "
            f"and the image folder under '{cfg.DATASET_PATH}' as described in the README."
        )

    print(f"Loading metadata from: {cfg.METADATA_PATH}")
    metadata = pd.read_csv(cfg.METADATA_PATH)
    validate_metadata(metadata)

    print("Checking that referenced images exist on disk...")
    metadata = check_images_exist(metadata)
    if metadata.empty:
        raise ValueError(
            "No valid (image, label) pairs remain after checking the images "
            "directory. Verify your dataset placement."
        )

    class_to_idx = build_class_mapping(metadata)
    print(f"Detected {len(class_to_idx)} classes: {class_to_idx}")

    train_df, val_df, test_df = grouped_stratified_split(metadata)

    train_df.to_csv(cfg.TRAIN_CSV, index=False)
    val_df.to_csv(cfg.VAL_CSV, index=False)
    test_df.to_csv(cfg.TEST_CSV, index=False)

    save_json(class_to_idx, os.path.join(cfg.DATASET_PATH, "class_to_idx.json"))

    print("\nSplit summary:")
    print(f"  Train: {len(train_df)} images")
    print(f"  Val:   {len(val_df)} images")
    print(f"  Test:  {len(test_df)} images")
    print(f"\nClass distribution (train):\n{train_df[cfg.LABEL_COLUMN].value_counts()}")
    print(f"\nSaved: {cfg.TRAIN_CSV}\nSaved: {cfg.VAL_CSV}\nSaved: {cfg.TEST_CSV}")
    print("\nData preparation complete.")


if __name__ == "__main__":
    main()
