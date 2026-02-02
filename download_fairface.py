#!/usr/bin/env python3
"""
Download balanced samples from FairFace dataset on Hugging Face.
Downloads 30 images per age+gender combination (9 ages × 2 genders = 540 total).
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# Configuration
SAMPLES_PER_GROUP = 30
OUTPUT_DIR = Path(__file__).parent / "fairface"
GROUND_TRUTH_FILE = OUTPUT_DIR / "ground_truth.json"

# FairFace age and gender mappings
AGE_LABELS = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]
GENDER_LABELS = ["Male", "Female"]

# Normalize age label for output (convert "more than 70" to "70+")
def normalize_age(age_label: str) -> str:
    if age_label == "more than 70":
        return "70+"
    return age_label


def main():
    print("Loading FairFace dataset from Hugging Face...")
    # Use HuggingFaceM4/FairFace which includes actual images
    # Config '1.25' = larger margin around faces for better context
    dataset = load_dataset("HuggingFaceM4/FairFace", "1.25", split="train")

    print(f"Total images in dataset: {len(dataset)}")

    # Debug: print available columns/keys
    print(f"Dataset columns: {dataset.column_names}")
    print(f"First item keys: {list(dataset[0].keys())}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Track counts per group
    group_counts = defaultdict(int)
    ground_truth = []
    image_counter = 1

    total_target = SAMPLES_PER_GROUP * len(AGE_LABELS) * len(GENDER_LABELS)
    print("\nDownloading balanced samples...")
    print(f"Target: {SAMPLES_PER_GROUP} images per age+gender group")
    print(f"Total groups: {len(AGE_LABELS)} ages × {len(GENDER_LABELS)} genders = {len(AGE_LABELS) * len(GENDER_LABELS)}")
    print(f"Expected total: {total_target} images\n")

    # Progress bar for downloaded images
    pbar = tqdm(total=total_target, desc="Downloading", unit="img")

    # Iterate through dataset and collect balanced samples
    scanned = 0
    for item in dataset:
        scanned += 1

        # Dataset returns integer indices, convert to string labels
        age_idx = item["age"]
        gender_idx = item["gender"]
        age_label = AGE_LABELS[age_idx]
        gender_label = GENDER_LABELS[gender_idx]

        group_key = (age_label, gender_label)

        # Skip if we already have enough samples for this group
        if group_counts[group_key] >= SAMPLES_PER_GROUP:
            continue

        # Save the image
        filename = f"fairface_{image_counter:04d}.jpg"
        image_path = OUTPUT_DIR / filename

        # Get the image and save it
        image: Image.Image = item["image"]
        image.save(image_path, "JPEG", quality=95)

        # Add to ground truth
        ground_truth.append({
            "id": f"fairface_{image_counter:04d}",
            "filename": filename,
            "age": normalize_age(age_label),
            "gender": gender_label
        })

        group_counts[group_key] += 1
        image_counter += 1

        # Update progress bar
        total_downloaded = sum(group_counts.values())
        pbar.update(1)
        pbar.set_postfix({
            "scanned": scanned,
            "saved": total_downloaded,
            "last": f"{normalize_age(age_label)}/{gender_label}"
        })

        # Check if we have enough samples for all groups
        if all(group_counts[(age, gender)] >= SAMPLES_PER_GROUP
               for age in AGE_LABELS
               for gender in GENDER_LABELS):
            break

    pbar.close()

    # Save ground truth JSON
    with open(GROUND_TRUTH_FILE, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print("Download complete!")
    print(f"{'='*50}")
    print(f"Total images downloaded: {len(ground_truth)}")
    print(f"Ground truth saved to: {GROUND_TRUTH_FILE}")

    # Print summary by group
    print("\nSamples per group:")
    for age in AGE_LABELS:
        for gender in GENDER_LABELS:
            count = group_counts[(age, gender)]
            normalized_age = normalize_age(age)
            print(f"  {normalized_age:>10} + {gender:<6}: {count}")


if __name__ == "__main__":
    main()
