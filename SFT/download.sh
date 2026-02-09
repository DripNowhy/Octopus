#!/bin/bash

# Download script for LLaVA-CoT-100k dataset
# Downloads dataset from Hugging Face to SFT/dataset directory

set -e

# Create dataset directory if not exists
DATASET_DIR="$(dirname "$0")/dataset"
mkdir -p "$DATASET_DIR"

echo "Downloading LLaVA-CoT-100k dataset to $DATASET_DIR..."

# Install required packages if not installed
pip install -q huggingface_hub datasets

# Download LLaVA-CoT-100k dataset
python3 -c "
from datasets import load_dataset
import os

dataset = load_dataset('Xkev/LLaVA-CoT-100k')
dataset.save_to_disk(os.path.join('$DATASET_DIR', 'LLaVA-CoT-100k'))
print('LLaVA-CoT-100k dataset downloaded successfully!')
print(f'Train split size: {len(dataset[\"train\"])}')
"

echo "Dataset download completed!"
echo "Dataset location: $DATASET_DIR/LLaVA-CoT-100k"

# Check if image.zip.part-* files exist and extract them
if ls "$DATASET_DIR"/image.zip.part-* 1> /dev/null 2>&1; then
    echo "Found image.zip.part-* files, merging and extracting..."
    cat "$DATASET_DIR"/image.zip.part-* > "$DATASET_DIR"/image.zip
    unzip -o "$DATASET_DIR"/image.zip -d "$DATASET_DIR"/images
    echo "Images extracted to $DATASET_DIR/images"
else
    echo "No image.zip.part-* files found. Skipping image extraction."
fi
