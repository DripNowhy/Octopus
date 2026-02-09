#!/bin/bash

# Data Construction Pipeline Script
# Steps: 1. sft_in_dis_sampling -> 2. sft_data_judge -> 3. sft_mixed_sampling -> 4. make_sft_format
# Runs on 8 GPUs in parallel where applicable

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================="
echo "SFT Data Construction Pipeline"
echo "========================================="

# Check if dataset exists
if [ ! -d "./dataset/LLaVA-CoT-100k" ]; then
    echo "Error: Dataset not found at ./dataset/LLaVA-CoT-100k"
    echo "Please run ./download.sh first to download the dataset."
    exit 1
fi

# Create output directories
mkdir -p ./dataset/sft_output
mkdir -p ./dataset/sft_judge_output

# =========================================
# Step 1: SFT In-Distribution Sampling (8 GPUs)
# =========================================
echo ""
echo "========================================="
echo "Step 1: SFT In-Distribution Sampling"
echo "========================================="

if [ -f "./dataset/sft_output/final_merged_results.jsonl" ]; then
    echo "Found existing sampling results. Skipping Step 1."
else
    echo "Running sft_in_dis_sampling.py on 8 GPUs..."
    python3 sft_in_dis_sampling.py
    echo "Step 1 completed!"
fi

# =========================================
# Step 2: Judge (8 GPUs)
# =========================================
echo ""
echo "========================================="
echo "Step 2: Judge"
echo "========================================="

if [ -f "./dataset/sft_judge_output/final_judged_results.jsonl" ]; then
    echo "Found existing judge results. Skipping Step 2."
else
    if [ ! -f "./dataset/sft_output/final_merged_results.jsonl" ]; then
        echo "Error: Sampling results not found. Please check Step 1."
        exit 1
    fi
    echo "Running sft_data_judge.py on 8 GPUs..."
    python3 sft_data_judge.py
    echo "Step 2 completed!"
fi

# =========================================
# Step 3: Mixed Sampling
# =========================================
echo ""
echo "========================================="
echo "Step 3: Mixed Sampling"
echo "========================================="

if [ -f "./dataset/sft_output/octopus_corrected_10k.jsonl" ]; then
    echo "Found existing mixed sampling results. Skipping Step 3."
else
    if [ ! -f "./dataset/sft_judge_output/final_judged_results.jsonl" ]; then
        echo "Error: Judge results not found. Please check Step 2."
        exit 1
    fi
    echo "Running sft_mixed_sampling.py..."
    python3 sft_mixed_sampling.py \
        --input-file ./dataset/sft_judge_output/final_judged_results.jsonl \
        --output-file ./dataset/sft_output/octopus_corrected_10k.jsonl \
        --batch-size 512 \
        --max-new-tokens 4096 \
        --temperature 0.7
    echo "Step 3 completed!"
fi

# =========================================
# Step 4: Format Conversion
# =========================================
echo ""
echo "========================================="
echo "Step 4: Format Conversion"
echo "========================================="

if [ -f "./dataset/sft_output/octopus_corrected_10k_format.jsonl" ]; then
    echo "Found existing formatted output. Skipping Step 4."
else
    if [ ! -f "./dataset/sft_output/octopus_corrected_10k.jsonl" ]; then
        echo "Error: Mixed sampling results not found. Please check Step 3."
        exit 1
    fi
    echo "Running make_sft_format.py..."
    python3 make_sft_format.py \
        --input-jsonl ./dataset/sft_output/octopus_corrected_10k.jsonl \
        --output-json ./dataset/sft_output/octopus_corrected_10k_format.json \
        --shuffle
    echo "Step 4 completed!"
fi

# =========================================
# Summary
# =========================================
echo ""
echo "========================================="
echo "Pipeline Completed Successfully!"
echo "========================================="
echo ""
echo "Output files:"
echo "  - Step 1: ./dataset/sft_output/final_merged_results.jsonl"
echo "  - Step 2: ./dataset/sft_judge_output/final_judged_results.jsonl"
echo "  - Step 3: ./dataset/sft_output/octopus_corrected_10k.jsonl"
echo "  - Step 4: ./dataset/sft_output/octopus_corrected_10k_format.jsonl (Final)"
echo ""
echo "========================================="
