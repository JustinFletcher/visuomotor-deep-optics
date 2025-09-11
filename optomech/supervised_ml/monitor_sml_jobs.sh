#!/bin/bash

# SML Dataset Generation Monitor Script
# Usage: ./monitor_sml_jobs.sh [dataset_path]
# Default dataset path: $WORKDIR/datasets/optomech-1m

DATASET_PATH=${1:-"$WORKDIR/datasets/optomech-1m"}

echo "=========================================="
echo "SML Dataset Generation Monitor"
echo "Dataset path: $DATASET_PATH"
echo "Time: $(date)"
echo "=========================================="

# Check if dataset directory exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "Dataset directory not found: $DATASET_PATH"
    echo "Jobs may not have started yet, or path is incorrect."
    exit 1
fi

# Count episode files
EPISODE_COUNT=$(find "$DATASET_PATH" -name "episode_*.json" | wc -l)
echo "Episode files found: $EPISODE_COUNT"

# Estimate total samples (rough estimate assuming ~50-100 samples per episode)
ESTIMATED_SAMPLES=$((EPISODE_COUNT * 75))
echo "Estimated samples: ~$ESTIMATED_SAMPLES"

# Show dataset size
echo "Dataset size:"
du -sh "$DATASET_PATH"

# Show recent episode files (last 10)
echo ""
echo "Recent episode files:"
find "$DATASET_PATH" -name "episode_*.json" -printf "%T@ %p\n" | sort -n | tail -10 | while read timestamp filepath; do
    filename=$(basename "$filepath")
    date_str=$(date -d "@$timestamp" "+%Y-%m-%d %H:%M:%S")
    echo "  $date_str - $filename"
done

# Check for running SML jobs
echo ""
echo "Running SLURM jobs:"
squeue -u $USER --name=sml-1m-dataset --format="%.10i %.20j %.8T %.10M %.6D %.20R" 2>/dev/null || echo "No squeue available or no jobs running"

# Show progress toward 1M samples target
TARGET_SAMPLES=1000000
PROGRESS_PERCENT=$((ESTIMATED_SAMPLES * 100 / TARGET_SAMPLES))
echo ""
echo "Progress toward 1M samples target:"
echo "  Current: ~$ESTIMATED_SAMPLES / $TARGET_SAMPLES ($PROGRESS_PERCENT%)"

# Progress bar
BAR_LENGTH=50
FILLED_LENGTH=$((PROGRESS_PERCENT * BAR_LENGTH / 100))
BAR=$(printf "%-${BAR_LENGTH}s" "$(printf "%*s" $FILLED_LENGTH | tr ' ' '=')")
echo "  [$BAR] $PROGRESS_PERCENT%"

echo ""
echo "Monitor complete. Use 'poetry run python optomech/supervised_ml/sml_job_watcher.py' for detailed monitoring."
