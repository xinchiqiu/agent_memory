#!/bin/bash
#SBATCH --job-name=sage-encoder
#SBATCH --partition=ampere
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/encoder_train_%j.out
#SBATCH --error=logs/encoder_train_%j.err

# --- Setup ---
set -e
PROJECT_DIR=/nfs-share/xinchi/agent_memory
CONDA_ENV=/nfs-share/xinchi/envs/cp-agent/bin

mkdir -p ${PROJECT_DIR}/logs

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Start:  $(date)"
echo "================"

cd ${PROJECT_DIR}

# --- Step 1: Generate training pairs ---
echo ""
echo ">>> Step 1: Generating training pairs..."
${CONDA_ENV}/python encoder_training/train_encoder.py \
    --dataset_dir dataset_cc/ \
    --output_dir models/technique_encoder \
    --loss mnrl \
    --num_pairs 50000 \
    --epochs 5 \
    --batch_size 64

# --- Step 2: Evaluate encoder ---
echo ""
echo ">>> Step 2: Evaluating encoder..."
${CONDA_ENV}/python encoder_training/evaluate_encoder.py \
    --dataset_dir dataset_cc/ \
    --finetuned_model models/technique_encoder

echo ""
echo "=== Done: $(date) ==="
