#!/bin/bash
#SBATCH --job-name=sage-sanity
#SBATCH --partition=hopper
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/sanity_%j.out
#SBATCH --error=logs/sanity_%j.err

# === Sanity Test ===
# Quick end-to-end test: 10 seed problems, 5 eval problems, G3 only
# Should complete in ~10-15 min. Run this first before the big experiments.

set -e

# Redirect caches to NFS (avoid /home quota)
export HF_HOME="/nfs-share/xinchi/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="/nfs-share/xinchi/.cache/huggingface/hub"
export VLLM_CACHE_ROOT="/nfs-share/xinchi/.cache/vllm"
export TORCH_HOME="/nfs-share/xinchi/.cache/torch"
export XDG_CACHE_HOME="/nfs-share/xinchi/.cache"

PROJECT_DIR=/nfs-share/xinchi/agent_memory
CONDA_ENV=/nfs-share/xinchi/envs/cp-agent/bin
VLLM_ENV=/nfs-share/xinchi/envs/vllm-judge/bin

mkdir -p ${PROJECT_DIR}/logs

echo "=== Sanity Test ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Start:  $(date)"
echo "==================="

cd ${PROJECT_DIR}

# Start vLLM server
echo ""
echo ">>> Starting vLLM server with Qwen3-7B..."
${VLLM_ENV}/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 &
VLLM_PID=$!

echo "Waiting for vLLM server to start..."
for i in $(seq 1 300); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "vLLM server ready after ${i}s"
        break
    fi
    sleep 1
done

if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "ERROR: vLLM server failed to start"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# Run quick granularity test (just G1 and G3)
echo ""
echo ">>> Running sanity test: 10 seed, 5 eval..."
${CONDA_ENV}/python run_experiments.py \
    --exp granularity \
    --dataset_dir dataset_cc/ \
    --backend vllm \
    --seed_count 10 \
    --eval_count 5 \
    --output_base results/sanity

echo ""
echo ">>> Sanity test complete!"
echo "End: $(date)"

kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
