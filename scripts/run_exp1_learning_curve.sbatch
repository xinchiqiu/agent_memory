#!/bin/bash
#SBATCH --job-name=sage-exp1
#SBATCH --partition=hopper
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/exp1_%j.out
#SBATCH --error=logs/exp1_%j.err

# === Experiment 1: Learning Curve ===
# Runs: Full system, No Memory, Random Retrieval, Full History, Tag Oracle
# This is the main result showing the agent improves over time.

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

echo "=== Experiment 1: Learning Curve ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Start:  $(date)"
echo "====================================="

cd ${PROJECT_DIR}

# Start vLLM server in background
echo ""
echo ">>> Starting vLLM server with Qwen3-7B..."
${VLLM_ENV}/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 &
VLLM_PID=$!

# Wait for server to be ready
echo "Waiting for vLLM server to start..."
for i in $(seq 1 300); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "vLLM server ready after ${i}s"
        break
    fi
    sleep 1
done

# Verify server is running
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "ERROR: vLLM server failed to start"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# Run experiment
echo ""
echo ">>> Running Experiment 1: Learning Curve..."
${CONDA_ENV}/python run_experiments.py \
    --exp learning_curve \
    --dataset_dir dataset_cc/ \
    --backend vllm \
    --seed_count 200 \
    --eval_count 300 \
    --num_seeds 1 \
    --output_base results

echo ""
echo ">>> Experiment 1 complete!"
echo "End: $(date)"

# Cleanup
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
