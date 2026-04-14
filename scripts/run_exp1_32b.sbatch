#!/bin/bash
#SBATCH --job-name=sage-32b
#SBATCH --partition=hopper
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/exp1_32b_%j.out
#SBATCH --error=logs/exp1_32b_%j.err

# === Exp 1 with Qwen3-32B — test if stronger model raises ceiling ===
# Runs: Full system, No Memory, Random Retrieval
# Skips Full History and Tag Oracle to save time

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

echo "=== Exp 1: Qwen3-32B ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Start:  $(date)"
echo "========================="

cd ${PROJECT_DIR}

# Start vLLM server with Qwen3-32B
echo ""
echo ">>> Starting vLLM server with Qwen3-32B..."
${VLLM_ENV}/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-32B \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.92 &
VLLM_PID=$!

echo "Waiting for vLLM server to load model..."
for i in $(seq 1 600); do
    if curl -s http://localhost:8000/v1/models 2>/dev/null | grep -q "Qwen3-32B"; then
        echo "vLLM server ready after ${i}s"
        break
    fi
    sleep 2
done

if ! curl -s http://localhost:8000/v1/models 2>/dev/null | grep -q "Qwen3-32B"; then
    echo "ERROR: vLLM server failed to load model"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# Override model config for 32B
export SAGE_MODEL_OVERRIDE="Qwen/Qwen3-32B"

# Run exp1 with 32B model
echo ""
echo ">>> Running Experiment 1 with Qwen3-32B..."
${CONDA_ENV}/python run_experiments.py \
    --exp learning_curve \
    --dataset_dir dataset_cc/ \
    --backend vllm \
    --seed_count 200 \
    --eval_count 300 \
    --num_seeds 1 \
    --output_base results_32b

echo ""
echo ">>> Done!"
echo "End: $(date)"

kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
