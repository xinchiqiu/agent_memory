#!/bin/bash
#SBATCH --job-name=sage-gemma4
#SBATCH --partition=hopper
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/exp1_gemma4_%j.out
#SBATCH --error=logs/exp1_gemma4_%j.err

# === Exp 1 with Gemma-4-31B-it — multi-model comparison ===

set -e

# Redirect caches to NFS (avoid /home quota)
export HF_HOME="/nfs-share/xinchi/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="/nfs-share/xinchi/.cache/huggingface/hub"
export VLLM_CACHE_ROOT="/nfs-share/xinchi/.cache/vllm"
export TORCH_HOME="/nfs-share/xinchi/.cache/torch"
export XDG_CACHE_HOME="/nfs-share/xinchi/.cache"

# Override model for this experiment
export SAGE_MODEL_OVERRIDE="google/gemma-4-31B-it"

PROJECT_DIR=/nfs-share/xinchi/agent_memory
CONDA_ENV=/nfs-share/xinchi/envs/cp-agent/bin
VLLM_ENV=/nfs-share/xinchi/envs/vllm-judge/bin

mkdir -p ${PROJECT_DIR}/logs

echo "=== Exp 1: Gemma-4-31B-it ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Start:  $(date)"
echo "=============================="

cd ${PROJECT_DIR}

# Start vLLM server with Gemma-4-31B
echo ""
echo ">>> Starting vLLM server with Gemma-4-31B-it..."
${VLLM_ENV}/python -m vllm.entrypoints.openai.api_server \
    --model google/gemma-4-31B-it \
    --served-model-name google/gemma-4-31B-it \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.92 \
    --trust-remote-code &
VLLM_PID=$!

echo "Waiting for vLLM server to load model..."
for i in $(seq 1 600); do
    # Check /v1/models endpoint — only returns the model after it's fully loaded
    if curl -s http://localhost:8000/v1/models 2>/dev/null | grep -q "gemma-4"; then
        echo "vLLM server ready after ${i}s"
        break
    fi
    sleep 2
done

if ! curl -s http://localhost:8000/v1/models 2>/dev/null | grep -q "gemma-4"; then
    echo "ERROR: vLLM server failed to load model"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# Run exp1 with Gemma 4
echo ""
echo ">>> Running Experiment 1 with Gemma-4-31B-it..."
${CONDA_ENV}/python run_experiments.py \
    --exp learning_curve \
    --dataset_dir dataset_cc/ \
    --backend vllm \
    --seed_count 200 \
    --eval_count 300 \
    --num_seeds 1 \
    --output_base results_gemma4

echo ""
echo ">>> Done!"
echo "End: $(date)"

kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
