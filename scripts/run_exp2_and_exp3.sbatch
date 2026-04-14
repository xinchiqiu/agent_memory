#!/bin/bash
#SBATCH --job-name=sage-exp2-3
#SBATCH --partition=hopper
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/exp2_3_%j.out
#SBATCH --error=logs/exp2_3_%j.err

# === Experiment 2 + 3 (sequential, resumable) ===
# Runs granularity ablation first, then retrieval quality.
# Both are resumable — resubmit if time limit is hit.

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

echo "=== Experiment 2 + 3 ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Start:  $(date)"
echo "========================"

cd ${PROJECT_DIR}

# Start vLLM server in background
echo ""
echo ">>> Starting vLLM server with Qwen3-8B..."
${VLLM_ENV}/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 &
VLLM_PID=$!

echo "Waiting for vLLM server to load model..."
for i in $(seq 1 300); do
    if curl -s http://localhost:8000/v1/models 2>/dev/null | grep -q "Qwen3-8B"; then
        echo "vLLM server ready after ${i}s"
        break
    fi
    sleep 2
done

if ! curl -s http://localhost:8000/v1/models 2>/dev/null | grep -q "Qwen3-8B"; then
    echo "ERROR: vLLM server failed to load model"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# --- Experiment 2: Granularity Ablation ---
echo ""
echo ">>> Running Experiment 2: Granularity Ablation..."
${CONDA_ENV}/python run_experiments.py \
    --exp granularity \
    --dataset_dir dataset_cc/ \
    --backend vllm \
    --seed_count 200 \
    --eval_count 300 \
    --output_base results

# --- Experiment 3: Retrieval Quality ---
echo ""
echo ">>> Running Experiment 3: Retrieval Quality..."
${CONDA_ENV}/python run_experiments.py \
    --exp retrieval \
    --dataset_dir dataset_cc/ \
    --backend vllm \
    --seed_count 200 \
    --eval_count 300 \
    --output_base results

echo ""
echo ">>> Experiments 2+3 complete!"
echo "End: $(date)"

# Cleanup
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
