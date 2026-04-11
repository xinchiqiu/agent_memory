"""All tunable parameters in one place."""

# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------
# "hf_local"   — load model directly via HuggingFace transformers (no server)
# "vllm"       — OpenAI-compatible endpoint served by vLLM
# "openai"     — OpenAI API (gpt-4o, etc.)
# "anthropic"  — Anthropic API (claude-*)

BACKEND = "hf_local"   # change this to switch inference backend

# ---------------------------------------------------------------------------
# Model profiles — pick one per role, or set the same model for all
# ---------------------------------------------------------------------------

# Qwen3 (2025) — recommended for NeurIPS experiments
QWEN3_8B  = "Qwen/Qwen3-8B"           # thinking-capable, 8B params
QWEN25_CODER_7B = "Qwen/Qwen2.5-Coder-7B-Instruct"   # code-specialized, no thinking

# Which model to use for each role
EXTRACTION_MODEL  = QWEN3_8B
ALIGNMENT_MODEL   = QWEN3_8B
GENERATION_MODEL  = QWEN3_8B
DIAGNOSIS_MODEL   = QWEN3_8B

CONFIG = {
    # --- Backend ---
    "backend": BACKEND,

    # --- Model names (used by the factory) ---
    "extraction_model":  EXTRACTION_MODEL,
    "alignment_model":   ALIGNMENT_MODEL,
    "generation_model":  GENERATION_MODEL,
    "diagnosis_model":   DIAGNOSIS_MODEL,

    # --- HF local inference ---
    "hf_device_map":  "auto",       # "auto" spreads across all visible GPUs
    "hf_torch_dtype": "bfloat16",   # "bfloat16" | "float16" | "float32"
    "hf_max_new_tokens": 4096,

    # --- Qwen3 thinking mode ---
    # Qwen3 supports explicit <think> reasoning via /think or /no_think suffixes.
    # For alignment (strategy reasoning) use thinking; for code gen disable it.
    "qwen3_thinking_for_alignment":  True,   # adds /think to alignment prompts
    "qwen3_thinking_for_generation": False,  # adds /no_think to code-gen prompts

    # --- vLLM server (used when backend == "vllm") ---
    "vllm_base_url": "http://localhost:8000/v1",
    "vllm_api_key":  "dummy",

    # --- OpenAI API (used when backend == "openai") ---
    "openai_api_key":  "",   # set via env var OPENAI_API_KEY or here
    "openai_model":    "gpt-4o",

    # --- Anthropic API (used when backend == "anthropic") ---
    "anthropic_api_key": "",  # set via env var ANTHROPIC_API_KEY or here
    "anthropic_model":   "claude-opus-4-6",

    # --- Retrieval ---
    "encoder_model": "models/technique_encoder_v2",
    "top_k": 3,

    # --- Verification ---
    "timeout_seconds": 10,
    "max_verification_tests": 50,  # cap tests per problem (CodeContests has 100+)

    # --- Memory ---
    "max_memory_size": 5000,

    # --- Experiment ---
    "seed_problems_count":  200,
    "eval_problems_count":  500,
    "log_dir":              "logs/experiment_001",
    "checkpoint_interval":  50,

    # --- Granularity ablation (Experiment 2) ---
    # Controls what information from retrieved entries is shown to the model.
    # "G1" = no retrieval (free generation only)
    # "G2" = tag hints only (~20 tokens per entry)
    # "G3" = strategy only (technique chain + insight + preconditions) — DEFAULT
    # "G4" = strategy + code snippet (first 400 chars of solution)
    # "G5" = full solution code (up to 1000 chars)
    # "G6" = 3 full solutions (up to 1000 chars each)
    "granularity_mode": "G3",
}

ALLOWED_ALGORITHM_TAGS = {
    "greedy", "constructive",
    "dp",
    "binary_search", "brute_force",
    "graph_dfs", "graphs", "graph_shortest_path", "trees",
    "network_flow", "graph_matching", "two_sat",
    "data_structures", "union_find",
    "implementation", "interactive",
    "math", "number_theory", "combinatorics", "probability",
    "geometry", "fft", "matrices",
    "strings", "hashing", "string_suffix",
    "sorting", "two_pointers",
    "divide_and_conquer", "meet_in_the_middle",
    "bitmasks", "game_theory", "parsing",
}
