# Speculative Execution for HotPotQA

Part of [Speculative Actions](../README.md) (root repo). This sub-project applies speculative execution to **multi-hop question answering** on the [HotPotQA](https://hotpotqa.github.io/) benchmark. A lightweight speculator LLM races ahead of the Wikipedia API, predicting observations before they arrive, so the agent can pre-compute its next reasoning step.

> **Note:** All commands assume you are inside this directory (`hotpotqa/`). Paths in the codebase are relative to here.

## Background

In a standard ReAct loop the agent waits for each Wikipedia response before reasoning about the next step. With speculative execution, a fast secondary model predicts the API response while the call is still in flight. When the prediction matches, the agent's next thought is already prepared — saving one round-trip of latency per correct guess.

### Pipeline

1. The **agent** follows a Thought → Action → Observation loop, querying Wikipedia via Search and Lookup calls
2. In parallel, a smaller **speculator** predicts what each Wikipedia call will return
3. Every step records both the real and speculated observations side-by-side
4. After the run, we evaluate how often the speculator's top-k predictions matched the agent's actual next action, and how much wall-clock time could have been saved

## Repository Layout

```
hotpotqa/
├── src/
│   ├── runner.py               # Experiment driver (HotPotQARun)
│   ├── llm_client.py           # Multi-provider LLM client (local, OpenRouter, Gemini, OpenAI)
│   ├── environment.py          # Wikipedia environment (WikiEnv)
│   ├── wrappers.py             # Gymnasium wrappers (HotPotQA, Logging, History)
│   ├── metrics.py              # Prediction accuracy computation
│   ├── grapher.py              # Bar charts and accuracy visualizations
│   ├── prompts.py              # Prompt templates
│   ├── utils.py                # I/O helpers
│   └── constants.py            # Hyperparameters, model names, local endpoints
├── run.py                      # CLI entrypoint
├── prompts/                    # Few-shot examples (JSON)
├── data/                       # HotPotQA dataset splits
├── run_metrics/                # Saved results: per-sample trajectories, aggregate metrics, plots
├── trajs/                      # Raw logged trajectories
└── requirements.txt
```

## Getting Started

### 1. Install dependencies

Use Python 3.10 or newer (the tested local environment uses Python 3.13):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The project environment contains the HotPotQA workflow dependencies. Install and run vLLM in a separate serving environment appropriate for your CUDA/PyTorch setup; `requirements.txt` does not install vLLM.

### 2. Start local model servers

This local setup assumes the Hugging Face models have already been downloaded under `../models/`:

```text
../models/Qwen/Qwen3.6-35B-A3B
../models/Qwen/Qwen3.5-9B
```

Run one OpenAI-compatible server for the main QA agent and one for the faster speculator:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1 \
vllm serve ../models/Qwen/Qwen3.6-35B-A3B \
  --served-model-name ../models/Qwen/Qwen3.6-35B-A3B --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 2 --max-model-len 65536 --max-num-seqs 2 \
  --gpu-memory-utilization 0.90 --kv-cache-dtype auto --enable-prefix-caching \
  --enable-chunked-prefill --max-num-batched-tokens 8192 \
  --disable-custom-all-reduce --reasoning-parser qwen3 --language-model-only \
  --trust-remote-code
```

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2 \
vllm serve ../models/Qwen/Qwen3.5-9B \
  --served-model-name ../models/Qwen/Qwen3.5-9B --host 127.0.0.1 --port 8001 \
  --max-model-len 131072 --max-num-seqs 2 --gpu-memory-utilization 0.85 \
  --kv-cache-dtype auto --enable-prefix-caching --reasoning-parser qwen3 \
  --language-model-only --trust-remote-code
```

SGLang works too as long as it exposes OpenAI-compatible `/v1/chat/completions` endpoints on the same ports. No OpenRouter, Gemini, or OpenAI API keys are required for the default local workflow.

## Running Experiments

### Launch a run

```bash
python run.py --samples 20 --steps 8 --num-guesses 3
```

This evaluates the agent on HotPotQA questions with speculative simulation active. Trajectories, observations, and per-sample metrics land in `run_metrics/`.

You can swap models on the fly:

```bash
python run.py --modelname "../models/Qwen/Qwen3.6-35B-A3B" --guessmodelname "../models/Qwen/Qwen3.5-9B" --samples 20 --steps 8 --num-guesses 3
```

Pass `--no-run` to skip execution and only post-process existing results. `--norun` remains as a compatibility alias. Likewise, use `--no-print` (or `--noprint`) for silent generation. Use `--metrics-dir` to analyze a trajectory directory explicitly and `--output-dir` to select where aggregate metrics and graph images are written.

### Compute metrics

```bash
# Aggregate accuracy for the selected agent/speculator pair
python run.py --no-run --getmetric --savemetrics --num-guesses 3

# Per-sample top-1 and top-3 accuracy lists
python run.py --no-run --getmetric2 --savemetrics --num-guesses 3
```

### Plot results

```bash
# Wall-clock time comparison: normal API vs speculative
python run.py --no-run --graph --num-guesses 3

# Top-1 vs top-3 prediction accuracy per speculator
python run.py --no-run --graph3 --num-guesses 3
```

## Hyperparameters

Defaults are defined in `src/constants.py`; the main run controls also have CLI overrides:

| Parameter | Default | Description |
|---|---|---|
| `n_samples_to_run` | 20 | Questions to evaluate per run (`--samples`) |
| `n_steps_to_run` | 8 | Max reasoning steps per question (`--steps`) |
| `guess_num_actions` | 3 | Top-k speculative candidates (`--num-guesses`) |
| `temperature` | 1 | Agent sampling temperature |
| `guess_temperature` | 0.1 | Speculator sampling temperature |
| `max_agent_retries` | 1 | Agent retries on malformed output |
| `max_guess_retries` | 3 | Speculator retries on malformed output |

## Sample Data

A curated set of trajectories and metrics is included under `run_metrics/` and `trajs/` to illustrate the output format across different agent/speculator model pairings.

## CLI command reference

Run these commands from `speculative-action-local/hotpotqa/`.

### Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Local model servers

Run each server in its own terminal from a vLLM-capable serving environment. The explicit served names match the model identifiers used by the Python client.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1 \
vllm serve ../models/Qwen/Qwen3.6-35B-A3B \
  --served-model-name ../models/Qwen/Qwen3.6-35B-A3B --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 2 --max-model-len 65536 --max-num-seqs 2 \
  --gpu-memory-utilization 0.90 --kv-cache-dtype auto --enable-prefix-caching \
  --enable-chunked-prefill --max-num-batched-tokens 8192 \
  --disable-custom-all-reduce --reasoning-parser qwen3 --language-model-only \
  --trust-remote-code
```

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2 \
vllm serve ../models/Qwen/Qwen3.5-9B \
  --served-model-name ../models/Qwen/Qwen3.5-9B --host 127.0.0.1 --port 8001 \
  --max-model-len 131072 --max-num-seqs 2 --gpu-memory-utilization 0.85 \
  --kv-cache-dtype auto --enable-prefix-caching --reasoning-parser qwen3 \
  --language-model-only --trust-remote-code
```

Optional endpoint checks:

```bash
curl http://localhost:8000/v1/models
curl http://localhost:8001/v1/models
```

### Run experiments

```bash
python run.py --samples 20 --steps 8 --num-guesses 3
```

```bash
python run.py --no-print --samples 20 --steps 8 --num-guesses 3
```

```bash
python run.py --modelname ../models/Qwen/Qwen3.6-35B-A3B --guessmodelname ../models/Qwen/Qwen3.5-9B --samples 20 --steps 8 --num-guesses 3
```

Environment-variable endpoint overrides:

```bash
LOCAL_MAIN_BASE_URL=http://localhost:8000/v1 LOCAL_GUESS_BASE_URL=http://localhost:8001/v1 python run.py --samples 20 --steps 8 --num-guesses 3
```

### Clean incomplete trajectories

```bash
python run.py --no-run --cleanuptrajs
```

```bash
python run.py --no-run --cleanuptrajs --metrics-dir ./run_metrics/agent_Qwen3.6-35B-A3B_top3/trajs_Qwen3.5-9B
```

### Compute and save metrics

Use the model arguments so the derived trajectory path matches the generation run:

```bash
python run.py --no-run --modelname ../models/Qwen/Qwen3.6-35B-A3B --guessmodelname ../models/Qwen/Qwen3.5-9B --num-guesses 3 --getmetric --savemetrics
```

```bash
python run.py --no-run --modelname ../models/Qwen/Qwen3.6-35B-A3B --guessmodelname ../models/Qwen/Qwen3.5-9B --num-guesses 3 --getmetric2 --savemetrics
```

Analyze an explicit trajectory directory and write aggregate files elsewhere:

```bash
python run.py --no-run --modelname ../models/Qwen/Qwen3.6-35B-A3B --guessmodelname ../models/Qwen/Qwen3.5-9B --num-guesses 3 --metrics-dir ./run_metrics/agent_Qwen3.6-35B-A3B_top3/trajs_Qwen3.5-9B --output-dir ./run_metrics/local_summary --getmetric --getmetric2 --savemetrics
```

### Plot saved metrics

```bash
python run.py --no-run --num-guesses 3 --graph
```

```bash
python run.py --no-run --num-guesses 3 --graph3
```

For a custom aggregate output directory:

```bash
python run.py --no-run --num-guesses 3 --output-dir ./run_metrics/local_summary --graph
```

```bash
python run.py --no-run --num-guesses 3 --output-dir ./run_metrics/local_summary --graph3
```

### Inspect CLI options

```bash
python run.py --help
```
