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

```bash
pip install -r requirements.txt
```

### 2. Start local model servers

This local setup assumes the Hugging Face models have already been downloaded under `../models/`:

```text
../models/Qwen/Qwen3.6-35B-A3B
../models/Qwen/Qwen3-4B-Instruct
```

Run one OpenAI-compatible server for the main QA agent and one for the faster speculator:

```bash
vllm serve ../models/Qwen/Qwen3.6-35B-A3B --host 0.0.0.0 --port 8000 --max-model-len 8192
```

```bash
vllm serve ../models/Qwen/Qwen3-4B-Instruct --host 0.0.0.0 --port 8001 --max-model-len 8192
```

SGLang works too as long as it exposes OpenAI-compatible `/v1/chat/completions` endpoints on the same ports. No OpenRouter, Gemini, or OpenAI API keys are required for the default local workflow.

## Running Experiments

### Launch a run

```bash
python run.py
```

This evaluates the agent on HotPotQA questions with speculative simulation active. Trajectories, observations, and per-sample metrics land in `run_metrics/`.

You can swap models on the fly:

```bash
python run.py --modelname "../models/Qwen/Qwen3.6-35B-A3B" --guessmodelname "../models/Qwen/Qwen3-4B-Instruct"
```

Pass `--norun` to skip execution and only post-process existing results. Pass `--noprint` for silent mode.

### Compute metrics

```bash
# Aggregate accuracy across every agent/speculator pair
python run.py --norun --getmetric --savemetrics

# Per-sample top-1 and top-3 accuracy lists
python run.py --norun --getmetric2 --savemetrics
```

### Plot results

```bash
# Wall-clock time comparison: normal API vs speculative
python run.py --norun --graph

# Top-1 vs top-3 prediction accuracy per speculator
python run.py --norun --graph3
```

## Hyperparameters

Adjustable in `src/constants.py`:

| Parameter | Default | Description |
|---|---|---|
| `n_samples_to_run` | 20 | Questions to evaluate per run |
| `n_steps_to_run` | 8 | Max reasoning steps per question |
| `guess_num_actions` | 3 | Top-k speculative candidates |
| `temperature` | 1 | Agent sampling temperature |
| `guess_temperature` | 0.1 | Speculator sampling temperature |
| `max_agent_retries` | 1 | Agent retries on malformed output |
| `max_guess_retries` | 3 | Speculator retries on malformed output |

## Sample Data

A curated set of trajectories and metrics is included under `run_metrics/` and `trajs/` to illustrate the output format across different agent/speculator model pairings.
