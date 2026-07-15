This case is built based on the [tau-bench](https://github.com/sierra-research/tau-bench) repository.


## How to setup the environment
1. Go to the `speculative-action-local/e-commerce/tau-bench` directory
2. Create a virtual environment and install the dependencies
```bash
pip install -e .
```
3. Start local OpenAI-compatible model servers. This local setup assumes the Hugging Face models have already been downloaded under `../../models/`:

```text
../../models/Qwen/Qwen3.6-35B-A3B
../../models/Qwen/Qwen3-14B
```

Run the main agent/user model:

```bash
vllm serve ../../models/Qwen/Qwen3.6-35B-A3B --host 0.0.0.0 --port 8000 --max-model-len 8192 --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3
```

Run the fast guess model:

```bash
vllm serve ../../models/Qwen/Qwen3-14B --host 0.0.0.0 --port 8001 --max-model-len 8192
```

All guess configurations use the 14B server on port 8001; the presets only change reasoning effort.

SGLang works too as long as it exposes OpenAI-compatible `/v1/chat/completions` endpoints on the same ports. No OpenAI / Anthropic / Google / Mistral API keys are required for the local workflow.


## How to generate the speculative action results
1. Go to the `speculative-action-local/e-commerce/tau-bench` directory
2. Run the `exp_static.sh` script to generate the speculative action results
```bash
./exp_static.sh
```
3. The results will be saved in the `results` directory.

   The sample results from gpt-5 family and gemini 2.5 flash family used in the paper are saved in the `results` directory. The local script writes new results for the local low / medium / high guesser configs.


## How to analyze the speculative action results
1. Go to the `speculative-action/e-commerce/tau-bench` directory
2. Run the `analysis_static_combine.ipynb` notebook to generate the figures in the paper.
3. The sample figures are saved in the `figures` directory.

## CLI command reference

Run the workflow commands from `speculative-action-local/e-commerce/tau-bench/`.

### Enter the project directory

```bash
cd /data1/jhkim/speculative-action-local/e-commerce/tau-bench
```

### Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

### Local model servers

```bash
vllm serve ../../models/Qwen/Qwen3.6-35B-A3B --served-model-name ../../models/Qwen/Qwen3.6-35B-A3B --host 0.0.0.0 --port 8000 --max-model-len 8192 --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3
```

```bash
vllm serve ../../models/Qwen/Qwen3-14B --served-model-name ../../models/Qwen/Qwen3-14B --host 0.0.0.0 --port 8001 --max-model-len 8192
```

```bash
```

```bash
curl http://localhost:8000/v1/models
curl http://localhost:8001/v1/models
```

### Run all local speculative configurations

```bash
./exp_static.sh
```

### Run the low configuration

```bash
python run.py --agent-strategy tool-calling-static --env retail \
  --model ../../models/Qwen/Qwen3.6-35B-A3B --model-provider local \
  --user-model ../../models/Qwen/Qwen3.6-35B-A3B --user-model-provider local \
  --user-strategy llm --max-concurrency 10 \
  --start-index 0 --end-index 115 \
  --guesser-config guess_configs/local_low.json \
  --baseline-config historical_trajectories/gpt-4o-retail.json
```

### Run the medium configuration

```bash
python run.py --agent-strategy tool-calling-static --env retail \
  --model ../../models/Qwen/Qwen3.6-35B-A3B --model-provider local \
  --user-model ../../models/Qwen/Qwen3.6-35B-A3B --user-model-provider local \
  --user-strategy llm --max-concurrency 10 \
  --start-index 0 --end-index 115 \
  --guesser-config guess_configs/local_medium.json \
  --baseline-config historical_trajectories/gpt-4o-retail.json
```

### Run the high configuration

```bash
python run.py --agent-strategy tool-calling-static --env retail \
  --model ../../models/Qwen/Qwen3.6-35B-A3B --model-provider local \
  --user-model ../../models/Qwen/Qwen3.6-35B-A3B --user-model-provider local \
  --user-strategy llm --max-concurrency 10 \
  --start-index 0 --end-index 115 \
  --guesser-config guess_configs/local_high.json \
  --baseline-config historical_trajectories/gpt-4o-retail.json
```

### Run selected retail tasks

```bash
python run.py --agent-strategy tool-calling-static --env retail \
  --model ../../models/Qwen/Qwen3.6-35B-A3B --model-provider local \
  --user-model ../../models/Qwen/Qwen3.6-35B-A3B --user-model-provider local \
  --user-strategy llm --max-concurrency 3 \
  --task-ids 2 4 6 \
  --guesser-config guess_configs/local_low.json \
  --baseline-config historical_trajectories/gpt-4o-retail.json
```

### Choose a custom result directory

```bash
python run.py --agent-strategy tool-calling-static --env retail \
  --model ../../models/Qwen/Qwen3.6-35B-A3B --model-provider local \
  --user-model ../../models/Qwen/Qwen3.6-35B-A3B --user-model-provider local \
  --user-strategy llm --max-concurrency 10 \
  --start-index 0 --end-index 115 --log-dir results/local \
  --guesser-config guess_configs/local_low.json \
  --baseline-config historical_trajectories/gpt-4o-retail.json
```

### Compare actor and speculator tool calls

```bash
ACTOR_JSON="results/actor-results.json"
SPECULATOR_JSON="results/speculator-results.json"
```

```bash
python compare_actor_speculator_tools.py \
  --actor "$ACTOR_JSON" \
  --speculator "$SPECULATOR_JSON" \
  --actor-window next-response \
  --show 20 \
  --csv actor_vs_speculator.csv
```

```bash
python compare_actor_speculator_tools.py \
  --actor "$ACTOR_JSON" \
  --speculator "$SPECULATOR_JSON" \
  --compare-args \
  --actor-window remainder \
  --task-ids 2 4 6 \
  --show 20 \
  --csv actor_vs_speculator_args_remainder.csv
```

### Open the analysis notebook

```bash
jupyter lab analysis_static_combine.ipynb
```

### Inspect CLI options

```bash
python run.py --help
```

```bash
python compare_actor_speculator_tools.py --help
```
