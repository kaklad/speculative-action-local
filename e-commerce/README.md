This case is built based on the [tau-bench](https://github.com/sierra-research/tau-bench) repository.


## How to setup the environment
1. Go to the `speculative-action-local/e-commerce/tau-bench` directory
2. Create a virtual environment and install the dependencies
```bash
pip install -e .
```
3. Start local OpenAI-compatible model servers. The commands below are run from
   `speculative-action-local/e-commerce/tau-bench/` and assume that the Hugging
   Face models have already been downloaded under `../../models/`:

```text
../../models/Qwen/Qwen3.6-35B-A3B
../../models/Qwen/Qwen3.5-9B
```

Before starting the 35B server, check the model path and vLLM version. Qwen3.6
requires vLLM 0.19.0 or newer.

```bash
test -f ../../models/Qwen/Qwen3.6-35B-A3B/config.json
vllm --version
```

Run the main agent/user model on GPUs 0 and 1. This configuration uses a
65,536-token server context, keeps the KV cache in the model dtype, and enables
automatic prefix caching for repeated system prompts, tool definitions, and
conversation prefixes.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1 \
vllm serve ../../models/Qwen/Qwen3.6-35B-A3B \
  --served-model-name ../../models/Qwen/Qwen3.6-35B-A3B \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --max-model-len 65536 \
  --max-num-seqs 2 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype auto \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --disable-custom-all-reduce \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --language-model-only \
  --trust-remote-code
```

`--disable-custom-all-reduce` is included because tensor-parallel startup on the
RTX A6000 previously failed in vLLM's custom all-reduce kernel with
`custom_all_reduce.cuh: invalid argument`. Remove it only after confirming that
the installed vLLM/CUDA combination starts reliably without it.

Run the fast Qwen3.5-9B guess model on GPU 2:

Qwen's current instructions require a recent vLLM nightly build for Qwen3.5.
Install or upgrade it in the server environment before launching the model:

```bash
uv pip install vllm --torch-backend=auto \
  --extra-index-url https://wheels.vllm.ai/nightly
```

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2 \
vllm serve ../../models/Qwen/Qwen3.5-9B \
  --served-model-name ../../models/Qwen/Qwen3.5-9B \
  --host 127.0.0.1 \
  --port 8001 \
  --max-model-len 131072 \
  --max-num-seqs 2 \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype auto \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --language-model-only \
  --trust-remote-code
```

All guess configurations use the Qwen3.5-9B server on port 8001; the presets only
change reasoning effort.

### KV cache settings

vLLM uses a KV cache during normal generation automatically; there is no
separate switch required to turn the basic cache on. The relevant options in
the commands above are:

- `--kv-cache-dtype auto`: store the cache in the model dtype. This is the safe
  default for RTX A6000 GPUs.
- `--enable-prefix-caching`: reuse already computed KV blocks when requests
  share an identical token prefix. This reduces prefill work, but not decoding
  time, and does not increase the model's context-length limit.
- `--gpu-memory-utilization 0.90`: reserve up to 90% of each visible GPU for the
  model executor, including the KV cache.
- `--max-model-len 65536`: cap each request's prompt plus generated tokens. A
  larger value allocates more KV-cache capacity and reduces concurrency or may
  prevent startup when GPU memory is insufficient.

For a 131,072-token experiment, first stop the server and replace
`--max-model-len 65536` with `--max-model-len 131072`. Keep
`--max-num-seqs 2`; if startup reports insufficient KV-cache memory, return to
65,536 rather than raising `--gpu-memory-utilization` above the available safe
headroom. FP8 KV cache can reduce memory usage, but the RTX A6000 has no native
FP8 Tensor Core support, so `auto` is the recommended baseline for this setup.

### Verify the 35B server

The model is ready only after the server logs show that application startup is
complete. In a second terminal, verify the registered model:

```bash
curl -sS http://127.0.0.1:8000/v1/models | python -m json.tool
```

Then verify ordinary chat/reasoning output:

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "../../models/Qwen/Qwen3.6-35B-A3B",
    "messages": [{"role": "user", "content": "Return only: server-ok"}],
    "max_tokens": 128
  }' | python -m json.tool
```

Finally, verify the parser used by the decider. A healthy response contains a
`choices[0].message.tool_calls` entry whose function name is `lookup_order`:

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "../../models/Qwen/Qwen3.6-35B-A3B",
    "messages": [{"role": "user", "content": "Look up order A123 using the provided tool."}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "lookup_order",
        "description": "Look up an order",
        "parameters": {
          "type": "object",
          "properties": {"order_id": {"type": "string"}},
          "required": ["order_id"]
        }
      }
    }],
    "tool_choice": "auto",
    "max_tokens": 512
  }' | python -m json.tool
```

If this request returns `auto tool choice requires ...`, the server was not
started with both `--enable-auto-tool-choice` and `--tool-call-parser
qwen3_coder`. A Hermes JSON parsing traceback indicates that the 35B server is
using the wrong parser; restart it with the command above.

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

Use the GPU-specific server commands in **How to setup the environment** above.
The workflow does not start either vLLM server itself; both ports must already
be serving before `./exp_static.sh` is run.

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
