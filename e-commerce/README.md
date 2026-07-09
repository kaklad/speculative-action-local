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
../../models/Qwen/Qwen3-4B-Instruct
../../models/Qwen/Qwen3-14B-Instruct
```

Run the main agent/user model:

```bash
vllm serve ../../models/Qwen/Qwen3.6-35B-A3B --host 0.0.0.0 --port 8000 --max-model-len 8192
```

Run the fast guess model:

```bash
vllm serve ../../models/Qwen/Qwen3-4B-Instruct --host 0.0.0.0 --port 8001 --max-model-len 8192
```

Run the medium guess model:

```bash
vllm serve ../../models/Qwen/Qwen3-14B-Instruct --host 0.0.0.0 --port 8002 --max-model-len 8192
```

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

