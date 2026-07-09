python run.py --agent-strategy tool-calling-static --env retail \
 --model ../../models/Qwen/Qwen3.6-35B-A3B --model-provider local --user-model ../../models/Qwen/Qwen3.6-35B-A3B \
 --user-model-provider local --user-strategy llm --max-concurrency 10 \
 --start-index 0 --end-index 115 \
 --guesser-config guess_configs/local_low.json \
 --baseline-config ./historical_trajectories/gpt-4o-retail.json

python run.py --agent-strategy tool-calling-static --env retail \
 --model ../../models/Qwen/Qwen3.6-35B-A3B --model-provider local --user-model ../../models/Qwen/Qwen3.6-35B-A3B \
 --user-model-provider local --user-strategy llm --max-concurrency 10 \
 --start-index 0 --end-index 115 \
 --guesser-config guess_configs/local_medium.json \
 --baseline-config ./historical_trajectories/gpt-4o-retail.json

python run.py --agent-strategy tool-calling-static --env retail \
 --model ../../models/Qwen/Qwen3.6-35B-A3B --model-provider local --user-model ../../models/Qwen/Qwen3.6-35B-A3B \
 --user-model-provider local --user-strategy llm --max-concurrency 10 \
 --start-index 0 --end-index 115 \
 --guesser-config guess_configs/local_high.json \
 --baseline-config ./historical_trajectories/gpt-4o-retail.json
