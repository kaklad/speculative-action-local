# OPTIMISTIC (no check) — commits every speculation
python run.py --agent-strategy tool-calling-reduce --env retail \
  --model ../../models/Qwen/Qwen3.6-35B-A3B --model-provider local \
  --user-model ../../models/Qwen/Qwen3.6-35B-A3B --user-model-provider local \
  --user-strategy llm --max-concurrency 1 \
  --task-ids 34 41 91 101 102 \
  --guesser-config guess_configs/local_low.json \
  --baseline-config ./historical_trajectories/gpt-4o-retail.json \
  --log-dir results_check_exp

# NON-OPTIMISTIC — add this one flag:  --guesser-check