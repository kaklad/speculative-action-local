# Speculative Execution for Chess Games

Part of [Speculative Actions](../README.md) (root repo). This directory implements speculative execution for **chess**: a smaller "speculator" model predicts opponent moves in parallel to speed up gameplay. The system measures prediction accuracy and potential time savings.

**Working directory:** Run all commands below from **this directory** (`chess-game/`). Config and paths are relative to here: `config.yml` at the root, trajectories under `./trajectories/`.

## Overview

**Speculative action** is a technique where we predict what the opponent will do next and prepare a response in advance. If our prediction is correct, we can save time by using the pre-computed response instead of waiting for the full computation.

### How It Works
There are two separate ways we implement this idea in the chess environment: 1. speculative pipeline, and 2. regular pipeline with ad hoc predictions.

#### Speculative pipeline
This workflow generates trajectories with speculation built-in, computing speculative responses in parallel during gameplay. Note that since the speculative pipeline actually implements the workflow, whenever there is a prediction hit, the next step does NOT have a speculative window. The accuracy is then calculated by the expression (number of match) / (number of speculative windows).

To simulate the case where all steps launch speculations (to estimate the accuracy of Speculator), you can run `guess_over_spec_traj.py` to add additional speculative predictions for non-speculative windows.


#### Regular pipeline
Alternatively, we can implement a regular run of chess, and then on top of the regular trajectory, add in the speculations ad hoc and calculate a posterior. The workflow works as follows:

1. **Generate Base Trajectories**: Play chess games and record all moves, observations, and timing information
2. **Add Speculative Predictions**: Use a smaller/faster speculator model to predict opponent moves at each step
3. **Analyze Performance**: Compare prediction accuracy and calculate potential time savings

## Project Structure

All paths are relative to `chess-game/`, and you will always run from this directory. 

```
chess-game/                          # ← run all commands from here
├── config.yml                       # API keys, models, paths (./config.yml)
├── trajectories/                    # Default data (./trajectories/...)
│   ├── sample_trajectories/         # Example trajectories
│   └── test_trajectories/           # Generated runs (e.g. OpenAI_vs_OpenAI_guessgpt-5-2025-08-07)
├── speculative_workflow/
│   ├── Speculative_Chess.py         # Generate speculative trajectories
│   └── guess_over_spec_traj.py      # Add speculative predictions to speculative trajectories
├── regular_workflow/
│   ├── regular_chess.py             # Generate regular chess trajectories
│   └── guess_over_regular_traj.py   # Add speculative predictions to regular trajectories
├── utils.py                         # Shared utilities (imported by workflows)
├── guess_analysis.py                 # Analyze prediction accuracy (use --confidence for confidence-aware)
├── plot.py                          # Unified plotting: time-token scatter or combined bar charts
├── confidence-prediction.py         # Add confidence estimates to step files
└── textarena/                       # Chess environment framework
```

## Setup

**From the `chess-game/` directory** (see [root README](../README.md) for the working-directory convention):

### 1. Install dependencies

```bash
uv sync
```

The project environment contains the chess workflow dependencies. Install and run vLLM in a separate serving environment appropriate for your CUDA/PyTorch setup; `uv sync` does not install vLLM.

### 2. Start local model servers

This local setup assumes the Hugging Face models have already been downloaded under `./models/`:

```text
../models/Qwen/Qwen3.6-35B-A3B
../models/Qwen/Qwen3.5-4B
```

Run one OpenAI-compatible server for the main chess-playing model and one for the faster guess model:

```bash
vllm serve ../models/Qwen/Qwen3.6-35B-A3B --served-model-name ../models/Qwen/Qwen3.6-35B-A3B --host 0.0.0.0 --port 8000 --max-model-len 8192
```

```bash
vllm serve ../models/Qwen/Qwen3.5-4B --served-model-name ../models/Qwen/Qwen3.5-4B --host 0.0.0.0 --port 8001 --max-model-len 8192
```

SGLang works too as long as it exposes OpenAI-compatible `/v1/chat/completions` endpoints on the same ports.

### 3. Configure models

The config file has two types of models:

- **Main models**: Used for actual game playing (can be larger/smarter)
- **Speculator model**: Used for quick predictions (should be smaller/faster)

```yaml
api:
  local:
    key: "EMPTY"
    main_base_url: "http://localhost:8000/v1"
    guess_base_url: "http://localhost:8001/v1"

models:
  local:
    root: "../models"
    main: "Qwen/Qwen3.6-35B-A3B"       # Resolved as ../models/Qwen/Qwen3.6-35B-A3B
    guess: "Qwen/Qwen3.5-4B"    # Resolved as ../models/Qwen/Qwen3.5-4B

game:
  agent_name0: "Local"
  agent_name1: "Local"

guess:
  model_name: "Qwen/Qwen3.5-4B"
  provider: "local"
```

## Usage

Run all commands from `chess-game/`.

### Workflow 1: Speculative generation (Recommended)

Generate trajectories with speculation built-in (speculative responses computed in parallel during gameplay):

```bash
uv run speculative-chess
```

Output goes under `./trajectories/` (or `Spec_Chess_Trajs/` depending on config). Each run produces e.g.:
- `stepsinfo.json` — game history
- `rewards.json`, `game_info.json`, `log.txt`

Then format the trajectory for analysis:

```bash
uv run speculative_workflow/guess_over_spec_traj.py --base-dir ./trajectories/test_trajectories
```

`--base-dir` processes every immediate subdirectory containing `stepsinfo.json`. To process one run only, use `--stepsinfo ./trajectories/test_trajectories/${RUN_ID}/stepsinfo.json`. The script creates `steps_info_{model_name}_guess_{num_guesses}.json` beside each input file.

**Note:** In the speculative pipeline there is no speculative window after a prediction hit. Use `--add-spec` with `guess_over_spec_traj.py` to add speculative predictions for those windows too (for accuracy estimation).


### Workflow 2 (Alternative): Regular trajectory method

#### Step 1: Generate base trajectories

```bash
uv run regular-chess
```

This creates a trajectory under `./trajectories/` with `stepsinfo.json`, `rewards.json`, `game_info.json`, `log.txt`.

#### Step 2: Add speculative predictions

```bash
uv run regular_workflow/guess_over_regular_traj.py --base-dir ./trajectories/test_trajectories
```

Use `--base-dir` for all immediate run subdirectories, or `--stepsinfo/-i` for one trajectory directory or `stepsinfo.json`. This produces `steps_info_{model_name}_guess_{num_guesses}.json` in the same directory.

## Analyze results

From `chess-game/`, run analysis on a base path containing per-run subdirectories. Without `--base-dir`, the command uses `paths.trajectories` from `config.yml`:

```bash
uv run guess_analysis.py --base-dir ./trajectories/test_trajectories
```

This reports:
- **Accuracies:** speculative-window accuracy (hits / speculative windows) and step accuracy (hits / total steps)
- **Time savings** and **token usage** for speculation

Results are written to `analysis_results.csv` in each trajectory directory.


## Visualization

Use the unified `plot.py` with subcommands:

**Combined bar charts** (accuracy + time saved, aggregated over all trajectory dirs under a root, Figure 2 in the paper):

```bash
uv run python plot.py bars --root-path ./trajectories/sample_trajectories
```

The PDF is written to the provided directory by default: `<root_path>/combined_bar_plots.pdf`. Override with `-o path/to/file.pdf`.

Options for `bars`:
- `--target-steps 30 50` — which step counts to plot (or set `analysis.target_steps` in config)
- `--speculative-accuracy` — use speculative/window accuracy (CSV column 12); otherwise step accuracy (column 11)
- `--output` / `-o` — output path (default: `<root_path>/combined_bar_plots.pdf`)

**Time vs tokens** (scatter; one PDF per trajectory folder under the base dir):

```bash
uv run plot.py time-token
# Optional: base dir (default from config paths.sample_trajectories)
uv run plot.py time-token --base-dir ./trajectories/sample_trajectories
# With confidence-based policy overlay (reads analysis_results_with_confidence.csv):
uv run plot.py time-token --confidence
```

Output: each folder gets `plot.pdf`, or `plot_with_confidence.pdf` when using `--confidence`.



## Confidence-aware selective branching

To try the confidence-aware selective branching policy from the paper:

1. Add confidence estimates to a step file (`steps_info_*_guess_*.json`):
   ```bash
   uv run confidence-prediction.py --input ./trajectories/test_trajectories/${RUN_ID}/steps_info_Qwen_Qwen3_5_4B_guess_3.json
   ```
2. Run analysis with the `--confidence` flag (uses `steps_info*_confidence_prediction.json`, filters guesses by confidence > 50):
   ```bash
   uv run guess_analysis.py --base-dir ./trajectories/test_trajectories --confidence
   ```
3. Visualize (time vs tokens with confidence policy overlay):
   ```bash
   uv run plot.py time-token --base-dir ./trajectories/test_trajectories --confidence
   ```


## Configuration options

### Game settings

```yaml
game:
  num_players: 2
  stop_after: 50              # Number of moves before stopping
  agent_name0: "Local"        # First player's provider
  agent_name1: "Local"        # Second player's provider
  num_guesses: 3              # Number of speculative predictions per move
```

### Speculator configuration

```yaml
guess:
  model_name: "Qwen/Qwen3.5-4B"  # Faster model for speculation
  provider: "local"
```

### Analysis settings

```yaml
analysis:
  target_steps: [20, 30, 40, 50]  # Analyze at these step counts
```

### Path settings

Paths are relative to `chess-game/` when you run from this directory:

```yaml
paths:
  trajectories: "./trajectories/test_trajectories"
  sample_trajectories: "./trajectories/sample_trajectories"
```

### Different speculation counts

To experiment with different numbers of speculative predictions, edit `config.yml`:

```yaml
game:
  num_guesses: 3  
```


## Sample trajectories

`./trajectories/sample_trajectories/` contains example runs. Each subdirectory is one game with analysis files (e.g. different numbers of speculative predictions). 






## local server

vllm serve ../models/Qwen/Qwen3.6-35B-A3B --served-model-name ../models/Qwen/Qwen3.6-35B-A3B --host 0.0.0.0 --port 8000 --max-model-len 8192
vllm serve ../models/Qwen/Qwen3.5-4B --served-model-name ../models/Qwen/Qwen3.5-4B --host 0.0.0.0 --port 8001 --max-model-len 8192

## CLI command reference

Run these commands from `speculative-action-local/chess-game/`.

### Environment

```bash
uv sync
```

### Local model servers

Run each server in its own terminal from a vLLM-capable serving environment:

```bash
vllm serve ../models/Qwen/Qwen3.6-35B-A3B --served-model-name ../models/Qwen/Qwen3.6-35B-A3B --host 0.0.0.0 --port 8000 --max-model-len 8192
```

```bash
vllm serve ../models/Qwen/Qwen3.5-4B --served-model-name ../models/Qwen/Qwen3.5-4B --host 0.0.0.0 --port 8001 --max-model-len 8192
```

### Speculative workflow

For commands that target one generated run, set its directory name first:

```bash
RUN_ID="replace-with-run-directory-name"
```

```bash
uv run speculative-chess
```

```bash
uv run speculative-chess --stop-after 10 --trajectories-dir ./trajectories/test_trajectories
```

```bash
uv run speculative_workflow/guess_over_spec_traj.py --base-dir ./trajectories/test_trajectories
```

```bash
uv run speculative_workflow/guess_over_spec_traj.py --base-dir ./trajectories/test_trajectories --add-spec
```

```bash
uv run speculative_workflow/guess_over_spec_traj.py --stepsinfo ./trajectories/test_trajectories/${RUN_ID}/stepsinfo.json --add-spec
```

### Regular workflow

```bash
uv run regular-chess
```

```bash
uv run regular-chess --stop-after 10 --trajectories-dir ./trajectories/test_trajectories
```

```bash
uv run regular_workflow/guess_over_regular_traj.py --base-dir ./trajectories/test_trajectories
```

```bash
uv run regular_workflow/guess_over_regular_traj.py --stepsinfo ./trajectories/test_trajectories/${RUN_ID}/stepsinfo.json
```

### Analysis

```bash
uv run guess_analysis.py --base-dir ./trajectories/test_trajectories
```

```bash
uv run confidence-prediction.py --input ./trajectories/test_trajectories/${RUN_ID}/steps_info_Qwen_Qwen3_5_4B_guess_3.json
```

```bash
uv run guess_analysis.py --base-dir ./trajectories/test_trajectories --confidence
```

### Visualization

```bash
uv run plot.py time-token --base-dir ./trajectories/test_trajectories
```

```bash
uv run plot.py time-token --base-dir ./trajectories/test_trajectories --confidence
```

```bash
uv run plot.py bars --root-path ./trajectories/test_trajectories
```

```bash
uv run plot.py bars --root-path ./trajectories/test_trajectories --target-steps 20 30 40 50 --speculative-accuracy
```

```bash
uv run plot.py bars --root-path ./trajectories/test_trajectories --output ./trajectories/test_trajectories/combined_bar_plots.pdf
```
