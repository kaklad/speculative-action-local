import os
import json
import argparse
from os.path import join

from src import constants
from src.runner import HotPotQARun
from src.utils import Utils
from src.metrics import Metrics
from src.grapher import Grapher


def compute_metrics(runner, save=False, output_dir="./run_metrics"):
    agent = runner.model_name
    guess_model = runner.guess_model_name
    all_avg_metrics = {agent: {}}
    if not os.path.exists(runner.base_traj_path):
        print(f"Path {runner.base_traj_path} does not exist. Skipping...")
    else:
        avg_metrics_dict, n_samples = Metrics.get_action_specific_avg_metric(
            runner.base_traj_path, get_time=True
        )
        print(
            f"AVERAGE METRICS for agent {agent} using guess model {guess_model}:\n",
            json.dumps(avg_metrics_dict, indent=4),
            f"\nfor {n_samples} observations",
        )
        all_avg_metrics[agent][guess_model] = avg_metrics_dict
    if save:
        Utils.save_json(
            all_avg_metrics,
            join(output_dir, f"all_avg_metrics_top{constants.guess_num_actions}.json"),
        )
        metrics_df = Utils.convert_json_to_csv(all_avg_metrics)
        Utils.save_file(
            metrics_df.to_csv(index=False),
            join(output_dir, f"all_avg_metrics_top{constants.guess_num_actions}.csv"),
        )
    return all_avg_metrics


def compute_cumulative_metrics(runner, save=False, output_dir="./run_metrics"):
    agent = runner.model_name
    guess_model = runner.guess_model_name
    all_avg_metrics = {agent: {}}
    if not os.path.exists(runner.base_traj_path):
        print(f"Path {runner.base_traj_path} does not exist. Skipping...")
    else:
        avg_metrics_dict, n_samples = Metrics.cum_metrics(runner.base_traj_path)
        print(
            f"CUMULATIVE METRICS for agent {agent} using guess model {guess_model}:\n",
            json.dumps(avg_metrics_dict, indent=4),
            f"\nfor {n_samples} observations",
        )
        all_avg_metrics[agent][guess_model] = avg_metrics_dict
    if save:
        Utils.save_json(all_avg_metrics, join(output_dir, "list_metrics_top1_3.json"))
    return all_avg_metrics


def main():
    parser = argparse.ArgumentParser(description="HotPotQA Speculative Actions Runner")
    parser.add_argument("--getmetric", action="store_true", help="Compute and print average metrics")
    parser.add_argument("--getmetric2", action="store_true", help="Compute and print cumulative metrics")
    parser.add_argument("--savemetrics", action="store_true", help="Save computed metrics to disk")
    parser.add_argument("--graph", action="store_true", help="Plot agent time comparison graphs")
    parser.add_argument("--graph2", action="store_true", help="Plot top-1 vs top-3 comparison graphs")
    parser.add_argument("--graph3", action="store_true", help="Plot detailed metric comparison graphs")
    parser.add_argument("--no-print", "--noprint", dest="run_print", action="store_false", help="Suppress command line output")
    parser.add_argument("--no-run", "--norun", dest="run_experiment", action="store_false", help="Skip running the experiment")
    parser.add_argument("--modelname", default=constants.local_model_name, help="Agent model name")
    parser.add_argument("--guessmodelname", default=constants.local_guess_model_name, help="Guess model name")
    parser.add_argument("--cleanuptrajs", action="store_true", help="Clean up incomplete trajectories")
    parser.add_argument("--metrics-dir", default=None, help="Explicit trajectory directory for metrics/cleanup")
    parser.add_argument("--output-dir", default="./run_metrics", help="Directory for aggregate metric files")
    parser.add_argument("--samples", type=int, default=None, help="Number of dataset samples to run")
    parser.add_argument("--steps", type=int, default=None, help="Maximum reasoning steps per sample")
    parser.add_argument("--num-guesses", type=int, default=None, help="Number of speculative actions")
    args = parser.parse_args()

    if args.samples is not None:
        constants.n_samples_to_run = args.samples
    if args.steps is not None:
        constants.n_steps_to_run = args.steps
    if args.num_guesses is not None:
        constants.guess_num_actions = args.num_guesses

    runner = HotPotQARun(
        model_name=args.modelname,
        guess_model_name=args.guessmodelname,
        to_print_output=args.run_print,
    )
    if args.metrics_dir is not None:
        runner.base_traj_path = args.metrics_dir

    if args.cleanuptrajs:
        Utils.cleanup_trajs(runner.base_traj_path)

    if args.run_experiment:
        runner.run(webthink_simulate=True, skip_done=True)
        Utils.cleanup_trajs(runner.base_traj_path)

    if args.getmetric:
        compute_metrics(runner, save=args.savemetrics, output_dir=args.output_dir)

    if args.getmetric2:
        compute_cumulative_metrics(runner, save=args.savemetrics, output_dir=args.output_dir)

    if args.graph:
        data = Utils.read_json(
            join(args.output_dir, f"all_avg_metrics_top{constants.guess_num_actions}.json")
        )
        for agent in data.keys():
            Grapher.graph_agent_times(data, agent=agent, output_dir=args.output_dir)

    if args.graph2:
        data = Utils.read_json(join(args.output_dir, "list_metrics_top1_3.json"))
        Grapher.graph_metric3(data, output_dir=args.output_dir)

    if args.graph3:
        data = Utils.read_json(join(args.output_dir, "list_metrics_top1_3.json"))
        Grapher.graph_metric3(data, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
