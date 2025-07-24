import click
import sys
import os
import json
import warnings
from subprocess import check_call
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

warnings.filterwarnings("ignore")
PYTHON = sys.executable

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def format_lr(lr):
    return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e")


def launch_experiment(experiment_config: Dict[str, Any]) -> Dict[str, Any]:
    cmd = [
        PYTHON,
        "cloud_shadows_detection/train.py",
        f"--data_dir={experiment_config['data_dir']}",
        f"--model_name={experiment_config['model_name']}",
        f"--run_dir={experiment_config['run_dir']}",
        f"--batch_size={experiment_config['batch_size']}",
        f"--n_workers={experiment_config['n_workers']}",
        f"--lr={experiment_config['lr']}",
        f"--in_dim={experiment_config['in_dim']}",
        f"--fold={experiment_config['fold']}",
        f"--norm_type={experiment_config['norm']}",
        f"--hidden_dims={experiment_config['hidden_dims']}",
    ]

    if experiment_config["weighted"]:
        cmd.append("--weighted")
    if experiment_config["pretrained"]:
        cmd.append("--pretrained")
    if experiment_config["finetune"]:
        cmd.append("--finetune")
    if experiment_config["use_amp"]:
        cmd.append("--use_amp")

    logger.info(f"Starting experiment with command: {' '.join(cmd)}")
    check_call(cmd)

    # Load and return results
    lr_str = format_lr(experiment_config["lr"])
    job_name = f"{experiment_config['model_name']}_lr{lr_str}_{experiment_config['norm']}_w{experiment_config['weighted']}_f{experiment_config['fold']}"
    directory = Path(experiment_config["run_dir"]) / job_name
    with open(Path(directory, "metrics_test.json"), "r") as f:
        results = json.load(f)

    return {**experiment_config, **results}


def resume_experiment(experiment_config: Dict[str, Any]) -> Dict[str, Any]:
    lr_str = format_lr(experiment_config["lr"])
    job_name = f"{experiment_config['model_name']}_lr{lr_str}_{experiment_config['norm']}_w{experiment_config['weighted']}_f{experiment_config['fold']}"
    directory = Path(experiment_config["run_dir"]) / job_name

    pth_test = Path(directory, "metrics_test.json")
    pth_val = Path(directory, "metrics_val.json")

    if os.path.isfile(pth_test):
        experiment_config["pretrained"] = True
        logger.info(f"Experiment {job_name} already completed. Skipping.")
        with open(pth_test, "r") as f:
            results = json.load(f)
        return {**experiment_config, **results}
    elif os.path.isfile(pth_val):
        logger.info(f"Resuming experiment {job_name} from checkpoint.")
        experiment_config["pretrained"] = True
    else:
        logger.info(f"Starting new experiment {job_name}.")
        experiment_config["pretrained"] = False
    return launch_experiment(experiment_config)


@click.command()
@click.option(
    "--config", default="experiments_config.yaml", help="Path to experiment configuration file"
)
def run_cli(config):
    """Main function to run experiments for the HCSR algorithms."""
    cfg = load_config(config)
    ft = cfg["finetune"]

    experiments = []
    for model_name in cfg["model_names"]:
        for norm in cfg["norm_types"]:
            for weighted in cfg["weights"]:
                for lr in cfg["lrs"]:
                    for fold in range(cfg["n_folds"]):
                        exp_config = {
                            "model_name": model_name,
                            "norm": norm,
                            "weighted": weighted,
                            "lr": float(lr),
                            "fold": fold,
                            "finetune": ft,
                            **cfg["fixed_params"],
                        }
                        experiments.append(exp_config)

    logger.info(f"Launching {len(experiments)} experiments")

    if cfg["n_parallel_training"] > 1:
        with ProcessPoolExecutor(max_workers=cfg.get("n_parallel_training", 1)) as executor:
            results = list(executor.map(resume_experiment, experiments))
    else:
        results = list(map(resume_experiment, experiments))

    results_df = pd.DataFrame(results)
    exp_pth = cfg["fixed_params"]["run_dir"]
    results_df.to_csv(f"{exp_pth}/experiment_results.csv", index=False)
    logger.info("All experiments completed. Results saved.")
    logger.info(f"Best experiment: {results_df.loc[results_df['f1'].idxmax()].to_dict()}")


if __name__ == "__main__":
    run_cli()
