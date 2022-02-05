import argparse
import os.path
from datetime import datetime

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import TensorBoardOutputFormat

from lander import MarsLanderEnv


class StoreDict(argparse.Action):
    """
    Custom argparse action for storing dict.
    In: args1:0.0 args2:"dict(a=1)"
    Out: {'args1': 0.0, arg2: dict(a=1)}
    """

    def __call__(self, parser, namespace, values, option_string=None):
        arg_dict = {}
        for arguments in values:
            key, *value = arguments.split(":")
            arg_dict[key] = eval(":".join(value))
        setattr(namespace, self.dest, arg_dict)


def setup_experiment_dir(args):
    name = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.experiment:
        name += "-" + args.experiment
    dir = os.path.join(args.output, name)
    os.mkdir(dir)
    return dir


def flatten_dict(dd, separator=".", prefix=""):
    # taken from https://stackoverflow.com/a/19647596
    return (
        {
            prefix + separator + k if prefix else k: v
            for kk, vv in dd.items()
            for k, v in flatten_dict(vv, separator, kk).items()
        }
        if isinstance(dd, dict)
        else {prefix: dd}
    )


def main(args):
    exp_name = setup_experiment_dir(args)

    env = MarsLanderEnv()
    eval_env = MarsLanderEnv(eval=True)

    env.reset()
    eval_env.reset()

    # RL agent
    if args.algo == "ppo":
        model_cls = PPO
    else:
        raise NotImplementedError(f"Only PPO is a supported algorithm (for now). Got {args.algo}")
    if "learning_rate" in args.hyperparams:
        base_lr = args.hyperparams["learning_rate"]
        args.hyperparams["learning_rate"] = lambda x: np.sin(x * np.pi / 2) * base_lr * 9 / 10 + base_lr / 10
    model = model_cls("MultiInputPolicy", env, **args.hyperparams, tensorboard_log=exp_name, verbose=2)

    if args.checkpoint:
        model.set_parameters(args.checkpoint)

    try:
        model.learn(
            total_timesteps=args.steps,
            log_interval=args.log_interval,
            eval_env=eval_env,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            eval_log_path=exp_name,
        )
    finally:
        # Retrieve metrics and log hyper-parameters
        evaluation = np.load(os.path.join(exp_name, "evaluations.npz"))
        mean_ep_length = np.mean(evaluation["ep_lengths"])
        mean_reward = np.mean(evaluation["results"])
        tbs = [
            formatter for formatter in model.logger.output_formats if isinstance(formatter, TensorBoardOutputFormat)
        ]
        hparams = {
            k: v if isinstance(v, (int, float, str, bool, torch.Tensor)) else v.__class__.__name__
            for k, v in flatten_dict(vars(args)).items()
        }
        for tb in tbs:
            tb.writer.add_hparams(
                hparams,
                {
                    "eval/mean_reward": mean_reward,
                    "eval/mean_ep_length": mean_ep_length,
                },
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training script for CodinGame Mars Lander RL agent")

    # TODO: support more algorithms
    parser.add_argument("--algo", help="RL algorithm", type=str, default="ppo", choices=["ppo"])
    parser.add_argument("--steps", help="Total number of training steps", type=int, default=5_000_000)
    parser.add_argument(
        "--hyperparams",
        "-params",
        type=str,
        nargs="+",
        action=StoreDict,
        help="Overwrite hyper-parameter of the RL algorithm (e.g. learning_rate:0.01 train_freq:10)",
    )
    parser.add_argument("--checkpoint", help="Path to saved parameters", type=str, default="")

    parser.add_argument("--output", help="Path to output directory", default="logs")
    parser.add_argument(
        "--experiment", default="", type=str, metavar="NAME", help="Name of experiment, name of sub-folder for output"
    )
    parser.add_argument(
        "--log-interval", help="Number of timesteps between two consecutive logging events", default=100, type=int
    )
    parser.add_argument(
        "--eval-freq", help="Run evaluation of the agent every `eval_freq` timesteps", default=10000, type=int
    )
    parser.add_argument(
        "--n-eval-episodes", help="Number of episode used for evaluation of the agent", default=100, type=int
    )

    args = parser.parse_args()
    main(args)
