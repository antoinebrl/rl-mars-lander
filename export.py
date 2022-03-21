import base64
import inspect
import os
import sys
from pathlib import Path

import numpy as np
import torch.fx
from stable_baselines3 import PPO
from torch import nn
from torch.nn.modules.module import _addindent

from lander import MarsLanderEnv


class Linear:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def __call__(self, x):
        return np.matmul(self.weight, x) + self.bias


class ReLU:
    def __call__(self, x):
        return np.maximum(0, x)


class Sequential:
    def __init__(self, *modules):
        self.modules = modules

    def __call__(self, x):
        for module in self.modules:
            x = module(x)
        return x

    def __getattr__(self, item):
        return self.modules[int(item)]


class PPOPolicy(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.policy_net = model.policy.mlp_extractor.policy_net
        self.action_net = model.policy.action_net

    def forward(self, observation):
        latent_pi = self.policy_net(observation)
        action = self.action_net(latent_pi)
        return action


def format_module(module, indent=2):
    tab = " " * 4
    if type(module) == nn.Module:  # nn.Sequential
        indent += 1
        module_str = "Sequential(\n" + "\n".join(
            [f"{tab * indent}{format_module(m, indent)}," for m in module.children()]
        )
        indent -= 1
        module_str += f"\n{tab * indent})"
    elif type(module) in [nn.ReLU, nn.Linear]:
        module_str = f"{module.__class__.__name__}("
        if len(list(module.parameters())):
            module_str += f"\n{tab * indent}"
            for name, params in module.named_parameters():
                module_str += (
                    f"{tab}{name}=np.frombuffer(base64.decodebytes("
                    + f"{base64.b64encode(params.detach().half().numpy())}"
                    + f"), dtype=np.float16).reshape({list(params.shape)})"
                    + f",\n{tab * indent}"
                )
        module_str += ")"
    else:
        raise NotImplementedError
    return module_str


def module_to_folder(graph: torch.fx.GraphModule, folder, module_name: str = "FxModule"):
    folder = Path(folder)
    Path(folder).mkdir(exist_ok=True)

    tab = " " * 4
    model_str = """import numpy as np\nimport base64\n\n"""
    model_str += inspect.getsource(Sequential) + "\n"
    model_str += inspect.getsource(Linear) + "\n"
    model_str += inspect.getsource(ReLU) + "\n"
    model_str += f"""class {module_name}:\n{tab}def __init__(self):\n{tab*2}super().__init__()\n"""

    for module_name, module in graph.named_children():
        module_str = format_module(module)
        model_str += f"{tab * 2}self.{module_name} = {module_str}\n"

    model_str += f"{_addindent(graph.code, len(tab))}\n"

    module_file = folder / "module.py"
    module_file.write_text(model_str)

    init_file = folder / "__init__.py"
    init_file.write_text("from .module import *")


def main(path):
    env = MarsLanderEnv()
    model = PPO.load(os.path.join(path, "best_model.zip"), env)
    policy = PPOPolicy(model)
    module_to_folder(torch.fx.symbolic_trace(policy), "exported", "MarsLanderPolicy")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python export.py log/2022-01-01/")
        sys.exit(1)

    main(sys.argv[1])
