# Mars Lander with Reinforcement Learning

Environment for landing a rover on Mars.

# Usage

## Play the game yourself

You think the game is easy? try it yourself!

```shell
python play.py
```

At each step, the program is expected two values separated by a space:
 - the change in rotation as an integer between -15 and 15
 - the change in thrust can take value -1, 0 and 1

A valid input would be `12 1`.

## Visualize environment

The command below will open a graphical window. It might not work if you use
a remote device or an online notebook (Google Colab, etc). The agent will
take random action.

```shell
PYTHONPATH=$PYTHONPATH:$(pwd) python lander/environment.py
```

# Contributing

- Coding style:
    ```shell
    pip install pre-commit
    pre-commit install
    ```
- Run tests:
    ```shell
    python -m unittest discover --pattern "*test.py"
    ```
