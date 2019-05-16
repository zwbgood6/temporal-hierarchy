# Servoing

This directory contains code for visual servoing with CEM, environment rollout simulators, and custom MuJoCo environments for experiment usage.

## Installing Custom MuJoCo Environments

If you wish to use custom environments such as `CustomPush-v0`, execute `cd gym-push` to get into the directory containing the MuJoCo environment code and run `pip3 install --user -e .` to install the MuJoCo environment. Successful installation requires that `gym` and `mujoco` are installed in your machine.

## Playing with Custom Environments

You can play with `CustomPush-v0` using `debug_env.py` in this directory. Use `WASD` keys to move the robot arm; use `1-8` to move other parts of the robot (rotation, grippers, etc.). This code needs to be executed in a machine with complete display settings.

## Running CEM

Execute `cd cem` to enter the `cem` directory and run `python3 main.py` with custom parameters (inspect `main.py` for more details on what parameters are available).

## Last Thoughts

Code on!
