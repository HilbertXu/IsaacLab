#!/bin/bash

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Bimanual-Hammer-Assembly-Vel-WRef-Async-Kpts-Left-Asymmetric-v0 --num_envs 4096 --headless --max_iterations 10000

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Bimanual-Hammer-Assembly-Vel-WRef-Async-Kpts-Right-Asymmetric-v0 --num_envs 4096 --headless --max_iterations 10000

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Bimanual-Hammer-Assembly-Vel-WRef-Async-Kpts-Asymmetric-v0 --num_envs 4096 --headless --max_iterations 40000