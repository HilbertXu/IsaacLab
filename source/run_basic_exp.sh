#!/bin/bash

# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Bimanual-Hammer-Assembly-Vel-WRef-Async-Kpts-Left-Asymmetric-v0 --num_envs 4096 --headless --max_iterations 10000

# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Bimanual-Hammer-Assembly-Vel-WRef-Async-Kpts-Right-Asymmetric-v0 --num_envs 4096 --headless --max_iterations 10000

# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Bimanual-Hammer-Assembly-Vel-WRef-Async-Kpts-Asymmetric-v0 --num_envs 4096 --headless --max_iterations 40000


# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Bimanual-Hammer-Assembly-Vel-WRef-Async-Kpts-Left-Asymmetric-v1 --num_envs 4096 --headless --max_iterations 1000 --amp --project hammer-assembly-v1

# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Bimanual-Hammer-Assembly-Vel-WRef-Async-Kpts-Right-Asymmetric-v1 --num_envs 4096 --headless --max_iterations 1000 --amp --project hammer-assembly-v1

# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Bimanual-Hammer-Assembly-Vel-WRef-Async-Kpts-Asymmetric-v1 --num_envs 4096 --headless --max_iterations 2000 --amp --project hammer-assembly-v1

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Bimanual-Hammer-Assembly-Sync-Asymmetric-v1 --num_envs 4096 --headless --max_iterations 6000 --project hammer-assembly-sweep-ratio --reset_ratio 0.0

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Bimanual-Hammer-Assembly-Sync-Asymmetric-v1 --num_envs 4096 --headless --max_iterations 6000 --project hammer-assembly-sweep-ratio --reset_ratio 0.25

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Bimanual-Hammer-Assembly-Sync-Asymmetric-v1 --num_envs 4096 --headless --max_iterations 6000 --project hammer-assembly-sweep-ratio --reset_ratio 0.5

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Bimanual-Hammer-Assembly-Sync-Asymmetric-v1 --num_envs 4096 --headless --max_iterations 6000 --project hammer-assembly-sweep-ratio --reset_ratio 0.75