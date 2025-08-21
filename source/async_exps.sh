#!/bin/bash

################################################################
# with reference, velocity control, with/without reaching stage
################################################################
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Async-Assembly-Baseline-wref-vel-wreach --num_envs 6144 --reset_ratio 0.05 --seed 3407 --sweep --headless --max_iterations 4000

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Async-Assembly-Baseline-wref-vel-woreach --num_envs 6144 --reset_ratio 0.05 --seed 3407 --sweep --headless --max_iterations 4000


# ################################################################
# # with reference, position control, with/without reaching stage
# ################################################################
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Async-Assembly-Baseline-wref-pos-wreach --num_envs 6144 --reset_ratio 0.05 --seed 3407 --sweep --headless --max_iterations 4000

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Async-Assembly-Baseline-wref-pos-woreach --num_envs 6144 --reset_ratio 0.05 --seed 3407 --sweep --headless --max_iterations 4000


################################################################
# without reference, velocity control, with/without reaching stage
################################################################
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Async-Assembly-Baseline-woref-vel-wreach --num_envs 6144 --reset_ratio 0.05 --seed 3407 --sweep --headless --max_iterations 4000

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Async-Assembly-Baseline-woref-vel-woreach --num_envs 6144 --reset_ratio 0.05 --seed 3407 --sweep --headless --max_iterations 4000


################################################################
# without reference, position control, with/without reaching stage
################################################################
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Async-Assembly-Baseline-woref-pos-wreach --num_envs 6144 --reset_ratio 0.05 --seed 3407 --sweep --headless --max_iterations 4000

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Async-Assembly-Baseline-woref-pos-woreach --num_envs 6144 --reset_ratio 0.05 --seed 3407 --sweep --headless --max_iterations 4000

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Async-Assembly-Baseline-wref-vel-wreach-nochunk --num_envs 6144 --reset_ratio 0.05 --seed 3407 --sweep --headless --max_iterations 5000

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Async-Assembly-Baseline-wref-vel-wreach-noreset --num_envs 4096 --reset_ratio 0.0 --seed 3407 --sweep --headless --max_iterations 5000

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Async-Assembly-Baseline-wref-vel-wreach --num_envs 6144 --reset_ratio 0.05 --seed 3407 --sweep --headless --max_iterations 5000

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Async-Assembly-Baseline-wref-vel-wreach --num_envs 6144 --reset_ratio 0.1 --seed 3407 --sweep --headless --max_iterations 5000


# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Async-Assembly-Baseline-wref-vel-wreach-nochunk --num_envs 6144 --reset_ratio 0.05 --seed 3407 --sweep --headless --max_iterations 4000