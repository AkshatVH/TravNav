# Setup Guide

~~~
source .venv/bin/activate
export PYTHONPATH=$PWD/diffusion_policy:$PYTHONPATH
export PYTHONPATH=$PWD/dinov2:$PYTHONPATH
export PYTHONPATH=$PWD/diffusion_policy:$PWD/train:$PYTHONPATH
~~~


# To run a test environment

## Run Without Goal Image/ Exploration Only
~~~
cd waffle_pi_mujoco
python make_world.py --gap 1.0 --seed 0 --shift 0.5
uv run run_sim.py --oracle --steps 400
uv run run_sim.py --ckpt ../deployment/model_weights/nomad.pth --steps 400 --save-frames robot_view_g_0d8_s1d2.mp4 --save-map map_g_0d8_s1d2.png

~~~

## Run with Goal Image / Goal based Navigation

~~~
cd waffle_pi_mujoco
python make_world.py --gap 0.8 --seed 0 --shift 0.5
uv run run_sim.py --oracle --steps 400
python run_sim.py --render-goal 13.4 0 0 --goal-out goal.png
uv run run_sim.py --ckpt ../deployment/model_weights/nomad.pth --goal goal.png --steps 400 --save-frames robot_view.mp4 --save-map map.png
~~~

## Allocating GPU hours
~~~
interactive --gpus=1 -t 01:00:00 -A berzelius-2026-300
~~~


