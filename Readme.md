# Introduction to DRL Homework 2

This file will help you setup and run the codes.


## Requirements

We strongly recommend you to use Anaconda or Miniconda to setup Python environments, as they will automatically install required dependencies.

To install requirements:

```setup
conda env create -n hw2
conda activate hw2
pip install -r requirements.txt
conda install tensorflow-gpu # can also be torch, tensorflow or keras, any DL library you like
```



## Running the Codes
With the environment ready, you can start writing and testing your codes. An example command is:
```setup
python dqn_atari.py --env Breakout-v0
```
You can change the env name to run on different environments.

If you have any problems with the code, feel free to contact me at jin-zhan20@mails.tsinghua.edu.cn.

21:48 uniform action selection perfers action 1

22:19 use zhang667's wrapper

22:42 optimizer does not set any parameters except lr

23:12 ddqn + double + perfers action 1 + zhang667's wrapper

23:32 ddqn + double + perfers action 1 + zhang667's wrapper + linear frame 5e5

0419 10：28
python dqn_atari.py --gpu 0 --model LN --double 0 --num_burn_in 50000 --iter 3000000 &
python dqn_atari.py --gpu 1 --model LN --double 1 --num_burn_in 50000 --iters 3000000 &
python dqn_atari.py --env Enduro-v0 --gpu 2 --model DQN --double 0 --num_burn_in 50000 --iters 3000000 &
python dqn_atari.py --env Enduro-v0 --gpu 3 --model DQN --double 1 --num_burn_in 50000 --iters 3000000 &
python dqn_atari.py --env Enduro-v0 --gpu 0 --model DDQN --double 0 --num_burn_in 50000 --iters 3000000 &
python dqn_atari.py --env Enduro-v0 --gpu 1 --model DDQN --double 1 --num_burn_in 50000 --iters 3000000 &
