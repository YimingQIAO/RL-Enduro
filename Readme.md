# Introduction to Q-learning with Deep Neural Network

This file will help you setup and run the codes.

## Requirements

We strongly recommend you to use Anaconda or Miniconda to setup Python environments, as they will automatically install required dependencies.

To install requirements:

```setup
conda create -n DRL
conda activate DRL
pip install -r requirements.txt
conda install pytorch # can also be torch, tensorflow or keras, any DL library you like
```



## Running the Codes
With the environment ready, you can start writing and testing your codes. An example command is:
```setup
python dqn_atari.py --env Enduro-v0 --gpu 2 --model DQN --double 0 --num_burn_in 50000 --iters 3000000 &
```
