# Cartpole using Reinforcement Q-Learning Algorithm
<h3 align="center">
<img src="/cartpole_example.gif" width="300">
</h3>

## About

> A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center, or episode length is greater than 200 .

### Hyperparameters:

* GAMMA = 1.0
* LEARNING_RATE = 0.1
* EXPLORATION_RATE = 0.1

## Performance

> CartPole-v0 defines "solving" as getting average reward of 195.0 over 100 consecutive trials.
>

##### Example trial chart

<img src="/episode vs mean_scores.png">

##### Example console chart

<img src="/console.png">

## Run

``
python CartpoleQ.py
``

* `--episode`: direct the episode count
* `--render`: render the GUI

You have to install `gym` to run. And it works on Python > 3.
