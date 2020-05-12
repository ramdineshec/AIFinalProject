# Cartpole using Reinforcement Q-Learning Algorithm

#### Cartpole
<h3 align="center">
<img src="/cartpole_example.gif" width="300">
</h3>

## About

> A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center, or episode length is greater than 200 .

### Hyperparameters:

* GAMMA = 1.0
* LEARNING_RATE = 0.1
* EXPLORATION_RATE = 0.1

#### The original domains of the input features are these.
* cart position ∈ [-4.8, 4.8]
* cart velocity ∈ [-3.4 10^38, 3.4 10^38]
* angle ∈ [-0.42, 0.42]
* angle velocity ∈ [-3.4 10^38, 3.4 10^38]


### Q-Learning Algorithm
```
The Q-Learning algorithm goes as follows:
1. Set the gamma parameter, and environment rewards in matrix R.
2. Initialize matrix Q to zero.
3. For each episode:
Select a random initial state.
Do While the goal state hasn't been reached.
•	Select one among all possible actions for the current state.
•	Using this possible action, consider going to the next state.
•	Get maximum Q value for this next state based on all possible actions.
•	Compute: Q(state, action) = R(state, action) + Gamma * Max[Q(next state, all actions)]
•	Set the next state as the current state.
End Do
End For
```

### code
```
            while not goal:
                #self.env.render()
                action = self.choose_action(current_state, epsilon)
                obs, reward, done, _ = self.env.step(action)
                goal = done
                new_state = self.discretize(obs)
                self.update_q(current_state, action, reward, new_state, alpha)
                current_state = new_state
                i += 1
```


## Performance

> CartPole-v0 defines "solving" as getting average reward of 195.0 over 100 consecutive trials.
>

##### Example trial chart

<img src="/episode vs mean_scores.png">


## Run

```
python CartpoleQ.py
```

* `--episode`: direct the episode count
* `--render`: render the GUI

You have to install `gym` to run. And it works on Python > 3.
