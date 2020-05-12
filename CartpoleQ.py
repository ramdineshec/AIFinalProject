import gym
import numpy as np
import math
from collections import deque
import datetime
import matplotlib.pyplot as plt

X = []
Y = []


class QlearningCartpole():
    def __init__(self):
        self.buckets = (1, 1, 6, 12,)  # down-scaling feature space to discrete range
        self.n_episodes = 1000  # training episodes
        self.n_win_ticks = 195  # average ticks over 100 episodes required for win
        self.min_alpha = 0.1  # learning rate
        self.min_epsilon = 0.1  # exploration rate
        self.gamma = 1.0  # discount factor
        self.ada_divisor = 25  # only for development purposes

        self.env = gym.make('CartPole-v0')

        self.Q = np.zeros(self.buckets + (self.env.action_space.n,))  # Q matrix

    def discretize(self, obs):
        upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50)]
        lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50)]
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)

    def choose_action(self, state, epsilon):
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.Q[state])

    def update_q(self, state_old, action, reward, state_new, alpha):
        self.Q[state_old][action] += alpha * (
                reward + self.gamma * np.max(self.Q[state_new]) - self.Q[state_old][action])

    def get_epsilon(self, t):
        return max(self.min_epsilon, min(1, 1.0 - math.log10((t + 1) / self.ada_divisor)))

    def get_alpha(self, t):
        return max(self.min_alpha, min(1.0, 1.0 - math.log10((t + 1) / self.ada_divisor)))

    def run(self):
        scores = deque(maxlen=100)

        for e in range(self.n_episodes):
            current_state = self.discretize(self.env.reset())

            alpha = self.get_alpha(e)
            epsilon = self.get_epsilon(e)
            goal = False
            i = 0

            while not goal:
                #self.env.render()
                action = self.choose_action(current_state, epsilon)
                obs, reward, done, _ = self.env.step(action)
                goal = done
                new_state = self.discretize(obs)
                #td_target = reward + self.gamma * np.max(self.Q[new_state])
                #td_delta = td_target - self.Q[current_state][action]
                #self.Q[current_state][action] += alpha * td_delta
                self.update_q(current_state, action, reward, new_state, alpha)
                current_state = new_state
                i += 1

            scores.append(i)
            mean_score = np.mean(scores)
            X.append(e)
            Y.append(mean_score)
            if mean_score >= self.n_win_ticks and e >= 100:
                print('Ran {} episodes. Solved after {} trials '.format(e, e - 100))
                return e - 100
            if e % 100 == 0:
                print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))


if __name__ == "__main__":
    cartPole = QlearningCartpole()
    startTime = datetime.datetime.now()
    cartPole.run()
    endTime = datetime.datetime.now()
    time_diff = endTime - startTime
    total_seconds = time_diff.total_seconds()
    minutes = total_seconds / 60
    print('Time taken to solve cartpole Problem using Q Learning Algorithm : %d seconds' % (total_seconds))
    plt.xlabel('Episodes')
    plt.ylabel('mean_score')
    plt.title('Cartpole using Q learning Algorithm')
    plt.plot(X, Y)
    plt.show()
