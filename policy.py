import random

import numpy as np


class PolicyBot(object):
    def __init__(self, input_size, weight_size, output_size, output_space, learning_rate, discount_factor):
        self.input_size = input_size
        self.weight_size = weight_size
        self.output_size = output_size
        self.output_space = output_space
        self.weights = np.random.randint(0, self.weight_size, (input_size, output_size))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def policy(self, state):
        x = state.dot(self.weights)
        exp = np.exp(x)
        return exp / np.sum(exp)

    def choose_act(self, probs):
        return random.choices(self.output_space, weights=probs, k=1)

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros(len(rewards))
        cumulative_rewards = 0
        for i in reversed(range(0, len(rewards))):
            cumulative_rewards = cumulative_rewards * self.discount_factor + rewards[i]
            discounted_rewards[i] = cumulative_rewards

        return discounted_rewards


pbot = PolicyBot(100, 2, 3, [0, 1, 2], 0.03, 0.6)
y = np.random.random(100)
for _ in range(100):
    x = pbot.policy(y)
    print(x)
    print(sum(x))
    print(pbot.choose_act(x))
