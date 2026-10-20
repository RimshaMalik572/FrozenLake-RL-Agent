
import numpy as np


class QLearningAgent:
    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.8,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.9995,
        min_epsilon=0.01,
    ):
        self.state_size = state_size
        self.action_size = action_size

        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Q-table: rows represent states, columns represent actions
        self.q_table = np.zeros((state_size, action_size))

    def choose_action(self, state, training=True):
        # Explore: choose a random action
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)

        # Exploit: choose the action with the highest Q-value
        return int(np.argmax(self.q_table[state]))

    def learn(self, state, action, reward, next_state, terminated):
        old_value = self.q_table[state, action]

        if terminated:
            next_max = 0.0
        else:
            next_max = np.max(self.q_table[next_state])

        # Q-Learning update rule
        new_value = old_value + self.lr * (
            reward + self.gamma * next_max - old_value
        )

        self.q_table[state, action] = new_value

    def decay_epsilon(self):
        self.epsilon = max(
            self.min_epsilon,
            self.epsilon * self.epsilon_decay,
        )

    def save(self, filename="q_table.npy"):
        np.save(filename, self.q_table)

    def load(self, filename="q_table.npy"):
        self.q_table = np.load(filename)
