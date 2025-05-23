import numpy as np
import random
import pickle
from collections import defaultdict

class SarsaAgent:
    def __init__(
        self,
        action_size,
        buckets=(8,) * 14,
        alpha=0.01,
        gamma=0.80,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    ):
        self.action_size = action_size
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        self.last_loss = 0
        self.buckets = buckets
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.state_bounds = [
            (0.0, 1.0),  # norm_x
            (0.0, 1.0),  # norm_y
            (-1.0, 1.0), # norm_dx
            (-1.0, 1.0), # norm_dy
            (0.0, 1.0),  # ghost_dist
            (-1.0, 1.0), # ghost_dx
            (-1.0, 1.0), # ghost_dy
            (0.0, 1.0),  # berry_dist
            (0, 1),      # immune
            (0, 1),      # wall_up
            (0, 1),      # wall_down
            (0, 1),      # wall_left
            (0, 1),      # wall_right
            (0, 1)       # danger_flag
        ]

    def _discretize_state(self, state):
        discretized = []
        for i, value in enumerate(state):
            low, high = self.state_bounds[i]
            if high == low:
                idx = 0
            else:
                ratio = (value - low) / (high - low)
                idx = int(ratio * self.buckets[i])
                idx = min(self.buckets[i] - 1, max(0, idx))
            discretized.append(idx)
        return tuple(discretized)

    def act(self, state):
        if isinstance(state, np.ndarray) and state.ndim == 2:
            state = state[0]
        s = self._discretize_state(state)
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return int(np.argmax(self.q_table[s]))

    def remember(self, state, action, reward, next_state, next_action, done):
        s = self._discretize_state(state)
        s_next = self._discretize_state(next_state)

        # SARSA: usa l'azione effettivamente scelta (on-policy)
        q_next = self.q_table[s_next][next_action] if not done else 0.0
        td_target = reward + self.gamma * q_next
        td_error = td_target - self.q_table[s][action]

        self.q_table[s][action] += self.alpha * td_error
        self.last_loss = td_error ** 2

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset_episode(self):
        self.last_loss = 0

    def save(self, filename="sarsa_model.pkl"):
        data = {
            "q_table": dict(self.q_table),
            "alpha": self.alpha,
            "epsilon": self.epsilon
        }
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        print(f"Modello SARSA salvato in {filename}")

    def load(self, filename="sarsa_model.pkl"):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        self.q_table = defaultdict(lambda: np.zeros(self.action_size), data["q_table"])
        self.alpha = data.get("alpha", self.alpha)
        self.epsilon = data.get("epsilon", self.epsilon)
        print(f"Modello SARSA caricato da {filename}: alpha={self.alpha}, epsilon={self.epsilon}")
