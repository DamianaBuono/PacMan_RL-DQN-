import numpy as np
import random
import pickle
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# Rete neurale semplice
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.model(x)

# Agente DQN
class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        lr=0.001,
        batch_size=64,
        memory_size=100000
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.visited_positions = set()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = QNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.last_loss = 0

    def act(self, state):
        if isinstance(state, np.ndarray) and state.ndim == 2:
            state = state[0]
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return int(torch.argmax(q_values).item())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        # Se non ci sono abbastanza esperienze, esci restituendo None
        if len(self.memory) < self.batch_size:
            return None

        # Estrai un minibatch casuale
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Costruisci tensori su device
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Metti il modello in training
        self.model.train()

        # Q(s, a) correnti
        q_values = self.model(states_tensor).gather(1, actions_tensor)

        # Calcola target Q usando la stessa rete (per semplicità)
        with torch.no_grad():
            max_next_q_values = self.model(next_states_tensor).max(1, keepdim=True)[0]
            target_q_values = rewards_tensor + (1 - dones_tensor) * self.gamma * max_next_q_values

        # MSE loss
        loss = self.loss_fn(q_values, target_q_values)
        self.last_loss = loss.item()

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return self.last_loss

    def reset_episode(self):
        self.last_loss = 0
        self.visited_positions.clear()

    def save(self, filename="dqn_model.pth"):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'epsilon': self.epsilon
        }, filename)
        print(f"Modello salvato in {filename}")

    def load(self, filename="dqn_model.pth"):
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        print(f"Modello caricato da {filename}: epsilon={self.epsilon}")
