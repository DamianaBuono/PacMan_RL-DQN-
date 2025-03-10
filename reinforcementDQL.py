import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Hyperparametri
GAMMA = 0.99
LEARNING_RATE = 0.001
MEMORY_SIZE = 50000
BATCH_SIZE = 32
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.995


# Definizione della rete neurale per DQN
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Classe per l'agente DQN
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = EPSILON_START
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.MSELoss()
        self.update_target_network()

    def update_target_network(self, tau=0.1):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0)

        if random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
            q_values = None
        else:
            with torch.no_grad():
                q_values = self.model(state)
            action = torch.argmax(q_values).item()

        print(
            f"Azione scelta: {action}, Q-values: {q_values.cpu().numpy() if q_values is not None else 'Random action'}")  # Debug
        return action

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * GAMMA * next_q_values

        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)


#def save_model(self, filename="pacman_dqn.pth"):
    #torch.save(self.model.state_dict(), filename)
     #print(f"Modello salvato in {filename}")

#def load_model(self, filename="pacman_dqn.pth"):
    #self.model.load_state_dict(torch.load(filename))
    #self.model.eval()
    #print(f"Modello caricato da {filename}")