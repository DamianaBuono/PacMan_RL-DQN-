import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from settings import HEIGHT, WIDTH
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def add(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        samples = [self.buffer[idx] for idx in indices]
        return samples

    def __len__(self):
        return len(self.buffer)


class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingDQN, self).__init__()
        # Layer condivisi per l'estrazione delle features
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)

        # Stream per il valore dello stato V(s)
        self.fc_value = nn.Linear(128, 64)
        self.value = nn.Linear(64, 1)

        # Stream per l'advantage A(s,a)
        self.fc_advantage = nn.Linear(128, 64)
        self.advantage = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        # Calcolo del valore
        value = torch.relu(self.fc_value(x))
        value = self.value(value)

        # Calcolo dell'advantage
        advantage = torch.relu(self.fc_advantage(x))
        advantage = self.advantage(advantage)

        # Combinazione: Q(s,a) = V(s) + (A(s,a) - media(A(s,·)))
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.bonus_applied = False
        self.state_size = state_size  # Es. 17 (16 feature + danger_flag)
        self.action_size = action_size
        self.double_dqn = True

        self.gamma = 0.99
        self.learning_rate = 0.0003
        self.batch_size = 64
        self.target_update_freq = 100

        self.stall_counter = 0
        self.stall_threshold = 500

        # Parametri per ε-greedy
        self.epsilon = 0.2
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        self.action_history = []  # per monitorare ripetizioni

        self.model = DuelingDQN(state_size, action_size).to(device)
        self.target_model = DuelingDQN(state_size, action_size).to(device)
        self.hard_update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss()

        self.memory = ReplayBuffer(capacity=10000)
        self.step = 0

        self.target_berries = 197
        self.collected_berries = 0
        self.visited_positions = set()

    def hard_update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def soft_update_target_model(self, tau=0.001):
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def remember(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, float(done)))

    def act(self, state):
        # Converte lo stato in un vettore flattenato
        state = np.array(state).flatten()
        # Utilizza le informazioni sugli ostacoli (ultime 4 feature fra i 6 finali)
        wall_up, wall_down, wall_left, wall_right = state[-6:-2]
        invalid_actions = []
        if wall_up:
            invalid_actions.append(0)
        if wall_down:
            invalid_actions.append(1)
        if wall_left:
            invalid_actions.append(2)
        if wall_right:
            invalid_actions.append(3)

        # Determina le azioni valide
        valid_actions = [i for i in range(self.action_size) if i not in invalid_actions]
        if not valid_actions:
            valid_actions = list(range(self.action_size))

        # Esplorazione ε-greedy: con probabilità epsilon sceglie azione casuale
        if np.random.rand() < self.epsilon:
            action = np.random.choice(valid_actions)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.model(state_tensor).squeeze()
            q_values = q_values.cpu().numpy()
            valid_qs = {i: q_values[i] for i in valid_actions}
            action = max(valid_qs, key=valid_qs.get)

        return action

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0

        minibatch = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(np.stack([np.squeeze(m[0]) for m in minibatch])).to(device)
        actions = torch.LongTensor(np.array([m[1] for m in minibatch])).to(device)
        rewards = torch.FloatTensor(np.array([m[2] for m in minibatch])).to(device)
        next_states = torch.FloatTensor(np.stack([np.squeeze(m[3]) for m in minibatch])).to(device)
        dones = torch.FloatTensor(np.array([m[4] for m in minibatch])).to(device)
        if dones.ndim > 1:
            dones = dones[:, 0]
        rewards = rewards.view(-1)
        dones = dones.view(-1)

        current_qs = self.model(states)
        with torch.no_grad():
            if self.double_dqn:
                next_actions = torch.argmax(self.model(next_states), dim=1, keepdim=True)
                next_q_values = self.target_model(next_states)
                next_qs = next_q_values.gather(1, next_actions).squeeze(1)
            else:
                next_qs, _ = self.target_model(next_states).max(dim=1)
            targets = rewards + self.gamma * next_qs * (1 - dones)

        current_qs_taken = current_qs.gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = self.criterion(current_qs_taken, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.step += 1

        if self.step % self.target_update_freq == 0:
            self.hard_update_target_model()
        else:
            self.soft_update_target_model(tau=0.05)

        # Decay epsilon ad ogni chiamata di replay oppure al termine dell'episodio
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        return loss.item()

    def reset_episode(self):
        self.bonus_applied = False
        self.collected_berries = 0
        self.visited_positions = set()
        self.stall_counter = 0
        self.action_history = []  # Resetta la cronologia delle azioni
