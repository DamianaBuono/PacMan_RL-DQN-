
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

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


# Implementazione di NoisyLinear secondo il paper "Noisy Networks for Exploration"
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Pesi deterministici
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        # Bias deterministici
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.register_buffer('weight_epsilon', torch.zeros(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.zeros(out_features))
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init)

    def reset_noise(self):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(input, weight, bias)


# QNetwork definito in modo da avere la stessa architettura del checkpoint
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        #self.ln1 = nn.LayerNorm(256)
        #self.fc2 = nn.Linear(256, 256)
        #self.ln2 = nn.LayerNorm(256)
        #self.fc3 = nn.Linear(256, action_size)
        self.fc2 = NoisyLinear(256, 256)  # layer intermedio: 256 -> 256
        self.fc3 = NoisyLinear(256, action_size)  # da 25


    def forward(self, x):
        x = self.fc1(x)
        #x = self.ln1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        #x = self.ln2(x)
        x = torch.relu(x)
        q = self.fc3(x)
        return q

    def reset_noise(self):
        if hasattr(self.fc2, 'reset_noise'):
            self.fc2.reset_noise()
        if hasattr(self.fc3, 'reset_noise'):
            self.fc3.reset_noise()

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = 0.99
        self.learning_rate = 0.001
        self.batch_size = 64
        self.target_update_freq = 100

        self.action_history = []
        self.visited_positions = deque(maxlen=20)

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.model = QNetwork(state_size, action_size).to(device)
        self.target_model = QNetwork(state_size, action_size).to(device)
        self.hard_update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss()

        self.memory = ReplayBuffer(capacity=10000)
        self.step = 0
        self.target_berries = 98

    def hard_update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def soft_update_target_model(self, tau=0.005):
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def remember(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, float(done)))

    def act(self, state):
        state = np.array(state).flatten()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        self.model.reset_noise()
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze()
        if was_training:
            self.model.train()

        q_values = q_values.cpu().numpy()

        # Estraggo le informazioni dei muri dallo stato (assumo che siano alle posizioni 13-16)
        wall_up, wall_down, wall_left, wall_right = state[13:17]

        if wall_up:
            q_values[0] = -np.inf
        if wall_down:
            q_values[1] = -np.inf
        if wall_left:
            q_values[2] = -np.inf
        if wall_right:
            q_values[3] = -np.inf


        action = int(np.argmax(q_values))
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

        self.model.reset_noise()
        current_qs = self.model(states)
        with torch.no_grad():
            self.target_model.reset_noise()
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
            self.soft_update_target_model()
        return loss.item()

    def reset_episode(self):
        self.visited_positions.clear()
        self.action_history = []
