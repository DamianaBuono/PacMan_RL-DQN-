import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha  # quanto pesare le priorità
        self.buffer = []
        self.pos = 0
        # Inizialmente, le priorità saranno impostate al massimo, così da essere campionate
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def add(self, experience):
        # Se il buffer non è ancora pieno, aggiungi la nuova esperienza
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.buffer)]

        # Calcola la probabilità di ciascuna esperienza (raised to alpha)
        probs = prios ** self.alpha
        probs /= probs.sum()

        # Campiona gli indici in base a queste probabilità
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # Calcola i pesi di importance sampling
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalizza per evitare valori troppo alti
        weights = np.array(weights, dtype=np.float32)

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
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


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = NoisyLinear(state_size, 128)
        self.fc2 = NoisyLinear(128, 128)
        self.fc3 = NoisyLinear(128, 64)
        self.fc4 = NoisyLinear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.bonus_applied = None
        self.state_size = state_size
        self.action_size = action_size

        # Parametri dell'algoritmo
        self.gamma = 0.99
        self.learning_rate = 0.001  # <-- Learning rate aggiornato
        self.batch_size = 128       # <-- Batch size maggiore
        self.target_update_freq = 10

        self.stall_counter = 0
        self.stall_threshold = 20

        self.double_dqn = True

        # Costruzione dei modelli (rete online e target) usando NoisyNet
        self.model = DQN(state_size, action_size).to(device)
        self.target_model = DQN(state_size, action_size).to(device)
        self.update_target_model()  # aggiornamento soft iniziale
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss()

        # Sostituisco il deque con il Prioritized Replay Buffer
        self.memory = PrioritizedReplayBuffer(capacity=10000, alpha=0.6)
        self.step = 0

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.9)

        self.target_berries = 1000  # bacche necessarie per completare l'episodio
        self.collected_berries = 0

        # Parametri PER
        self.beta_start = 0.4
        self.beta_frames = 100000  # numero di frame per annealing di beta

    def update_target_model(self, tau=0.01):
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def remember(self, state, action, reward, next_state, done):
        # Aggiunge la transizione con la massima priorità attuale
        self.memory.add((state, action, reward, next_state, float(done)))

    def act(self, state):
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        self.model.reset_noise()
        with torch.no_grad():
            q_values = self.model(state).squeeze()
        action = torch.argmax(q_values).item()
        return action

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0  # non ci sono abbastanza esperienze

        # Aggiorno beta in base al numero di step
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * self.step / self.beta_frames)

        minibatch, indices, weights = self.memory.sample(self.batch_size, beta=beta)

        states = torch.FloatTensor(np.array([m[0] for m in minibatch])).to(device).squeeze(1)
        actions = torch.LongTensor(np.array([m[1] for m in minibatch])).to(device)
        rewards = torch.FloatTensor(np.array([m[2] for m in minibatch])).to(device)
        next_states = torch.FloatTensor(np.array([m[3] for m in minibatch])).to(device).squeeze(1)
        dones = torch.FloatTensor(np.array([m[4] for m in minibatch])).to(device)

        if dones.ndim > 1:
            dones = dones[:, 0]

        rewards = rewards.view(-1)
        dones = dones.view(-1)
        weights = torch.FloatTensor(weights).to(device)

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
        td_errors = torch.abs(current_qs_taken - targets).detach().cpu().numpy()
        loss = (self.criterion(current_qs_taken, targets) * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        self.step += 1
        self.update_target_model()  # aggiornamento soft continuo

        # Aggiorna le priorità dei campioni in base all'errore TD
        new_priorities = td_errors + 1e-6  # aggiungi un piccolo offset per evitare priorità zero
        self.memory.update_priorities(indices, new_priorities)

        return loss.item()

    def reset_episode(self):
        self.collected_berries = 0
        self.bonus_applied = False  # Flag per controllare l'applicazione del bonus/penalità

    def calculate_reward(self, game):
        reward = 0
        pacman = game.player.sprite

        berries = game.berries.sprites()

        for berry in berries:
            if pacman.rect.colliderect(berry.rect):
                if berry.power_up:
                    pacman.immune = True
                    pacman.immune_time = 150
                    pacman.pac_score += 50
                    reward += 50  # bonus per raccolta power-up
                    print("Bacca power-up raccolta: bonus +50")
                else:
                    pacman.pac_score += 10
                    reward += 10  # bonus per bacca normale
                    print("Bacca normale raccolta: bonus +10")
                pacman.n_bacche += 1
                self.collected_berries += 1
                berry.kill()

        if not self.bonus_applied:
            if self.collected_berries >= self.target_berries and pacman.life > 0:
                reward += 150
                self.bonus_applied = True
                print("Obiettivo bacche raggiunto: reward bonus +150")
            elif pacman.life <= 0 and self.collected_berries < self.target_berries:
                reward -= 50
                self.bonus_applied = True
                print("Pacman morto senza raccogliere abbastanza bacche: penalità -50")

        for ghost in game.ghosts.sprites():
            distance = game.get_distance(pacman.rect.center, ghost.rect.center)
            if pacman.immune:
                if hasattr(pacman, 'last_ghost_distance'):
                    ghost_distance_delta = pacman.last_ghost_distance - distance
                    if ghost_distance_delta > 0:
                        bonus = int(ghost_distance_delta * 2)
                        reward += bonus
                        print(
                            f"Inseguimento fantasma in power-up: distanza delta={ghost_distance_delta}, bonus={bonus}")
                pacman.last_ghost_distance = distance
            else:
                if distance < 100:
                    penalty = int((100 - distance) * 1)
                    reward -= penalty
                    print(f"Troppo vicino al fantasma: distanza={distance}, penalità={penalty}")

        return int(reward)
