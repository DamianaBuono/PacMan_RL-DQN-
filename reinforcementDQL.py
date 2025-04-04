import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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
        indices = np.random.choice(len(self.buffer), batch_size)
        samples = [self.buffer[idx] for idx in indices]
        return samples

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q = self.out(x)
        return q


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.bonus_applied = False
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = 0.99
        self.learning_rate = 0.0003
        self.batch_size = 64
        self.target_update_freq = 5  # frequenza per aggiornamento hard della rete target

        self.stall_counter = 0
        self.stall_threshold = 500

        # Parametri per esplorazione ε-greedy
        self.epsilon = 0.2  # probabilità iniziale di scegliere un'azione casuale
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        # Per monitorare la ripetizione delle azioni
        self.action_history = []

        # Reti: principale e target
        self.model = DQN(state_size, action_size).to(device)
        self.target_model = DQN(state_size, action_size).to(device)
        self.hard_update_target_model()  # aggiornamento iniziale completo
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss()

        self.memory = ReplayBuffer(capacity=10000)
        self.step = 0
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50000, gamma=0.99)

        self.target_berries = 70  # bacche necessarie per completare l'episodio
        self.collected_berries = 0
        self.visited_positions = set()

        self.step_penalty = 0.1
        self.temperature = 1.0

    def hard_update_target_model(self):
        # Aggiornamento hard: copia completa dei pesi dalla rete principale a quella target
        self.target_model.load_state_dict(self.model.state_dict())

    def soft_update_target_model(self, tau=0.001):
        # Aggiornamento soft: miscela dei pesi
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def remember(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, float(done)))

    def act(self, state):
        # Assicurati che state sia un array monodimensionale
        state = np.array(state).flatten()
        # Gli ultimi 5 elementi sono: [walls_up, walls_down, walls_left, walls_right, n_bacche]
        wall_up, wall_down, wall_left, wall_right = state[-5:-1]

        # Determina le azioni invalide in base ai muri
        invalid_actions = []
        if wall_up:
            invalid_actions.append(0)
        if wall_down:
            invalid_actions.append(1)
        if wall_left:
            invalid_actions.append(2)
        if wall_right:
            invalid_actions.append(3)

        # Prepara lo stato per la rete neurale
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze()

        # Calcola la softmax sui valori Q
        probabilities = torch.softmax(q_values / self.temperature, dim=0).cpu().numpy()

        # Ottieni solo le azioni valide
        valid_indices = [i for i in range(self.action_size) if i not in invalid_actions]
        if not valid_indices:
            valid_indices = list(range(self.action_size))

        # Strategia ε-greedy: con probabilità epsilon scegli un'azione casuale
        if np.random.rand() < self.epsilon:
            action = np.random.choice(valid_indices)
        else:
            valid_probabilities = probabilities[valid_indices]
            if valid_probabilities.sum() > 0:
                valid_probabilities /= valid_probabilities.sum()
            else:
                valid_probabilities = np.ones(len(valid_indices)) / len(valid_indices)
            action = np.random.choice(valid_indices, p=valid_probabilities)

        # Controllo per azioni ripetitive: se le ultime 3 azioni sono uguali, forza un'azione casuale
        self.action_history.append(action)
        if len(self.action_history) > 3:
            self.action_history.pop(0)
        if len(self.action_history) == 3 and (self.action_history[0] == self.action_history[1] == self.action_history[2]):
            action = np.random.choice(valid_indices)
            self.action_history = []  # resetta la cronologia

        self.last_action = action

        # Decay della probabilità di esplorazione
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
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
            next_actions = torch.argmax(self.model(next_states), dim=1, keepdim=True)
            next_q_values = self.target_model(next_states)
            next_qs = next_q_values.gather(1, next_actions).squeeze(1)
            targets = rewards + self.gamma * next_qs * (1 - dones)

        current_qs_taken = current_qs.gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = self.criterion(current_qs_taken, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1

        # Aggiornamento della rete target:
        if self.step % self.target_update_freq == 0:
            self.hard_update_target_model()
        else:
            self.soft_update_target_model(tau=0.001)

        return loss.item()

    def calculate_reward(self, game):
        reward = 0
        pacman = game.player.sprite
        pos = (pacman.rect.x, pacman.rect.y)
        reward -= self.step_penalty

        # Se la posizione è già stata visitata, aumenta il contatore di stallo
        if pos in self.visited_positions:
            self.stall_counter += 1
            if self.stall_counter > self.stall_threshold:
                reward -= 5 # Penalità per loop prolungato
                self.stall_counter = 0
        else:
            reward += 1
            self.visited_positions.add(pos)
            self.stall_counter = 0  # resetta il contatore se viene trovata una posizione nuova

        # Gestione delle bacche
        for berry in game.berries.sprites():
            if pacman.rect.colliderect(berry.rect):
                if berry.power_up:
                    reward += 50  # Bonus per power-up
                    pacman.immune = True
                    pacman.immune_time = 150
                    pacman.pac_score += 50
                else:
                    reward += 30  # Bonus per bacca normale
                    pacman.pac_score += 10
                pacman.n_bacche += 1
                self.collected_berries += 1
                berry.kill()

        # Gestione del singolo fantasma
        ghost = game.ghost
        if ghost and pacman.rect.colliderect(ghost.rect):
            if pacman.immune:
                reward += 100
                ghost.move_to_start_pos()
            else:
                reward -= 20


        return int(reward)

    def finalize_episode(self, pacman):
        if not self.bonus_applied:
            if pacman.n_bacche >= self.target_berries and pacman.life > 0:
                bonus = 150
                print("Obiettivo bacche raggiunto: reward bonus +150")
            elif pacman.life <= 0 and pacman.n_bacche < self.target_berries:
                bonus = -50
                print("Pacman morto senza raccogliere abbastanza bacche: penalità -50")
            else:
                bonus = 0
            self.bonus_applied = True
            return bonus
        return 0

    def reset_episode(self):
        self.bonus_applied = False
        self.collected_berries = 0
        self.visited_positions = set()
        self.stall_counter = 0
        self.action_history = []  # resetta la cronologia delle azioni
