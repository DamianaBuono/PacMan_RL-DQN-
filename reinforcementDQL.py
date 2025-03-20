import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.memory = []
        self.priorities = []
        self.alpha = alpha

    def add(self, state, action, reward, next_state, done, error):
        priority = (error + 1e-5) ** self.alpha
        if len(self.memory) < self.capacity:
            self.memory.append((state, action, reward, next_state, done))
            self.priorities.append(priority)
        else:
            self.memory.pop(0)
            self.priorities.pop(0)
            self.memory.append((state, action, reward, next_state, done))
            self.priorities.append(priority)

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        experiences = [self.memory[i] for i in indices]

        weights = (len(self.memory) * probs[indices]) ** (-beta)
        weights /= weights.max()

        return experiences, torch.FloatTensor(weights).to(device), indices

    def update_priorities(self, indices, errors):
        for i, error in zip(indices, errors):
            self.priorities[i] = (error + 1e-5) ** self.alpha


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = PrioritizedReplayBuffer(capacity=5000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005
        self.batch_size = 64
        self.target_update_freq = 10
        self.stall_counter = 0
        self.stall_threshold = 50
        self.model = DQN(state_size, action_size).to(device)
        self.target_model = DQN(state_size, action_size).to(device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        state_t = torch.FloatTensor(state).to(device)
        next_state_t = torch.FloatTensor(next_state).to(device)
        target = reward + (1 - done) * self.gamma * torch.max(self.target_model(next_state_t)).item()
        prediction = self.model(state_t)
        prediction_value = prediction[0][action].item()
        error = abs(target - prediction_value)
        self.memory.add(state, action, reward, next_state, done, error)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self):
        if len(self.memory.memory) < self.batch_size:
            return 0  # Restituisce zero se non ci sono abbastanza esperienze

        beta = 0.4
        minibatch, weights, indices = self.memory.sample(self.batch_size, beta)
        states, targets, errors = [], [], []

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            state_t = torch.FloatTensor(state).to(device)
            next_state_t = torch.FloatTensor(next_state).to(device)
            target = reward
            if not done:
                target += self.gamma * torch.max(self.target_model(next_state_t)).item()

            target_f = self.model(state_t).detach().cpu().numpy().squeeze()
            error = abs(target - target_f[action])
            errors.append(error)

            target_f[action] = target
            states.append(state_t.cpu().numpy())
            targets.append(target_f)

        states = torch.FloatTensor(np.array(states)).to(device)
        targets = torch.FloatTensor(np.array(targets)).to(device)

        self.optimizer.zero_grad()
        predictions = self.model(states)
        predictions = predictions.view(-1, self.action_size)
        targets = targets.view(-1, self.action_size)
        loss = (weights * self.criterion(predictions, targets)).mean()
        loss.backward()
        self.optimizer.step()

        self.memory.update_priorities(indices, errors)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()  # Ritorna il valore della loss


    def calculate_reward(self, game):
        reward = 0

        if game.player.sprite.last_position == game.player.sprite.rect.topleft:
            self.stall_counter += 1
        else:
            self.stall_counter = 0

        if self.stall_counter >= self.stall_threshold:
            reward -= 10  # Penalità per stallo

        if game.player.sprite.last_position == game.player.sprite.rect.topleft:
            reward -= 2  # Penalità per il looping
        else:
            reward += 1  # Ricompensa per il movimento sicuro
        game.player.sprite.last_position = game.player.sprite.rect.topleft

        for berry in game.berries.sprites():
            if game.player.sprite.rect.colliderect(berry.rect):
                reward += 50 if berry.power_up else 10
                berry.kill()

        for ghost in game.ghosts.sprites():
            distance = game.get_distance(game.player.sprite.rect.center, ghost.rect.center)

            if game.player.sprite.rect.colliderect(ghost.rect):
                if not game.player.sprite.immune:
                    reward -= 50
                else:
                    ghost.move_to_start_pos()
                    reward += 100
                    game.player.sprite.combo_counter += 1
                    reward += 50 * game.player.sprite.combo_counter

            elif distance < 50:
                reward += 2
            elif distance > 100:
                reward += 1

        if not game.berries and game.player.sprite.life > 0:
            reward += 100

        if game.game_over:
            reward -= 100

        return reward
