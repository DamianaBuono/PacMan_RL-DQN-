import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Imposta il dispositivo (CPU o GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005
        self.batch_size = 64
        self.target_update_freq = 10  # Aggiornamento ogni 10 episodi

        self.model = DQN(state_size, action_size).to(device)
        self.target_model = DQN(state_size, action_size).to(device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def update_target_model(self):
        """Aggiorna la rete target con i pesi della rete principale"""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Memorizza le transizioni"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Decide un'azione (esplorazione o sfruttamento)"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self):
        """Allenamento con replay memory"""
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)

        states, targets = [], []

        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            target = reward

            if not done:
                target += self.gamma * torch.max(self.target_model(next_state)).item()

            # Ottenere i valori predetti e assicurarsi della forma corretta
            target_f = self.model(state).detach().cpu().numpy().squeeze()

            # Controllo di debug per garantire che target_f abbia la giusta dimensione
            if target_f.shape[0] != self.action_size:
                print(f"Errore: target_f ha dimensione {target_f.shape}, dovrebbe essere ({self.action_size},)")
                continue  # Salta l'iterazione se la dimensione non è corretta

            target_f[action] = target  # Aggiorna solo il valore relativo all'azione scelta

            states.append(state.cpu().numpy())
            targets.append(target_f)

        states = torch.FloatTensor(np.array(states)).to(device)
        targets = torch.FloatTensor(np.array(targets)).to(device)

        self.optimizer.zero_grad()
        predictions = self.model(states).squeeze(1)
        loss = self.criterion(predictions, targets)
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
