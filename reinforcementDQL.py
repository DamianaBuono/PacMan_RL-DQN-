import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)  # Aumentata capacità della rete
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
        self.memory = deque(maxlen=5000)  # Aumentata memoria
        self.gamma = 0.99  # Maggiore importanza ai futuri stati
        self.epsilon = 1.0  # Esplorazione iniziale
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005  # Ridotto per stabilità
        self.batch_size = 64  # Aumentato batch per stabilità

        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
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
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self):
        """Allenamento con replay memory"""
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)

        states, targets = [], []

        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            target = reward

            if not done:
                target += self.gamma * torch.max(self.target_model(next_state)).item()

            # Inizializza target_f come un array di dimensione (action_size)
            target_f = self.model(state).detach().numpy()
            target_f = target_f.flatten()  # Assicurati che sia un array 1D

            # Aggiorna solo l'azione scelta con il valore target
            target_f[action] = target

            # Aggiungi il target aggiornato alle liste
            states.append(state.numpy())
            targets.append(target_f)

        # Conversione a tensore
        states = torch.FloatTensor(np.array(states))
        targets = torch.FloatTensor(np.array(targets))

        # Ottimizzazione
        self.optimizer.zero_grad()
        predictions = self.model(states)
        loss = self.criterion(predictions, targets)
        loss.backward()
        self.optimizer.step()

        # Decadimento dell'epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

