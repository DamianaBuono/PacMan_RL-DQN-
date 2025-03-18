import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


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


class PrioritizedReplayBuffer:
    """Replay buffer con Prioritized Experience Replay (PER)"""

    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.memory = []
        self.priorities = []
        self.alpha = alpha  # Fattore di priorità

    def add(self, state, action, reward, next_state, done, error):
        """Aggiunge un'esperienza con una priorità basata sull'errore TD"""
        priority = (error + 1e-5) ** self.alpha  # Evita priorità zero
        if len(self.memory) < self.capacity:
            self.memory.append((state, action, reward, next_state, done))
            self.priorities.append(priority)
        else:
            self.memory.pop(0)
            self.priorities.pop(0)
            self.memory.append((state, action, reward, next_state, done))
            self.priorities.append(priority)

    def sample(self, batch_size, beta=0.4):
        """Seleziona un minibatch in base alle priorità"""
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()  # Probabilità normalizzate
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        experiences = [self.memory[i] for i in indices]

        # Calcolo dei pesi di importanza per correggere il bias di priorità
        weights = (len(self.memory) * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalizzazione

        return experiences, torch.FloatTensor(weights).to(device), indices

    def update_priorities(self, indices, errors):
        """Aggiorna le priorità delle esperienze nel buffer"""
        for i, error in zip(indices, errors):
            self.priorities[i] = (error + 1e-5) ** self.alpha


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = PrioritizedReplayBuffer(capacity=5000)
        self.gamma = 0.99
        self.learning_rate = 0.0005
        self.batch_size = 64
        self.target_update_freq = 10  # Aggiornamento ogni 10 episodi

        self.tau = 1.0
        self.tau_min = 0.5
        self.tau_decay = 0.999

        self.model = DQN(state_size, action_size).to(device)
        self.target_model = DQN(state_size, action_size).to(device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def update_target_model(self):
        """Aggiorna la rete target con i pesi della rete principale"""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Memorizza le transizioni con priorità basata sull'errore TD"""
        state_t = torch.FloatTensor(state).to(device)
        next_state_t = torch.FloatTensor(next_state).to(device)

        # Calcola il valore Q target per ottenere l'errore TD
        target = reward + (1 - done) * self.gamma * torch.max(self.target_model(next_state_t)).item()

        # Ottieni la previsione dalla rete per tutte le azioni
        prediction = self.model(state_t)

        # Ottieni il valore per l'azione specifica
        prediction_value = prediction[0][action].item()  # prediction[0] perché `state_t` è un batch di dimensione 1

        error = abs(target - prediction_value)  # Errore assoluto come misura di priorità

        # Aggiungi al buffer con priorità
        self.memory.add(state, action, reward, next_state, done, error)

    def act(self, state):
        """Seleziona un'azione usando Boltzmann Exploration"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.model(state)

        # Softmax sulle Q-values per ottenere probabilità
        probabilities = F.softmax(q_values / max(self.tau, 1e-2), dim=1).cpu().numpy().squeeze()

        # Normalizzazione per evitare errori numerici
        probabilities /= probabilities.sum()

        # Selezione dell'azione in base alle probabilità
        action = np.random.choice(self.action_size, p=probabilities)

        return action

    def replay(self):
        """Allenamento con Prioritized Experience Replay"""
        if len(self.memory.memory) < self.batch_size:
            return

        beta = 0.4  # Fattore di correzione del bias di priorità
        minibatch, weights, indices = self.memory.sample(self.batch_size, beta)

        states, targets, errors = [], [], []

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            state_t = torch.FloatTensor(state).to(device)
            next_state_t = torch.FloatTensor(next_state).to(device)

            target = reward
            if not done:
                target += self.gamma * torch.max(self.target_model(next_state_t)).item()

            # Predizione per tutte le azioni
            target_f = self.model(state_t).detach().cpu().numpy().squeeze()

            # Calcola l'errore TD
            error = abs(target - target_f[action])  # Ricalcola errore TD
            errors.append(error)

            # Aggiorna solo l'azione selezionata
            target_f[action] = target
            states.append(state_t.cpu().numpy())
            targets.append(target_f)

        states = torch.FloatTensor(np.array(states)).to(device)
        targets = torch.FloatTensor(np.array(targets)).to(device)

        self.optimizer.zero_grad()
        predictions = self.model(states)

        # Assicurati che predictions e targets abbiano la stessa forma
        predictions = predictions.view(-1, self.action_size)  # Ridimensiona predictions se necessario
        targets = targets.view(-1, self.action_size)  # Assicurati che target abbia la forma corretta

        # Calcola la perdita pesata con le priorità (Importanza IS)
        loss = (weights * self.criterion(predictions, targets)).mean()
        loss.backward()
        self.optimizer.step()

        # Aggiorna le priorità nel buffer
        self.memory.update_priorities(indices, errors)

        # Decadimento della temperatura tau
        if self.tau > self.tau_min:
            self.tau *= self.tau_decay
