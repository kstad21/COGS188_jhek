import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_dim, hidden_layers, activations):
        super(DQN, self).__init__()

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activations[i] == 'relu':
                layers.append(nn.ReLU())
            elif activations[i] == 'linear':  
                pass 
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class DQNAgent:
    def __init__(self, state_size, mem_size=10000, discount=0.95,
                 epsilon=1, epsilon_min=0, epsilon_stop_episode=0,
                 n_neurons=[32, 32], activations=['relu', 'relu', 'linear'],
                 loss_fn=nn.MSELoss(), optimizer_fn=optim.Adam, replay_start_size=None, modelFile=None):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.mem_size = mem_size
        self.memory = deque(maxlen=mem_size)
        self.discount = discount

        if epsilon_stop_episode > 0:
            self.epsilon = epsilon
            self.epsilon_min = epsilon_min
            self.epsilon_decay = (self.epsilon - self.epsilon_min) / epsilon_stop_episode
        else:
            self.epsilon = 0

        self.n_neurons = n_neurons
        self.activations = activations
        self.loss_fn = loss_fn
        self.optimizer_fn = optimizer_fn

        if not replay_start_size:
            replay_start_size = mem_size // 2
        self.replay_start_size = replay_start_size

        if modelFile is not None:
            self.model = DQN(state_size, n_neurons, activations).to(self.device)
            self.model.load_state_dict(torch.load(modelFile, map_location=self.device))
            self.model.eval()
        else:
            self.model = DQN(state_size, n_neurons, activations).to(self.device)

        self.optimizer = self.optimizer_fn(self.model.parameters())

    def add_to_memory(self, current_state, next_state, reward, done):
        self.memory.append((current_state, next_state, reward, done))

    def predict_value(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        return self.model(state).item()

    def best_state(self, states):
        max_value = None
        best_state = None

        if random.random() <= self.epsilon:
            return random.choice(list(states))

        with torch.no_grad():
            for state in states:
                value = self.predict_value(state)
                if max_value is None or value > max_value:
                    max_value = value
                    best_state = state

        return best_state

    def train(self, batch_size=32, epochs=3):
        if batch_size > len(self.memory):
            return

        if len(self.memory) < self.replay_start_size:
            return

        batch = random.sample(self.memory, batch_size)

        states = np.array([x[0] for x in batch])
        next_states = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        dones = np.array([x[3] for x in batch])

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            next_qs = self.model(next_states).squeeze()

        target_qs = rewards + (1 - dones) * self.discount * next_qs

        predicted_qs = self.model(states).squeeze()

        loss = self.loss_fn(predicted_qs, target_qs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)
