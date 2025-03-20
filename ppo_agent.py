import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PPOAgent:
    def __init__(self, state_size, action_dim, 
                 n_neurons=[128, 128, 64, 32], 
                 activations=['relu', 'relu', 'relu', 'relu'],
                 lr=1e-3, gamma=0.99, clip_epsilon=0.2, update_epochs=4, batch_size=32, 
                 dropout_rate=0.2, modelFile=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate

        self.actor = self._build_network(state_size, n_neurons, activations, output_dim=action_dim)
        self.critic = self._build_network(state_size, n_neurons, activations, output_dim=1)
        self.actor.to(self.device)
        self.critic.to(self.device)

        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.memory = []

        if modelFile is not None:
            self.load_model(modelFile)

    def _build_network(self, input_dim, hidden_layers, activations, output_dim):
        layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activations[i] == 'relu':
                layers.append(nn.ReLU())
            elif activations[i] == 'tanh':
                layers.append(nn.Tanh())
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def select_action(self, state, valid_action_indices):
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        logits = self.actor(state_tensor).squeeze()
        mask = torch.full((self.action_dim,), -1e8).to(self.device)
        valid_mask = torch.zeros(self.action_dim, dtype=torch.bool).to(self.device)
        valid_mask[valid_action_indices] = True
        masked_logits = torch.where(valid_mask, logits, mask)
        probs = torch.softmax(masked_logits, dim=-1)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        value = self.critic(state_tensor)
        return action.item(), log_prob, value, probs.detach().cpu().numpy()

    def store_transition(self, transition):
        self.memory.append(transition)

    def finish_episode(self):
        returns = []
        R = 0
        for (_, _, _, _, reward, done) in reversed(self.memory):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        states = [t[0] for t in self.memory]
        valid_actions_list = [t[1] for t in self.memory]
        actions = [t[2] for t in self.memory]
        old_log_probs = torch.tensor([t[3].item() for t in self.memory], dtype=torch.float32).to(self.device)

        for _ in range(self.update_epochs):
            total_actor_loss = 0
            total_critic_loss = 0
            total_entropy = 0

            for i in range(len(self.memory)):
                state = torch.tensor(states[i], dtype=torch.float32).to(self.device).unsqueeze(0)
                valid_indices = valid_actions_list[i]
                action = torch.tensor(actions[i]).to(self.device)

                logits = self.actor(state).squeeze()
                mask = torch.full((self.action_dim,), -1e8).to(self.device)
                valid_mask = torch.zeros(self.action_dim, dtype=torch.bool).to(self.device)
                valid_mask[valid_indices] = True
                masked_logits = torch.where(valid_mask, logits, mask)
                probs = torch.softmax(masked_logits, dim=-1)
                m = torch.distributions.Categorical(probs)
                new_log_prob = m.log_prob(action)
                ratio = torch.exp(new_log_prob - old_log_probs[i])

                value = self.critic(state).squeeze()
                advantage = returns[i] - value.detach()

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage
                actor_loss = -torch.min(surr1, surr2)
                critic_loss = self.loss_fn(value, returns[i])
                entropy = m.entropy()

                total_actor_loss += actor_loss
                total_critic_loss += critic_loss
                total_entropy += entropy

            n = len(self.memory)
            loss = (total_actor_loss / n) + 0.5 * (total_critic_loss / n) - 0.01 * (total_entropy / n)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.memory = []

    def save_model(self, filename):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict()
        }, filename)

    def load_model(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
