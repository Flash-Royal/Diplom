import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.python.framework.ops import disable_eager_execution
import numpy as np
disable_eager_execution()

class Agent(object):
    def __init__(self, alpha, gamma, n_actions = 4, layer1_size = 16, layer2_size = 16, input_dims = 128, fname = 'model_weights.h5'):
        self.gamma = gamma
        self.alpha = alpha
        self.G = 0
        self.input_dims = input_dims
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        self.n_actions = n_actions
        self.states = []
        self.actions = []
        self.rewards = []

        self.policy, self.predict = self.build_policy_network()
        self.action_space = [i for i in range(n_actions)]
        self.model_file = fname

    def build_policy_network(self):
        input = Input(shape = (self.input_dims,))
        advantages = Input(shape = [1])
        dense1 = Dense(self.fc1_dims, activation = 'relu')(input)
        dense2 = Dense(self.fc2_dims, activation = 'relu')(dense1)
        probs = Dense(self.n_actions, activation = 'softmax')(dense2)

        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1 - 1e-8)
            log_lik = y_true * K.log(out)

            return K.sum(-log_lik * advantages)

        policy = Model([input, advantages], [probs])
        policy.compile(optimizer = Adam(lr = self.alpha), loss = custom_loss)
        predict = Model([input], [probs])

        return policy, predict

    def choose_action(self, obs):
        state = np.array([obs])
        # print("state = ", state)
        probabilities = self.predict.predict(state)[0]
        # print("prob = ", probabilities)
        action = np.random.choice(self.action_space, p = probabilities)
        # print("action = ", action)
        return action

    def store_transition(self, obs, action, reward):
        self.states.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)

    def learn(self):
        actions = np.array(self.actions)
        states = np.array(self.states)
        rewards = np.array(self.rewards)

        actions1 = np.zeros([len(actions), self.n_actions])
        actions1[np.arange(len(actions)), actions] = 1

        G = np.zeros_like(rewards)
        for t in range(len(rewards)):
            G_sum = 0
            discount = 1
            for k in range(t, len(rewards)):
                G_sum += rewards[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        self.G = (G - mean) / std

        cost = self.policy.train_on_batch([states, self.G], actions1)

        self.states = []
        self.actions = []
        self.rewards = []

    def save_model(self):
        self.policy.save_weights(self.model_file)

    def load_model(self):
        self.policy.load_weights(self.model_file)
