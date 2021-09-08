import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
from PolicyNetwork import PolicyGradientNetwork
from tensorflow.keras.models import load_model

class Agent:
    def __init__(self, alpha = 0.003,input_dims = 6, gamma = 0.95, n_actions = 4, fc1_dims = 256, fc2_dims = 256):
        self.gamma = gamma
        self.alpha = alpha
        self.n_actions = n_actions
        self.states = []
        self.rewards = []
        self.actions = []
        self.policy = PolicyGradientNetwork(n_actions = n_actions)
        self.policy.compile(optimizer = Adam(learning_rate = self.gamma))
        if os.path.exists('my_model_weights.h5'):
            self.loadModel()

    def choose_action(self, obs):
        state = tf.convert_to_tensor([obs], dtype = tf.float32)
        probs = self.policy(state)
        action_probs = tfp.distributions.Categorical(probs = probs)
        action = action_probs.sample()
        print(action)
        return action.numpy()[0]

    def store_transition(self, obs, action, reward):
        self.states.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)

    def learn(self):
        actions = tf.convert_to_tensor(self.actions, dtype = tf.float32)
        rewards = tf.convert_to_tensor(self.rewards)

        G = np.zeros_like(rewards)
        for t in range(len(rewards)):
            G_sum = 0
            discount = 1
            for k in range(t, len(rewards)):
                G_sum += float(rewards[k]) * discount
                discount *= self.gamma
            G[t] = G_sum

        with tf.GradientTape() as tape:
            loss = 0
            for idx, (g, state) in enumerate(zip(G, self.states)):
                state = tf.convert_to_tensor([state], dtype = tf.float32)
                probs = self.policy(state)
                action_probs = tfp.distributions.Categorical(probs = probs)
                log_prob = action_probs.log_prob(actions[idx])
                loss += -g * tf.squeeze(log_prob)
        gradient = tape.gradient(loss, self.policy.trainable_variables)
        self.policy.optimizer.apply_gradients(zip(gradient, self.policy.trainable_variables))

        self.states = []
        self.actions = []
        self.rewards = []
    def saveModel(self):
        self.policy.save_weights('my_model_weights.h5')

    def loadModel(self):
        self.policy.load_weights('my_model_weights.h5')
