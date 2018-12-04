import gym
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop, Adam
from collections import deque


class DQNAgent:
    def __init__(self, env):
        self._env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action.n
        self.discount_factor = 0.99
        self.learning_rate = 0.0025
        self.epsilon = 1.
        self.epsilon_min = 0.05
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / 1e6
        self.batch_size = 32

        self.memory = deque(maxlen=500000)

        self.episode = 0
        self.time_step = 0

        self.train_frequency = 4
        self.train_start = 10e3
        self.update_target_frequency = 10000
        self.eval_frequency = 250e3
        self.eval_frames = 200e3

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_normal'))
        rmsprop = RMSprop(lr=self.learning_rate)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)
        model.summary()

        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_epsilon(self):
        return self.epsilon - self.epsilon_decay * self.time_step

    def get_action(self, state):
        if random.random() < self.get_epsilon():
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    def save_experience(self, exp):
        self.memory.append(exp)

    def train(self):
        if len(self.memory) < self.train_start:
            return

        if self.time_step % self.train_frequency == 0:

            mini_batch = random.sample(self.memory, self.batch_size)

            update_input = np.zeros([self.batch_size, self.state_size])
            update_output = np.zeros([self.batch_size, self.action_size])

            self.model.fit(update_input, update_output, batch_size=self.batch_size, epochs=1, verbose=0)

            self.episode -= self.epsilon_decay

            if self.time_step % self.update_target_frequency == 0:
                self.update_target_model()

        return

    def save_model(self, filename):
        self.model.save_weights(filename)

    def load_model(self, filename):
        self.model.load_weights(filename)
        self.target_model.load_weights(filename)

    def run_experiment(self, no_frame=200e6):
        self.episode = 0
        self.time_step = 0

        observation = self.env.reset()


def run_experiment(env, no_frames = 2e6):
    return