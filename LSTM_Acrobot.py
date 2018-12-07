import gym
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import RMSprop, Adam
from collections import deque


class LSTMAgent:
    def __init__(self, env):
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.gamma = 0.99
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
        self.eval_frequency = 50e3
        self.eval_episodes = 10

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        model = Sequential()
        model.add(LSTM(32, input_shape=(1, self.state_size), activation='tanh', kernel_initializer='he_normal'))
        model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_normal'))
        # rmsprop = RMSprop(lr=self.learning_rate)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)
        # model.summary()

        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_epsilon(self):
        return self.epsilon - self.epsilon_decay * self.time_step

    def get_action(self, state, train=True):
        if random.random() < self.get_epsilon() and train:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state.reshape(-1, 1, self.state_size))
            return np.argmax(q_value[0])

    def save_experience(self, exp):
        self.memory.append(exp)

    def train(self, last_ob, action, reward, done, ob):

        target = self.model.predict(last_ob.reshape([-1, 1, self.state_size]))
        # next_q_for_action_selection = self.model.predict(current_observations)
        # next_q = self.target_model.predict(ob.reshape([1, self.state_size]))
        # next_action = np.argmax(next_q[0])

        # next_action_sel = np.argmax(next_q_for_action_selection[i])
        # current_action = mini_batch[i][1]
        # reward = mini_batch[i][2]
        # done = mini_batch[i][3]
        next_q_max = np.argmax(self.model.predict(ob.reshape([-1, 1, self.state_size]))[0])
        target[0][action] = reward + (1 - done) * self.gamma * next_q_max

        self.model.fit(last_ob.reshape([-1, 1, self.state_size]), target, epochs=1, verbose=0)

        # if self.time_step % self.update_target_frequency == 0:
        #     self.update_target_model()

        return

    def save_model(self, filename):
        self.model.save_weights(filename)

    def load_model(self, filename):
        self.model.load_weights(filename)
        self.target_model.load_weights(filename)

    def run_experiment(self, no_frame=200e6):
        self.episode = 0
        self.time_step = 0

        while self.time_step < no_frame:
            self.episode += 1
            done = False
            episode_time = 0
            episode_reward = 0
            evaluate_model = False
            last_observation = self.env.reset()

            while not done:
                episode_time += 1
                self.time_step += 1

                action = self.get_action(last_observation)
                observation, reward, done, info = self.env.step(action)

                episode_reward += reward
                self.train(last_observation, action, reward, done, observation)

                last_observation = observation

                if self.time_step % self.eval_frequency == 0:
                    evaluate_model = True

            # print('Training Episode:', self.episode)
            # print('Cumulative Reward:', episode_reward)
            # print('Episode Time Step:', episode_time)

            if evaluate_model:
                self.evaluate_model()

    def evaluate_model(self):
        evaluate_episode = 0
        evaluate_time_step = []
        evaluate_reward = []

        while evaluate_episode < self.eval_episodes:
            evaluate_episode += 1
            done = False
            episode_time = 0
            episode_reward = 0
            last_observation = self.env.reset()

            while not done:
                episode_time += 1
                self.env.render()

                action = self.get_action(last_observation, train=False)
                observation, reward, done, info = self.env.step(action)

                episode_reward += reward
                last_observation = observation

            evaluate_time_step.append(episode_time)
            evaluate_reward.append(episode_reward)

        avg_time_step = sum(evaluate_time_step) / len(evaluate_time_step)
        avg_reward = sum(evaluate_reward) / len(evaluate_reward)

        print('Model Evaluation Result at %s' % self.time_step)
        print('Avg Reward in 10 Episodes:', avg_reward)
        print('Avg Time Step Each Episode:', avg_time_step)
        print('-------------------------------------------')

        self.save_model('save_model/%s.h5' % str(int(avg_reward)))
        self.env.close()


def main():
    env = gym.make('Acrobot-v1')
    agent = LSTMAgent(env)
    agent.run_experiment()


if __name__ == '__main__':
    main()
