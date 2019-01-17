import gym
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop, Adam
from collections import deque


class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.gamma = 0.99  # discount value
        self.learning_rate = 0.0025  # learning rate
        self.epsilon = 1.  # initial epsilon
        self.epsilon_min = 0.05  # final epsilon
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / 1e5  # linearly decreasing the epsilon
        self.batch_size = 32  # batch size in each training step

        self.memory = deque(maxlen=50000)  # experience replay length

        self.episode = 0  # record how many episodes the agent experienced
        self.time_step = 0  # record how many time steps the agent experienced

        self.train_frequency = 4  # train the neural network every 4 steps
        self.train_start = 10e3  # begin to train after 1000 steps
        self.update_target_frequency = 500  # update target neural netwrok every 500 steps
        self.eval_frequency = 10e3  # evaluate the online model every 1000 steps
        self.eval_episodes = 10  # evaluate the model by averaging 10 episode returns

        self.model = self.build_model()  # online network
        self.target_model = self.build_model()  # target network
        self.update_target_model()  # synchronize the two networks

        self.result_to_plot = np.zeros([1, 2])  # record all the results

    def build_model(self):
        # build the neural network
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_normal'))
        rmsprop = RMSprop(lr=self.learning_rate)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)
        # model.summary()

        return model

    def update_target_model(self):
        # copy the weights from online network to target network
        self.target_model.set_weights(self.model.get_weights())

    def get_epsilon(self):
        # get the epsilon value based on the current step
        return np.max([self.epsilon_min, self.epsilon - self.epsilon_decay * self.time_step])

    def get_action(self, state, train=True):
        # get the action using epsilon greedy policy
        if random.random() < self.get_epsilon() and train:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state.reshape(1, self.state_size))
            return np.argmax(q_value[0])

    def save_experience(self, exp):
        # save experience to the memory
        self.memory.append(exp)
        if exp[3]:
            # if this is terminal transition, save it to memory 10 times
            for _ in range(10):
                self.memory.append(exp)

    def train(self):
        # train the neural network
        if len(self.memory) < self.train_start:
            return

        if self.time_step % self.train_frequency == 0:
            # train the neural network every ... time steps

            mini_batch = random.sample(self.memory, self.batch_size)

            last_observations = np.array([rep[0] for rep in mini_batch])
            current_observations = np.array([rep[-1] for rep in mini_batch])

            target = self.model.predict(last_observations)
            next_q_for_action_selection = self.model.predict(current_observations)
            next_q = self.target_model.predict(current_observations)

            for i in range(self.batch_size):
                # for each transition in minibatch, compute the target
                next_action_sel = np.argmax(next_q_for_action_selection[i])
                current_action = mini_batch[i][1]
                reward = mini_batch[i][2]
                done = mini_batch[i][3]
                target[i][current_action] = reward + (1 - done) * self.gamma * next_q[i][next_action_sel]

            self.model.fit(last_observations, target, batch_size=self.batch_size, epochs=1, verbose=0)

            # update target neural network from time to time
            if self.time_step % self.update_target_frequency == 0:
                self.update_target_model()

        return

    def save_model(self, filename):
        # save weights to file
        self.model.save_weights(filename)

    def load_model(self, filename):
        # load weights from file
        self.model.load_weights(filename)
        self.target_model.load_weights(filename)

    def run_experiment(self, no_frame=1e6):
        # begin running the experiment for no_frame frames
        self.episode = 0
        self.time_step = 0

        while self.time_step < no_frame:
            # at the beginning of each episode, set done to False, set time step in this episode to 0
            # set reward to 0, reset the environment
            self.episode += 1
            done = False
            episode_time = 0
            episode_reward = 0
            evaluate_model = False
            last_observation = self.env.reset()
            # normalized the state
            last_observation[4] /= self.env.observation_space.high[4]
            last_observation[5] /= self.env.observation_space.high[5]

            while not done:
                # at each time step
                episode_time += 1
                self.time_step += 1

                # get action from online network and execute the action
                action = self.get_action(last_observation)
                observation, reward, done, info = self.env.step(action)
                observation[4] /= self.env.observation_space.high[4]
                observation[5] /= self.env.observation_space.high[5]

                episode_reward += reward
                # experience replay and train neural network
                self.save_experience((last_observation, action, reward, done, observation))
                self.train()

                last_observation = observation

                # from time to time evaluate the online model
                if self.time_step % self.eval_frequency == 0:
                    evaluate_model = True

            # print training information for each training episode
            # print('Training Episode:', self.episode)
            # print('Cumulative Reward:', episode_reward)
            # print('Episode Time Step:', episode_time)

            # evaluate the online network
            if evaluate_model:
                self.evaluate_model()

        # after running the experiment, plot the final result
        self.plot_result(rolling_window=15)

    def evaluate_model(self):
        # evaluation by setting epsilon to 0
        # store the episode returns in a list evaluate_reward
        evaluate_episode = 0
        evaluate_time_step = []
        evaluate_reward = []

        # use the online model run 10 episode and average the final reward
        while evaluate_episode < self.eval_episodes:
            evaluate_episode += 1
            done = False
            episode_time = 0
            episode_reward = 0
            last_observation = self.env.reset()
            last_observation[4] /= self.env.observation_space.high[4]
            last_observation[5] /= self.env.observation_space.high[5]

            while not done:
                episode_time += 1
                self.env.render()
                import time
                time.sleep(0.1)

                # get the action by setting epsilon to 0
                action = self.get_action(last_observation, train=False)
                observation, reward, done, info = self.env.step(action)
                observation[4] /= self.env.observation_space.high[4]
                observation[5] /= self.env.observation_space.high[5]

                episode_reward += reward
                last_observation = observation

            evaluate_time_step.append(episode_time)
            evaluate_reward.append(episode_reward)

        avg_time_step = sum(evaluate_time_step) / len(evaluate_time_step)
        avg_reward = sum(evaluate_reward) / len(evaluate_reward)

        # save the model with its final score
        self.save_model('save_model/small_normalized_%s.h5' % str(int(avg_reward)))
        self.env.close()
        self.result_to_plot = np.concatenate([self.result_to_plot, np.array([[self.episode, avg_reward]])])
        print('Model Evaluation Result at Episode %s, Time step %s' % (self.episode, self.time_step))
        print('Avg Reward in 10 Episodes:', avg_reward)
        print('Avg Time Step Each Episode:', avg_time_step)
        print('Final Result:', max(self.result_to_plot[1:, 1]))
        print('Epsilon:', self.get_epsilon())
        print('-------------------------------------------')

    def plot_result(self, rolling_window):
        # plot the final result: average score at different episodes and the rolling average
        self.x = self.result_to_plot[1:, 0]
        self.y = self.result_to_plot[1:, 1]

        # rolling average
        self.x_roll = self.result_to_plot[rolling_window-1:, 0]
        self.y_roll = self.moving_average(self.result_to_plot[:, 1], n=rolling_window)

        fig = plt.figure()
        plt.plot(self.x, self.y, 'o-', label='raw')
        plt.plot(self.x_roll, self.y_roll, '-', label='rolling average')
        plt.xlabel('Number of Episodes')
        plt.ylabel('Average Episode Return')
        leg = plt.legend(loc='best')
        leg.get_frame().set_alpha(0.5)
        fig.tight_layout()
        plt.show()

    def moving_average(self, a, n):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n


def main():
    agent = DQNAgent(env=gym.make('Acrobot-v1'))
    agent.run_experiment()


if __name__ == '__main__':
    main()
