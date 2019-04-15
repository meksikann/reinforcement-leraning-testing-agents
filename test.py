from __future__ import division
import argparse
from PIL import Image
import numpy as np
import gym
import random
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from preprocessing import prepro, discount_rewards

UP_ACTION = 2
DOWN_ACTION = 3
gamma = 0.99
x_train = []
y_train = []
rewards = []
reward_sum = 0
episode_num = 0


def start_pong():
    env = gym.make('Pong-v0')
    observation = env.reset()
    prev_input = None
    model = get_model()

    while True:
        # preprocess observation  and set input x_train input as difference between two images(frames)
        current_input = prepro(observation)
        x_diff = current_input - prev_input if prev_input is not None else np.zeros(80*80)
        prev_input = current_input

        # use policy network to action
        proba = model.predict(np.expand_dims(x_diff, axis=1).T)




def get_model():
    model = Sequential()

    model.add(Dense(units=200, input_dim=80*80, activation='relu', kernel_initializer='glorot_uniform'))

    model.add(Dense(units=1, activation='sigmoid', kernel_initializer='RandomNormal'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


start_pong()
#
# def start_atary():
#
#     env = gym.make('BreakoutDeterministic-v4')
#
#     frame = env.reset()
#
#     env.render()
#
#     is_done = False
#
#     while not is_done:
#         frame, reward, is_done, _ = env.step(env.action_space.sample())
#         env.render()
#
#     print ('Start reinforcement learning testing')
#
#
# def to_gray(img):
#     return np.mean(img, axis=2).astype(np.uint8)
#
#
# def downsample(img):
#     return img[::2, ::2]
#
#
# def prepprocess(img):
#     return to_gray(downsample(img))
#
#
# def transform_reward(reward):
#     return np.sign(reward)
#
#
# def fit_batch(model, gamma, start_states, actions, rewards, next_states, is_terminal):
#     """one DQL iteration
#
#     :param model: DQN
#     :param gamma: discount factor
#     :param start_states: np.array
#     :param actions: np.array with one-hot encoded actions regarding start_states
#     :param rewards: np.array regarding to start_states received
#     :param next_states: np,.array of
#     """
#     next_Q_values = model.predict([next_states, np.ones(actions.shape)])
#
#     next_Q_values[is_terminal] = 0
#
#     start_Q_values = rewards + gamma * np.max(next_Q_values, axis=1)
#
#     model.fit(
#         [start_states, actions], actions * start_Q_values[:, None],
#         nb_epoch=1, batch_size=len(start_states), verbose=0
#     )
#
#
#
# def q_iteration(env, model, state, iteration, memory):
#     # Choose epsilon based on the iteration
#     epsilon = get_epsilon_for_iteration(iteration)
#
#     # Choose the action
#     if random.random() < epsilon:
#         action = env.action_space.sample()
#     else:
#         action = choose_best_action(model, state)
#
#     # Play one game iteration (note: according to the next paper, you should actually play 4 times here)
#     new_frame, reward, is_done, _ = env.step(action)
#     memory.add(state, action, new_frame, reward, is_done)
#
#     # Sample and fit
#     batch = memory.sample_batch(32)
#     fit_batch(model, batch)
#
#
#
#
