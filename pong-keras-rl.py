import gym
from PIL import Image
import numpy as np

import mathplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4


class GamePreprocessor(Processor):
    def process_observation(self, observation):
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L') # resize image and convert it to black/white
        processed_observation = np.array(img)
        return processed_observation.astype('uint8') # save in memory with type

    def process_state(self, batch):
        processed_batch = batch.astype('float32') / 255
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.) # values smaler then -1 become -1 and values bigger then 1 become 1

