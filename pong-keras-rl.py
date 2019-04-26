import gym
from PIL import Image
import numpy as np
import argparse

# import mathplotlib.pyplot as plt

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


print('Initialisation ---------------->>>>>>>>>>>>>>>>')


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


# get command arguments
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env_name', type=str, default='BreakoutDeterministic-v4')
parser.add_argument('--weights', type=str, default=None)
arguments = parser.parse_args()

# get training environment
env = gym.make(arguments.env_name)
np.random.seed(123)
env.seed(123)
actions_number = env.action_space.n

# build model
input_shape = (WINDOW_LENGTH, ) + INPUT_SHAPE
model = Sequential()

# check image ordering if 'tensorflow' or 'theano' and add first layer
if K.image_dim_ordering() == 'tf':
    # (width, height, channels)
    model.add(Permute((2, 3, 1), input_shape=input_shape))
elif K.image_dim_ordering() == 'th':
    # (channels, width, height)
    model.add(Permute((1, 2, 3), input_shape=input_shape))
else:
    raise RuntimeError('Unknown image_dim_ordering.')

model.add(Convolution2D(32, (8, 8), strides=(4, 4)))
model.add(Activation('relu'))
model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(actions_number))
model.add(Activation('linear'))

model.summary()

# init the Agent
memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
atari_agent = GamePreprocessor()

