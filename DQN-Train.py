# -*- coding: utf-8 -*-

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
# (nothing gets printed in Jupyter, only if you run it standalone)

config.allow_soft_placement = True
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

assert tf.test.is_gpu_available()
assert tf.test.is_built_with_cuda()

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
'''

import tensorflow
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
'''

import numpy as np
import gym

import gym_airsim.envs
import gym_airsim

import argparse

from keras.models import Model, Sequential
from keras.layers import Input, Reshape, Dense, Flatten, Conv2D, concatenate
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.utils import plot_model

from callbacks import *
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.processors import MultiInputProcessor
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='AirSimEnv-v42')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

with tf.device('/device:GPU:0'):
    env = gym.make(args.env_name)
    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n

    # Obtaining shapes from Gym environment
    img_shape = env.simage.shape
    vel_shape = env.svelocity.shape
    dst_shape = env.sdistance.shape
    geo_shape = env.sgeofence.shape

    AE_shape = env.sAE.shape

    # Keras-rl interprets an extra dimension at axis=0
    # added on to our observations, so we need to take it into account
    img_kshape = (1,) + img_shape

    input_layer = Input(shape=img_kshape)
    conv1 = Conv2D(32, (4, 4), strides=(4, 4), activation='relu', input_shape=img_kshape, name='conv1',
                   data_format="channels_first")(input_layer)
    conv2 = Conv2D(64, (3, 3), strides=(2, 2), activation='relu', name='conv2')(conv1)
    flat1 = Flatten(name='flat1')(conv2)

    auxiliary_input1 = Input(vel_shape, name='vel')
    auxiliary_input2 = Input(dst_shape, name='dst')
    auxiliary_input3 = Input(geo_shape, name='geo')
    auxiliary_input4 = Input(AE_shape, name='ae')

    denses = concatenate([flat1, auxiliary_input1, auxiliary_input2, auxiliary_input3, auxiliary_input4])
    denses = Dense(256, activation='relu')(denses)
    denses = Dense(256, activation='relu')(denses)
    denses = Dense(256, activation='relu')(denses)

    predictions = Dense(nb_actions, kernel_initializer='zeros', activation='linear')(denses)

    model = Model(inputs=[input_layer, auxiliary_input1, auxiliary_input2, auxiliary_input3, auxiliary_input4],
                  outputs=predictions)

    print(model.summary())

    train = True

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=100000, window_length=1)  # reduce memmory

    processor = MultiInputProcessor(nb_inputs=5)

    # Select a policy. We use eps-greedy action selection, which means that a random action is selected
    # with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
    # the agent initially explores the environment (high eps) and then gradually sticks to what it knows
    # (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05c
    # so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=0.0,
                                  nb_steps=50000)

    dqn = DQNAgent(model=model, processor=processor, nb_actions=nb_actions, memory=memory, nb_steps_warmup=50,
                   enable_double_dqn=True,
                   enable_dueling_network=False, dueling_type='avg',
                   target_model_update=1e-2, policy=policy, gamma=.99)

    dqn.compile(Adam(lr=0.00025), metrics=['mae'])

    if train:
        # Okay, now it's time to learn something! We visualize the training here for show, but this
        # slows down training quite a lot. You can always safely abort the training prematurely using
        # Ctrl + C.
        log_filename = 'dqn_{}_log.json'.format(args.env_name)
        callbacks = [FileLogger(log_filename, interval=10)]
        # Calling the function TrainEpisodeLogger() to get checkpoint weights at the best episode
        callbacks += [TrainEpisodeLogger()]
        # tb_log_dir = 'logs/tmp'
        # callbacks = [TensorBoard(log_dir=tb_log_dir, histogram_freq=0)]
        dqn.fit(env, callbacks=callbacks, nb_steps=125000, visualize=False, verbose=0, log_interval=100)

        # After training is done, we save the final weights.
        dqn.save_weights(''.format(args.env_name), overwrite=True)

    else:

        dqn.load_weights('checkpoint_reward_170.93000887107826.h5f')

        dqn.test(env, nb_episodes=100, visualize=False)


