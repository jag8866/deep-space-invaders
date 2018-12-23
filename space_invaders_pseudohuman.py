"""
Agent that plays Space Invaders using move predictions from a neural network. Assumes the model
is already trained and saved to disk.
"""

import argparse
import sys
import termios
import tty
import gym
from gym import wrappers, logger
import numpy as np
from time import clock
from preprocess import preprocess
from keras.models import model_from_json

class Agent(object):
    """This agent plays using a neural network attempting to predict the gameplay decisions I would
    have made after observing data of me playing."""
    def __init__(self, action_space):
        self.action_space = action_space

        #Load a trained model from the .json and .h5 files
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights("model.h5")

    def act(self, observation, reward, done):
        #Run the model on the current frame data, get a predicted output, convert it from one-hot to an integer, and return it
        return np.argmax(self.model.predict(np.expand_dims(np.expand_dims(preprocess(observation), axis=3), axis=0))) 

def run():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='SpaceInvaders-v0', help='Select the environment to run')
    args = parser.parse_args()

    logger.set_level(logger.INFO)

    env = gym.make(args.env_id)


    env.seed()
    agent = Agent(env.action_space)

    episode_count = 100
    reward = 0
    done = False
    score = 0
    special_data = {}
    special_data['ale.lives'] = 3
    ob = env.reset()
    while not done:
        
        action = agent.act(ob, reward, done)
        ob, reward, done, _ = env.step(action)
        score += reward
        env.render()
     
    time = str(clock()).replace(".", "_")

    env.close()
    return score
run()
