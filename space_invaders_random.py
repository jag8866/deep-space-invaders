'''
Agent that plays Space Invaders by choosing inputs completely at random.
Intended for use as a baseline with which to compare the pseudohuman model.
'''

import argparse
import sys
import gym
from gym import wrappers, logger

class Agent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

def run():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='SpaceInvaders-v0', help='Select the environment to run')
    args = parser.parse_args()

    logger.set_level(logger.INFO)

    env = gym.make(args.env_id)

    outdir = 'random-agent-results'


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
     
    # Close the env and write monitor result info to disk
    env.close()
    return score
