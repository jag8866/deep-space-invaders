'''
Allows a human to play Space Invaders (albeit frame by frame) with keyboard
input, which is stored along with the associated frames to be used later
as training data.
'''

import argparse
import sys
import termios
import tty
import gym
from gym import wrappers, logger
import numpy as np
from time import clock
from preprocess import preprocess

def getch():
    """Basic keyboard input function for human input to SI
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
                                     
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

class Agent(object):
    """This agent takes human input frame by frame and records the observations and corresponding
        human inputs to be used for training later."""
    def __init__(self, action_space):
        self.action_space = action_space
        self.delay = False
        self.observations = None
        self.actions = None
        self.quit = False

    def act(self, observation, reward, done):
        action = 0
        if self.delay == True:
            self.delay = False
            action = self.actions[-1]
        else:
            userin = getch()
            if userin == "d":
                self.delay = True
                action = 2
            elif userin == "a":
                self.delay = True
                action = 3
            elif userin == "w":
                self.Delay = True
                action = 1
            elif userin == "s":
                action = 0
            elif userin == "q":
                self.quit = True
                return 0
        np.save("test.npy", observation)   
        
        if self.actions is None:
            self.actions = np.array([action])
        elif action != 0: 
            self.actions = np.append(self.actions, action)
        
        if not isinstance(self.observations, np.ndarray):
            self.observations = np.expand_dims(np.expand_dims(preprocess(observation), axis=3), axis=0)
        elif action != 0: #When input is 0 we save no data, this makes 's' sort of a "skip" button
            obs = np.expand_dims(np.expand_dims(preprocess(observation), axis=3), axis=0)
            self.observations = np.append(self.observations, obs, axis=0)

        return action

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
    while not done and not agent.quit:
        
        action = agent.act(ob, reward, done)
        ob, reward, done, _ = env.step(action)
        score += reward
        env.render()
     
    time = str(clock()).replace(".", "_")
    np.save('X_' + time + '.npy', agent.observations)
    np.save('Y_' + time + '.npy', agent.actions)

    env.close()

run()
