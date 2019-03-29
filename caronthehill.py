import os
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam, sgd

from tensorflow import ConfigProto, Session
from keras.backend import set_session

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0";  

# Just disables the warning for CPU instruction set,
#  doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


config = ConfigProto( device_count = {'GPU': 1 , 'CPU': 8} ) 
sess = Session(config=config) 
set_session(sess)

import math
import random
import click
import numpy as np
import cv2
from matplotlib import pyplot as plt

from display_caronthehill import save_caronthehill_image
from collections import deque

def imagePreprocessing():

    directory = 'training/'
    cap = cv2.VideoCapture("{}/img%04d.jpg".format(directory)) 
    amount_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    for i in range(amount_of_frames):
        # Capture frame-by-frame
        _, frame = cap.read()
        frame = cv2.resize(frame, (200, 200))
        gray =  cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("processed/img{0:0>4}.jpg".format(i), gray)


    cap.release()
    cv2.destroyAllWindows()
# When everything done, release the capture
def makeVideo():
    """ 
    Creates a 10 fps video in mp4 format called 'caronthehill_clip'
    using frames whose names follow the regular expression 'img%05d.jpg'
    and are issued from a directory called video that has to be created beforehand

    Arguments : /
    -----------

    Returns :
    ---------

    -video in the current working directory
    """
    os.system("cd video && ffmpeg -r 10 -i img%05d.jpg -vcodec mpeg4 -y caronthehill_clip.mp4")




class Domain:

    def __init__(self, discount, actions, integration_step, discretisation_step):
        self.discount = discount
        self.actions = actions
        self.integration_step = integration_step
        self.discretisation_step = discretisation_step
        self.ratio = discretisation_step / integration_step
        self.stuck = False

    def _nextPosition(self, position, speed):
        """
        Computes the next position using the Euler integration method
        of a position and a speed.
        Arguments:
        ----------
        - position : position of the agent in the domain
        - speed : speed of the agent in the domain

        Returns:
        ----------
        - float value for the position
        """

        next_position = position + self.integration_step * speed

        # Check if you reach a terminal state
        if abs(next_position) > 1:
            self.stuck = True
        return next_position

    def _nextSpeed(self, position, speed, action):
        """
        Computes the next speed using the Euler integration method
        using the position, speed and action taken of the agent.
        Arguments:
        ----------
        - position : position of the agent in the domain
        - speed : speed of the agent in the domain
        - agent : action that the agent took in the domain

        Returns:
        ----------
        - float value for the speed
        """
        next_speed = speed + self.integration_step * self._speedDiff(position, speed, action)

        # Check if you reach a terminal state
        if abs(next_speed) > 3:
            self.stuck = True
        return next_speed

    def _speedDiff(self, position, speed, action):
        """ 
        Derivative used for the speed dyanmics 
        Arguments:
        ----------
        - position : position of the agent in the domain
        - speed : speed of the agent in the domain
        - agent : action that the agent took in the domain

        Returns:
        ----------
        - float value for the speed derivative
        """
        return (action/(1 + self._hill_diff(position)**2)
                - 9.81 * self._hill_diff(position) /
                (1 + self._hill_diff(position)**2)
                - ((self._hill_diff(position) * self._hill_diff_diff(position)) 
                * (speed**2))/(1 + self._hill_diff(position)**2))

    def _hill_diff(self, position):
        """ 
        Derivative of the Hill(position) function which represents
        the hill in the car on the hill problem 
        Arguments:
        ----------
        - position : position of the agent in the domain
        Returns:
        ----------
        - float value for the hill derivative
        """
        if position < 0:
            return 2 * position + 1
        else:
            return (1/math.sqrt(1 + 5 * position ** 2)
                    - 5 * position ** 2 * (1 + 5 * position ** 2)**-1.5)

    def _hill_diff_diff(self, position):
        """ 
        Second derivative of the Hill(position) function which represents
        the hill in the car on the hill problem 
        Arguments:
        ----------
        - position : position of the agent in the domain
        Returns:
        ----------
        - float value for the second hill derivative
        """
        if position < 0:
            return 2
        else:
            return position * ((75 * (position ** 2)/((1 + 5 * position**2)**2.5)) - 5/((1 + 5 * position ** 2)**2.5)) \
                - 10 * position/((1 + 5 * position ** 2)**1.5)

    def isStuck(self):
        """ Return true if the agent reached a terminal state in the domain"""
        return self.stuck

    def reset(self):
        """ Reset the terminal state lock of the domain"""
        self.stuck = False

    def expectedReward(self, starting_position, starting_speed, computePolicy, N):
        """ 
        Computes the expected reward of a state given a policy after N iterations 
        Arguments:
        ----------
        - starting_position : position of the agent in the domain
        - starting_speed : speed of the agent in the domain
        - computePolicy : function that takes position and speed as argument and return
            an action

        Returns:
        ----------
        - float value for the expected reward
        """
        total_reward = 0
        position = starting_position
        speed = starting_speed
        for i in range(N):
            if self.stuck:
                break
            action = computePolicy(position, speed)
            reward, position, speed = self.rewardAtState(position, speed, action)
            total_reward += (self.discount ** i) * reward
        return total_reward

    def nextState(self, position, speed, action):
        """ 
        Computes the next state (position and speed) for a given position,
        speed and action that the agent is going to take

        Arguments:
        ----------
        - position : position of the agent in the domain
        - speed : speed of the agent in the domain
        - agent : action that the agent took in the domain

        Returns:
        ----------
        - float value, float value
        """
        next_position = position
        next_speed = speed
        for _ in range(int(self.ratio)):
            next_position = self._nextPosition(position, speed)
            next_speed = self._nextSpeed(position, speed, action)
            position = next_position
            speed = next_speed
        return next_position, next_speed

    def rewardAtState(self, position, speed, action):
        """
        Computes the reward that the agent is going to get 
        when starting at the given position and speed and taking the
        given action
        ----------
        - position : position of the agent in the domain
        - speed : speed of the agent in the domain
        - agent : action that the agent took in the domain

        Returns:
        ----------
        - integer, float value, float value
        """

        # If terminal state, return 0 for subsequent moves and
        # do not update the position nor speed
        if(self.stuck):
            return 0, position, speed

        new_pos, new_speed = self.nextState(position, speed, action)

        if (new_pos < -1 or abs(new_speed) > 3):
            return -1, new_pos, new_speed
        elif (new_pos > 1 and abs(new_speed) <= 3):
            return 1, new_pos, new_speed
        else:
            return 0, new_pos, new_speed




class ParametricQLearningAgent:

    def __init__(self, domain, n_episodes):

        self.domain = domain
        self.n_episodes = n_episodes
        self.learning_rate = 0.05
        self.batch_size = 32
        self.epsilon = 0.1  # exploration rate
        self.epsilon_decay = 0.99
        self.model = self.create_model()
        self.memory = deque(maxlen=2000)
        
    def create_model(self):
        # Neural Net for Deep Q Learning
        model = Sequential()
        # Input Layer of state size(3) and Hidden Layer with 24 nodes
        model.add(Dense(24, input_shape=(3,), activation='relu'))
        # Hidden layer with 24 nodes
        model.add(Dense(24, activation='relu'))
        # Output Layer with reward
        model.add(Dense(1, activation='linear'))
        # Create the model based on the information above
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model


    def chooseAction(self, position, speed):
        """Epsilon greedy"""

        # Random action
        if np.random.rand() <= self.epsilon:
            return random.choice(self.domain.actions)

        # Action maximising Q
        prediction_f = self.model.predict(np.array([position, speed, 4]).reshape(1,3))
        prediction_b = self.model.predict(np.array([position, speed, -4]).reshape(1,3))
        return 4 if prediction_f > prediction_b else -4   

    def replay(self, batch_size):
        # Sample minibatch
        minibatch = random.sample(self.memory, batch_size)
        for position, speed, action, reward, next_position, next_speed in minibatch:
            # predict the future discounted reward
            forward_sample = np.array([next_position, next_speed, 4]).reshape(1,3)
            backward_sample = np.array([next_position, next_speed, -4]).reshape(1,3)
            forward_reward = self.model.predict(forward_sample)[0]
            backward_reward = self.model.predict(backward_sample)[0]
            target = reward + self.domain.discount * max(forward_reward, backward_reward)

            # Train the Neural Net with the state and target
            self.model.fit(np.array([position, speed, action]).reshape(1,3), target, epochs=1, verbose=0)
        self.epsilon *= self.epsilon_decay


    def train(self):
        # 1000 episodes with starting state (p,s) = (-0.5, 0)
        for ep in range(self.n_episodes):
            position = -0.5
            speed = 0
            i = 0
            while not self.domain.isStuck() and i < 200:
                action = self.chooseAction(position, speed)
                reward, next_pos, next_speed = self.domain.rewardAtState(position, speed, action)
                self.memory.append((position, speed, action, reward, next_pos, next_speed))
                position = next_pos
                speed = next_speed
                i += 1
            self.domain.reset()
            print("episode: {}/{}".format(ep, self.n_episodes))
            
            if (ep + 1) %50 == 0:
                self.createRewardHeatmap(ep + 1, 'mlp')
                self.createPolicyHeatmap(ep + 1, 'mlp')
            
            # Refit model on 32 samples
            if len(self.memory) < self.batch_size:
                self.replay(len(self.memory))
            else:
                self.replay(self.batch_size)
                
    def initTrainingSet(self, n_episodes, n_trans):
        for ep in range(n_episodes):
            position = -0.5
            speed = 0
            # Explore the domain until reaching a terminal state
            for i in range(n_trans):
                save_caronthehill_image(position, speed,"training/img{0:0>4}.jpg".format(i))
                action = random.choice(self.domain.actions)
                _, next_pos, next_speed = self.domain.rewardAtState(position, speed, action)
                position = next_pos
                speed = next_speed


    



if __name__ == '__main__':



    # CONSTANTS
    PLUS4 = 4
    MINUS4 = -4
    ACTIONS = [PLUS4, MINUS4]
    DISCOUNT = 0.95
    DISCRETE_STEP = 0.1
    INTEGRATION_STEP = 0.001


    # QNetwork = ParametricQLearningAgent(Domain(DISCOUNT, ACTIONS, INTEGRATION_STEP, DISCRETE_STEP), 1000)
    # QNetwork.initTrainingSet(1,50)

    imagePreprocessing()