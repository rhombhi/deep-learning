import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = .3
        self.gamma = 1
        print(self.alpha)

        
    def select_action(self, state, i_episode=None, eps=None):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
#         return np.random.choice(self.nA)
        return np.random.choice(np.arange(self.nA), p = self.epsilon_greedy_probs(state, i_episode, eps))
    
    def epsilon_greedy_probs(self, state, i_episode=None, eps=None):
        if eps is None:
            self.epsilon = 1/i_episode
        else:
            self.epsilon = eps
        policy = np.ones(self.nA) * self.epsilon / self.nA
        policy[np.argmax(self.Q[state])] = 1 - self.epsilon + self.epsilon/self.nA
        return policy

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        # method 1 : sarsa(0)
#         next_action = self.select_action(state, eps=.0001)
#         self.Q[state][action] += self.alpha * (reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action])
        # method 2 : sarsamax
#         self.Q[state][action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state][action]) - self.Q[state][action])
        #method 3: expected srasa
        policy_s = self.epsilon_greedy_probs(state, eps=0.001)
        self.Q[state][action] += self.alpha * (reward + self.gamma * np.dot(self.Q[next_state], policy_s) - self.Q[state][action])
#         self.Q[state][action] += self.alpha * (reward + self.gamma * self.Q[next_state][action] - self.Q[state][action])