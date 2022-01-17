import numpy as np
import random

class Agent():
	def __init__(self,lr, gamma, eps_start, eps_end, eps_dec, n_actions, n_states):
		self.n_actions = n_actions
		self.n_states = n_states
		self.lr = lr
		
		self.epsilon = eps_start
		self.eps_min = eps_end
		self.eps_dec = eps_dec
		self.gamma = gamma
		
		self.Q = {}
		self.init_Q()

	def init_Q(self):

		for state in range(self.n_states): 
			for action in range(self.n_actions):
				self.Q[(state,action)] = 0.0



	def choose_action(self,state):

		if np.random.random() < self.epsilon:
			action = np.random.choice([i for i in range(self.n_actions)])
			
		else:
			actions = np.array([self.Q[(state,a)] for a in range(self.n_actions)])
			action = np.argmax(actions)
		
		return action



	def decrement_epsilon(self):

		self.epsilon = self.epsilon * self.eps_dec if self.epsilon > self.eps_min\
								else self.eps_min
		return self.epsilon





	def learning(self,state,action,reward,new_state):

		actions = np.array([self.Q[(new_state,a)] for a in range(self.n_actions)])
		a_max = np.argmax(actions)
		
		self.Q[(state,action)] += self.lr*(reward + self.gamma*self.Q[(new_state,a_max)] - self.Q[(state,action)])
		#self.Q[state,action] = (1-self.lr) * self.Q[(state,action)] + self.lr * (reward + self.gamma * self.Q[(new_state,a_max)])

		self.decrement_epsilon()
		
		
		






