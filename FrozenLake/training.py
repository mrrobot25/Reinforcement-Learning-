import numpy as np
from q_algorithm import Agent
import gym
import matplotlib.pyplot as plt


if __name__ == "__main__":
	
	env = gym.make("FrozenLake-v1")
	n_actions = env.action_space.n
	n_states = env.observation_space.n
	print("n_actions",n_actions)
	print("n_states",n_states)
	agent = Agent(lr = 0.01,gamma = 0.9,eps_start = 1.0, eps_end = 0.01, eps_dec = 0.99995, n_actions = n_actions,n_states = n_states)
	
	n_episodes = 50000
	
	
	score = []
	win_pct_list = []
	
	for episode in range(n_episodes):
		done = False
		state = env.reset()
		
		reward_per_episode = 0
		
		while not done:
			action_ = agent.choose_action(state)
			new_state, reward, done, info = env.step(action_)
			agent.learning(state,action_,reward,new_state)

			state = new_state
			reward_per_episode  += reward


		score.append(reward_per_episode)
		if episode % 100 == 0:

			win_pct = np.mean(score[-100:])
			win_pct_list.append(win_pct)
			if episode % 1000 == 0:
				print('episode ',episode, 'win pct %2f' % win_pct,
					'epsilon %2f '% agent.epsilon)
 

	print("No errors")
	print(agent.Q)
	plt.plot(win_pct_list)
	plt.show()
	
	
			
	
	
	


	
			
