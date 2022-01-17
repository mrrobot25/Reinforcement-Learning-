import gym_super_mario_bros
from nes_py.wrappers  import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
'''
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
done = True

for step in range(10000):
		if done:
			env.reset()
		state,reward,done,info = env.step(env.action_space.sample())
		env.render()

env.close()
'''

# preprocessing environment

from gym.wrappers import FrameStack, GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from matplotlib import pyplot as plt


env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

#env = GrayScaleObservation(env, keep_dim=True)
#env = DummyVecEnv([lambda: env])
#env = VecFrameStack(env,4, channels_order='last')
from stable_baselines3 import PPO

model = PPO.load('/home/lenovo/super_mario/train/best_model 10000.zip')

state = env.reset()

while True:
	action, _state = model.predict(state)
	action, state, done, info = env.step(action)
	env.render()
