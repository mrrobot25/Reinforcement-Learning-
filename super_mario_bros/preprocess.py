from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
import matplotlib.pyplot as plt
import gym_super_mario_bros
from nes_py.wrappers  import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env,SIMPLE_MOVEMENT)

env = GrayScaleObservation(env, keep_dim=True)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env,4, channels_order='last')

'''
state = env.reset()

state, reward, done, info = env.step([3])
plt.figure(figsize = (10,8))
for i in range(state.shape[3]):
	plt.subplot(1,4,i+1)
	plt.imshow(state[0][:,:,i])
plt.show()	
'''
