import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import preprocess

class Callback(BaseCallback):
	def __init__(self,check_freq, save_path, verbose=1):
		super(Callback,self).__init__(verbose)
		self.check_freq = check_freq
		self.save_path = save_path

	def _init_callback(self):
		if self.save_path is not None:
			os.makedirs(self.save_path, exist_ok=True)

	def _on_step(self):
		if self.n_calls % self.check_freq ==0:
			model_path = os.path.join(self.save_path, 'model {}'.format(self.n_calls))
			self.model.save(model_path)
		return True

CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'

callback = Callback(check_freq = 10000, save_path = CHECKPOINT_DIR)

model = PPO('CnnPolicy', preprocess.env, verbose = 1, tensorboard_log=LOG_DIR, learning_rate = 0.000001, n_steps = 512)

model.learn(total_timesteps=50000,callback = callback)

