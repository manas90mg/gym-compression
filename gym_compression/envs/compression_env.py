import gym
from gym import error, spaces, utils
from gym.utils import seeding
from compress_layerresnet_v2 import reset, step

class CompressionEnv(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self):
		# action_space = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0). Divide by 10 to get actual alpha value
		self.action_space = spaces.Discrete(15) 
		# observation_space = tuple of layer #, total weights in layer and # of weights pruned.
		self.observation_space = spaces.Box(low=np.array([0,-8,0]), high=np.array([60,0,5]), dtype=float)
 
	def step(self, action):
		alpha = (action+10.0)/10.0
		return step(alpha)
	
	def reset(self):
		return reset()







