import numpy as np

class ImagePool(object):
	def __init__(self, pool_size=50):
		self.pool_size = pool_size
		self.pool = []
	def __call__(self, image):
		if self.pool_size <= 0:
			return image
		if len(self.pool) < self.pool_size:
			self.pool.append(image)
			return image
		if np.random.rand() < 0.5:
			idx = np.random.randomrange(0, self.pool_size)
			output = self.pool[idx].copy()
			self.pool[idx] = image
			return output
		else:
			return image



