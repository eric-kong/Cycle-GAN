import tensorflow as tf
import numpy as np
import scipy.misc
import cv2

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
			idx = np.random.randint(0, self.pool_size)
			output = self.pool[idx].copy()
			self.pool[idx] = image
			return output
		else:
			return image

def load_image(image_path, output_size=256):
	image = scipy.misc.imread(image_path, mode="RGB").astype(np.float32)
	image = scipy.misc.imresize(image, [output_size, output_size])

	image = image / 127.5 - 1

	return image

def convert2image(images):
	def resize(image):
		return tf.image.convert_image_dtype((image + 1.0) / 2.0, tf.uint8)
	return tf.map_fn(resize, images, dtype=tf.uint8)


def save_images(path, images, size):
	images = images.eval()
	for i in range(size):
		scipy.misc.imsave(path + str(i) + ".jpg", images[i, :, :, :])

