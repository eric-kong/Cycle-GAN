import tensorflow as tf 
from generator import Generator
from discriminator import Discriminator 

class CycleGAN:
	def __init__(self, name="cylegan", image_size=256, ngf=64, ndf=64):
		self.image_size = image_size
		self.ngf = ngf
		self.ndf = ndf

		# generate images in the domain X
		self.generator_X = Generator(name="generatorX", ngf=self.ngf, image_size=image_size)
		# generate images in the domain Y
		self.generator_Y = Generator(name="generatorY", ngf=self.ngf, image_size=image_size)
		# discriminate images in the domain X
		self.discriminator_X = Discriminator(name="discriminatorX", ndf=self.ndf)
		# discriminate images in the domain Y
		self.discriminator_Y = Discriminator(name="discriminatorY", ndf=self.ndf)


	def build_model(self):
		self.real_X = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name="real_X")
		self.real_Y = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name="real_Y")
		
		self.fake_X = self.generatorX(self.real_Y, reuse=False)
		self.fake_Y = self.generatorY(self.real_X, reuse=False)

		# real_Y -> fake_X -> fake_XY
		self.fake_XY = self.generatorY(self.fake_X, reuse=True)
		# real_X -> fake_Y -> fake_YX
		self.fake_YX = self.generatorX(self.fake_Y, reuse=True)

		self.d_fake_X = self.discrimatorX(self.fake_X, reuse=False)
		self.d_fake_Y = self.dircrimatorY(self.fake_Y, reuse=False)

		# GAN Loss
		# generate X
		self.loss_gan_generator_X = tf.reduce_mean(tf.squared_difference(self.d_fake_Y, tf.ones_like(self.d_fake_Y)))
		self.loss_gan_discriminator_Y = tf.reduce_mean(tf.squared_difference(self.real_Y, tf.ones_like(self.real_Y)))
		self.loss_gan_generator_Y = tf.reduce_mean(tf.squared_difference(self.d_fake_X, tf.ones_like(self.d_fake_X)))
		self.loss_gan_discriminator_X = tf.reduce_mean(tf.squared_difference(self.real_X, tf.ones_like(self.real_X)))

		# Cycle Consistency Loss
		self.loss_cycle = tf.reduce_mean(tf.abs(self.fake_X - self.real_X)) + tf.reduce_mean(tf.abs(self.fake_Y - self.real_Y))

		# Generator_X Loss
		self.loss_generator_X = self.loss_gan_generator_X + self.loss_cycle
		# Generator_Y Loss
		self.loss_generator_Y = self.loss_gan_generator_Y + self.loss_cycle
		# Discriminator_X Loss
		self.loss_discriminator_X = 
		# Discriminator_Y Loss



