import tensorflow as tf 
from generator import Generator
from discriminator import Discriminator 

class CycleGAN:
	def __init__(self, name="cylegan", image_size=256, ngf=64, ndf=64):
		self.image_size = image_size
		self.ngf = ngf
		self.ndf = ndf

		self.lambda1

		self.lambda2

		# adam 
		self.beta1

		self.batch_size

		# generate images in the domain X
		self.generator_X = Generator(name="generator_X", ngf=self.ngf, image_size=image_size)
		# generate images in the domain Y
		self.generator_Y = Generator(name="generator_Y", ngf=self.ngf, image_size=image_size)
		# discriminate images in the domain X
		self.discriminator_X = Discriminator(name="discriminator_X", ndf=self.ndf)
		# discriminate images in the domain Y
		self.discriminator_Y = Discriminator(name="discriminator_Y", ndf=self.ndf)

		_build_model()
		_init_optimizer()

	def _build_model(self):
		self.real_X = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name="real_X")
		self.real_Y = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name="real_Y")
		
		self.fake_X = self.generatorX(self.real_Y, reuse=False)
		self.fake_Y = self.generatorY(self.real_X, reuse=False)

		# real_Y -> fake_X -> fake_XY
		self.fake_XY = self.generatorY(self.fake_X, reuse=True)
		# real_X -> fake_Y -> fake_YX
		self.fake_YX = self.generatorX(self.fake_Y, reuse=True)

		self.d_fake_X = self.discriminatorX(self.fake_X, reuse=False)
		self.d_fake_Y = self.discriminatorY(self.fake_Y, reuse=False)

		self.d_real_X = self.discriminatorX(self.real_X, reuse=False)
		self.d_real_Y = self.discriminatorY(self.real_Y, reuse=False)

		# GAN Loss
		# generate X
		self.loss_gan_generator_X = tf.reduce_mean(tf.squared_difference(self.d_fake_Y, tf.ones_like(self.d_fake_Y)))
		self.loss_gan_discriminator_Y = tf.reduce_mean(tf.squared_difference(self.d_real_Y, tf.ones_like(self.real_Y)))
		self.loss_gan_generator_Y = tf.reduce_mean(tf.squared_difference(self.d_fake_X, tf.ones_like(self.d_fake_X)))
		self.loss_gan_discriminator_X = tf.reduce_mean(tf.squared_difference(self.d_real_X, tf.ones_like(self.real_X)))

		# Cycle Consistency Loss
		self.loss_cycle = self.lambda1 * tf.reduce_mean(tf.abs(self.fake_X - self.real_X)) + self.lambda2 * tf.reduce_mean(tf.abs(self.fake_Y - self.real_Y))

		# Generator_X Loss
		self.loss_generator_X = self.loss_gan_generator_X + self.loss_cycle
		# Generator_Y Loss
		self.loss_generator_Y = self.loss_gan_generator_Y + self.loss_cycle
		# Discriminator_X Loss
		self.loss_discriminator_X = self.loss_gan_discriminator_X + tf.reduce_mean(tf.square(self.d_fake_X))
		# Discriminator_Y Loss
		self.loss_discriminator_Y = self.loss_gan_discriminator_Y + tf.reduce_mean(tf.square(self.d_fake_Y))

		self.variables_generator_X = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator_X")
		self.variables_generator_Y = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator_Y")
		self.variables_discriminator_X = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator_X")
		self.variables_discriminator_Y = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator_Y")

	def _optimize(self, loss, variables, name="Adam"):
		optimize = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=self.beta1, name=name).minimize(loss, var_list=variables)



	def _init_optimizer(self, loss, name="Adam"):
		self.optimize_generator_X = _optimizer(self.loss_generator_X, self.variables_generator_X, name="Adam_generator_X")
		self.optimize_generator_Y = _optimizer(self.loss_generator_Y, self.variables_generator_Y, name="Adam_generator_Y")
		self.optimize_discriminator_X = _optimizer(self.loss_discriminator_X, self.variables_discriminator_X, name="Adam_discriminator_X")
		self.optimize_discriminator_Y = _optimizer(self.loss_discriminator_Y, self.variables_discriminator_Y, name="Adam_discriminator_Y")

	def train(self, epoches):
		



