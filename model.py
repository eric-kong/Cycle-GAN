import tensorflow as tf 
from generator import Generator
from discriminator import Discriminator 
import os
import numpy as np
import math
from utils import ImagePool, load_image, convert2image, save_images
import time, datetime

class CycleGAN:
	def __init__(self, sess, name="cylegan", dataset="horse2zebra", image_size=256, batch_size=1, ngf=64, ndf=64, lambda1=10, lambda2=10, beta1=0.5):
		self.image_size = image_size
		self.ngf = ngf
		self.ndf = ndf

		self.lambda1 = lambda1

		self.lambda2 = lambda2

		# adam 
		self.beta1 = beta1

		self.batch_size = batch_size


		self.sess = sess

		self.dataset = dataset

		self.pool_fake_X = ImagePool()

		self.pool_fake_Y = ImagePool()

		# generate images in the domain X
		self.generator_X = Generator(name="generator_X", ngf=self.ngf, image_size=image_size)
		# generate images in the domain Y
		self.generator_Y = Generator(name="generator_Y", ngf=self.ngf, image_size=image_size)
		# discriminate images in the domain X
		self.discriminator_X = Discriminator(name="discriminator_X", ndf=self.ndf)
		# discriminate images in the domain Y
		self.discriminator_Y = Discriminator(name="discriminator_Y", ndf=self.ndf)

		self._build_model()
		print ("[SUCCEED] build up model")
		self._init_optimizer()
		print ("[SUCCEED] initialize optimizers")

	def _build_model(self):
		self.real_X = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name="real_X")
		self.real_Y = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name="real_Y")
		
		# feed to discriminator
		self.fake_X = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name="fake_X")
		self.fake_Y = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name="fake_Y")

		self.g_fake_X = self.generator_X(self.real_Y, reuse=False)
		self.g_fake_Y = self.generator_Y(self.real_X, reuse=False)

		# real_Y -> fake_X -> fake_XY
		fake_XY = self.generator_Y(self.g_fake_X, reuse=True)
		# real_X -> fake_Y -> fake_YX
		fake_YX = self.generator_X(self.g_fake_Y, reuse=True)

		d_fake_X = self.discriminator_X(self.fake_X, reuse=False)
		d_fake_Y = self.discriminator_Y(self.fake_Y, reuse=False)

		d_g_fake_X = self.discriminator_X(self.g_fake_X, reuse=True)
		d_g_fake_Y = self.discriminator_Y(self.g_fake_Y, reuse=True)

		d_real_X = self.discriminator_X(self.real_X, reuse=True)
		d_real_Y = self.discriminator_Y(self.real_Y, reuse=True)

		# GAN Loss
		# generate X
		self.loss_gan_generator_X = tf.reduce_mean(tf.squared_difference(d_g_fake_X, tf.ones_like(d_g_fake_X)))
		self.loss_gan_discriminator_Y = tf.reduce_mean(tf.squared_difference(d_real_Y, tf.ones_like(d_real_Y)))
		self.loss_gan_generator_Y = tf.reduce_mean(tf.squared_difference(d_g_fake_Y, tf.ones_like(d_g_fake_Y)))
		self.loss_gan_discriminator_X = tf.reduce_mean(tf.squared_difference(d_real_X, tf.ones_like(d_real_X)))

		# Cycle Consistency Loss
		self.loss_cycle = self.lambda1 * tf.reduce_mean(tf.abs(fake_YX - self.real_X)) + self.lambda2 * tf.reduce_mean(tf.abs(fake_XY - self.real_Y))

		# Generator_X Loss
		self.loss_generator_X = self.loss_gan_generator_X + self.loss_cycle
		# Generator_Y Loss
		self.loss_generator_Y = self.loss_gan_generator_Y + self.loss_cycle
		# Discriminator_X Loss
		self.loss_discriminator_X = self.loss_gan_discriminator_X + tf.reduce_mean(tf.square(d_fake_X))
		# Discriminator_Y Loss
		self.loss_discriminator_Y = self.loss_gan_discriminator_Y + tf.reduce_mean(tf.square(d_fake_Y))

		self.variables_generator_X = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator_X")
		self.variables_generator_Y = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator_Y")
		self.variables_discriminator_X = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator_X")
		self.variables_discriminator_Y = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator_Y")

		# Tensorboard Setup
		sum_dxr = tf.summary.histogram("D_X_real", d_real_X)
		sum_dyr = tf.summary.histogram("D_Y_real", d_real_Y)
		sum_dxf = tf.summary.histogram("D_X_fake", d_g_fake_X)
		sum_dyf = tf.summary.histogram("D_Y_fake", d_g_fake_Y)

		sum_lgx = tf.summary.scalar("loss_G_X", self.loss_generator_X)
		sum_lgy = tf.summary.scalar("loss_G_Y", self.loss_generator_Y)
		sum_ldx = tf.summary.scalar("loss_D_X", self.loss_discriminator_X)
		sum_ldy = tf.summary.scalar("loss_D_Y", self.loss_discriminator_Y)

		sum_fx = tf.summary.image("fake_X", convert2image(self.g_fake_X))
		sum_fy = tf.summary.image("fake_Y", convert2image(self.g_fake_Y))
		sum_cx = tf.summary.image("cycle_X", convert2image(fake_YX))
		sum_cy = tf.summary.image("cycle_Y", convert2image(fake_XY))

		self.g_sum = tf.summary.merge(
			[sum_dxr, sum_dyr, sum_dxf, sum_dyf, sum_lgx, sum_lgy, sum_fx, sum_fy, sum_cx, sum_cy])

		self.d_sum = tf.summary.merge([sum_ldx, sum_ldy])




	def _optimize(self, loss, variables, name="Adam"):
		optimize = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta1, name=name).minimize(loss, var_list=variables)
		return optimize

	def _init_optimizer(self):
		self.lr = tf.placeholder(tf.float32, None, "learning_rate")
		self.optimize_generator_X = self._optimize(self.loss_generator_X, self.variables_generator_X, name="adam_generator_X")
		self.optimize_generator_Y = self._optimize(self.loss_generator_Y, self.variables_generator_Y, name="adam_generator_Y")
		self.optimize_discriminator_X = self._optimize(self.loss_discriminator_X, self.variables_discriminator_X, name="adam_discriminator_X")
		self.optimize_discriminator_Y = self._optimize(self.loss_discriminator_Y, self.variables_discriminator_Y, name="adam_discriminator_Y")
		self.lr_sum = tf.summary.scalar("learning rate", self.lr)

	def train(self, epoches=200, lr=2e-4, load_model=None, checkpoint_dir=None, save_freq=0.25):
		path_X = "./dataset/" + self.dataset + "/trainA/"
		path_Y = "./dataset/" + self.dataset + "/trainB/"
		data_X = [path_X + file for file in os.listdir(path_X)]
		data_Y = [path_Y + file for file in os.listdir(path_Y)]
		current_datetime = datetime.datetime.now()
		checkpoints_dir = "./checkpoints/" + self.dataset + "_" + current_datetime.strftime("%Y%m%d-%H%M") + "/"
		if not os.path.exists(checkpoints_dir):
			os.makedirs(checkpoints_dir)
		self.writer = tf.summary.FileWriter(checkpoints_dir, self.sess.graph)
		self.saver = tf.train.Saver()

		if load_model is not None:
			if self.load(checkpoint_dir):
				print ("[SUCCEED] Load Model")	
			else:
				print ("[FAIL] Load Model")

		self.sess.run(tf.global_variables_initializer())

		step = 1
		lr_original = lr
		self.epoch = tf.placeholder(tf.int32, None, "epoch")
		self.epoch_sum = tf.summary.scalar("epoch", self.epoch)
		self.statistics_sum = tf.summary.merge([self.lr_sum, self.epoch_sum])
		for epoch in range(1, epoches + 1):
			start_time = time.time()
			print ("-----------------------------")
			print ("[*] {}th epoch:".format(epoch))
			# update learning rate, first half the same, second half linear decay
			if epoch > epoches / 2:
				lr = lr_original * (epoches - epoch) / (epoches / 2)

			np.random.shuffle(data_X)
			np.random.shuffle(data_Y)

			# TO CHECK: ceil or floor
			batch_num = math.ceil(min(len(data_X), len(data_Y)) / self.batch_size)
			print ("[*] {} batches to train".format(batch_num))
			for i in range(0, batch_num):
				if i % 50 == 0:
					print ("[**] {}th batch".format(i))
				batch_X = np.array([load_image(image_path) for image_path in data_X[i * self.batch_size : (i + 1) * self.batch_size]]).astype(np.float32)
				batch_Y = np.array([load_image(image_path) for image_path in data_Y[i * self.batch_size : (i + 1) * self.batch_size]]).astype(np.float32)
				g_fake_X, g_fake_Y, _, _, g_sum, statistics_sum = self.sess.run(
					[self.g_fake_X, self.g_fake_Y, self.optimize_generator_X, self.optimize_generator_Y, self.g_sum, self.statistics_sum],
					feed_dict={self.real_X: batch_X, self.real_Y: batch_Y, self.lr: lr, self.epoch: epoch})
				fake_X = self.pool_fake_X(g_fake_X)
				fake_Y = self.pool_fake_Y(g_fake_Y)
				_, _, d_sum = self.sess.run(
					[self.optimize_discriminator_X, self.optimize_discriminator_Y, self.d_sum],
					feed_dict={self.real_X: batch_X, self.real_Y: batch_Y, self.fake_X: fake_X, self.fake_Y: fake_Y, self.lr: lr})

				self.writer.add_summary(g_sum, step)
				self.writer.add_summary(d_sum, step)
				self.writer.add_summary(statistics_sum, step)
				step += 1

			print ("[*] Time Consumed: {} mins".format((time.time() - start_time)/60))
			self.store_temporal_results(checkpoints_dir, epoch, 5) # store 5 batches images
			print ("[SUCCEED] Saved Temporal Fake Images in {}".format(checkpoints_dir))
			if epoch % (0.25 * epoches) == 0:
				self.save_model(checkpoints_dir, epoch)
				print ("[SUCCEED] Saved Model in {}th epoch in {}".format(epoch, checkpoints_dir))

	def store_temporal_results(self, output_dir, epoch, num_batch):
		path_X = "./dataset/" + self.dataset + "/trainA/"
		path_Y = "./dataset/" + self.dataset + "/trainB/"
		data_X = [path_X + file for file in os.listdir(path_X)]
		data_Y = [path_Y + file for file in os.listdir(path_Y)]
		np.random.shuffle(data_X)
		np.random.shuffle(data_Y)
		batch_X = np.array([load_image(image_path) for image_path in data_X[: self.batch_size * num_batch]]).astype(np.float32)
		batch_Y = np.array([load_image(image_path) for image_path in data_Y[: self.batch_size * num_batch]]).astype(np.float32)

		fake_X, fake_Y = self.sess.run(
			[self.g_fake_X, self.g_fake_Y],
			feed_dict={self.real_X: batch_X, self.real_Y: batch_Y})

		fake_X = convert2image(fake_X)
		fake_Y = convert2image(fake_Y)


		directory = output_dir + "tem_results/"
		if not os.path.exists(directory):
			os.makedirs(directory)

		save_images(directory + "X_" + str(epoch) + "_", fake_X, num_batch)
		save_images(directory + "Y_" + str(epoch) + "_", fake_Y, num_batch)

	def save_model(self, output_dir, epoch):
		model_name = "model.ckpt"
		directory = output_dir + "models/"
		if not os.path.exists(directory):
			os.makedirs(directory)
		self.saver.save(self.sess, directory + model_name, epoch)

	def load_model(self, checkpoint_dir):
		directory = checkpoint_dir + "models/"
		ckpt = tf.train.get_checkpoint_state(directory)
		if ckpt and ckpt.model_checkpoint_path:
			meta_graph_path = ckpt.model_checkpoint_path + ".meta"
			restore = tf.train.import_meta_graph(meta_graph_path)
			restore.restore(self.sess, tf.train.latest_checkpoint(directory))
			return True
		else:
			return False


