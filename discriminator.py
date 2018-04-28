import tensorflow as tf 
import layers

class Discriminator:
	def __init__(self, name="discriminator", ndf=64):
		self.name = name
		self.ndf = ndf

	def __call__(self, input, reuse):
		with tf.variable_scope(self.name):
			if reuse:
				tf.get_variable_scope().reuse_variables()
			else:
				assert tf.get_variable_scope().reuse is False

			C64 = layers.Ck(input, 64, use_norm=False, name="C64")
			C128 = layers.Ck(C64, 128, use_norm=True, name="C128")
			C256 = layers.Ck(C128, 256, use_norm=True, name="C256")
			C512 = layers.Ck(C256, 512, use_norm=True, name="C512")

			conv = layers.Ck(C512, 1, use_norm=False, use_relu=False, name="C1norelu") # Different
			biases = tf.get_variable("biases", [1], initializer=tf.constant_initializer(0.0))
			sigmoid = tf.sigmoid(conv + biases) # TODO: Test whether bias is needed

			return sigmoid
