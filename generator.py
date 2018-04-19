import tensorflow as tf
import layers

class Generator:
	def __init__(self, name="generator", ngf, image_size=128):
		self.name = name
		self.ngf = ngf
		self.image_size = image_size

	def __call__(self, input, reuse):
		with tf.variable_scope(self.name):
			if reuse:
				tf.get_variable_scope().reuse_variables()
			else:
				assert tf.get_variable_scope().reuse is False
				
			c7s1_32 = layers.c7s1_k(input, self.ngf, name="c7s1_32")
			d64 = layers.dk(c7s1_32, 2 * self.ngf, name="d64")
			d128 = layers.dk(d64, 4 * self.ngf, name="d128")

			block_num = 6 if image_size <= 128 else 9
			res_input = d128
			for i in range(block_num):
				res_output = layers.Rk(res_input, 4 * self.ngf, name="R{}".format(i))
				res_input = res_output

			u64 = layers.uk(res_output, 2 * self.ngf, name="u64")
			u32 = layers.uk(u64, self.ngf, output_size=self.image_size)

			return u32


