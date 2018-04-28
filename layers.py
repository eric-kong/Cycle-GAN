import tensorflow as tf 

# Naming follows the format in https://github.com/jcjohnson/fast-neural-style
def c7s1_k(input, k, use_norm=True, use_relu=True, name="c7s1_k"):
	with tf.variable_scope(name):
		filters = tf.get_variable("filters", 
			shape=[7, 7, input.get_shape()[3], k],
			initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02, dtype=tf.float32))
		# 3 = (7 - 1) / 2
		padded = tf.pad(input, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
		conv = tf.nn.conv2d(padded, filters, strides=[1, 1, 1, 1], padding="VALID")
		if use_norm:
			norm = instance_norm(conv)
		else:
			norm = conv
		if use_relu:
			output = tf.nn.relu(norm)
		else:
			output = tf.nn.tanh(norm)

		return output

def dk(input, k, name="dk"):
	with tf.variable_scope(name):
		filters = tf.get_variable("filters",
			shape=[3, 3, input.get_shape()[3], k],
			initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02, dtype=tf.float32))
		# 1 = (3 - 1) / 2
		padded = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
		conv = tf.nn.conv2d(padded, filters, strides=[1, 2, 2, 1], padding="VALID")
		norm = instance_norm(conv)
		relu = tf.nn.relu(norm)

		return relu

def Rk(input, k, name="Rk"):
	with tf.variable_scope(name):
		filters1 = tf.get_variable("filters1",
			shape=[3, 3, input.get_shape()[3], k],
			initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02, dtype=tf.float32))
		padded1 = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT", name="pad1")
		conv1 = tf.nn.conv2d(padded1, filters1, strides=[1, 1, 1, 1], padding="VALID", name="conv1")
		norm1 = instance_norm(conv1, name="norm1")
		relu1 = tf.nn.relu(norm1, name="relu1")

		filters2 = tf.get_variable("filters2",
			shape=[3, 3, input.get_shape()[3], k],
			initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02, dtype=tf.float32))
		padded2 = tf.pad(relu1, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT", name="pad2")
		conv2 = tf.nn.conv2d(padded2, filters2, strides=[1, 1, 1, 1], padding="VALID", name="conv2")
		norm2 = instance_norm(conv2, name="norm2")
		relu2 = tf.nn.relu(norm2, name="relu2")

		return relu2

def uk(input, k, output_size=None, name="uk"):
	with tf.variable_scope(name):
		filters = tf.get_variable("filters",
			shape=[3, 3, k, input.get_shape()[3]])
		if not output_size:
			output_size = input.get_shape()[1] * 2
		deconv = tf.nn.conv2d_transpose(input, filters, 
			output_shape=tf.stack([tf.shape(input)[0], output_size, output_size, k]),
			strides=[1, 2, 2, 1],
			padding="SAME", name="deconv")
		norm = instance_norm(deconv)
		relu = tf.nn.relu(norm)

		return relu


def instance_norm(input, name="instance_norm"):
	with tf.variable_scope(name):
		depth = input.get_shape()[3]
		scale = tf.get_variable("scale", 
			[depth], 
			initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02, dtype=tf.float32))
		offset = tf.get_variable("offset",
			[depth],
			initializer=tf.constant_initializer(0.0))
		mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
		epsilon = 1e-5
		inv = tf.rsqrt(variance + epsilon)
		normalized = (input - mean) * inv
		return normalized * scale + offset


# for discriminator
def Ck(input, k, use_norm=True, use_relu=True, name="Ck"):
	with tf.variable_scope(name):
		filters = tf.get_variable("filters",
			[4, 4, input.get_shape()[3], k],
			initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02, dtype=tf.float32))
		conv = tf.nn.conv2d(input, filters, strides=[1, 2, 2, 1], padding="SAME")
		if use_norm:
			norm = instance_norm(conv)
		else:
			norm = conv
		# Leaky ReLu
		if use_relu:
			lrelu = tf.maximum(norm, 0.2 * norm)
		else:
			lrelu = norm

		return lrelu
