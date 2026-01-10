import tensorflow as tf
import matplotlib.pyplot as plt

a = tf.constant([1.0, -0.5, 3.4, -2.1, 0.0, -6.5], dtype=tf.float32)

b = tf.nn.relu(a, name='ReLU')

tf.compat.v1.disable_eager_execution()


# Initiating a Tensorflow session
with tf.compat.v1.Session() as sess:
    print('Input type:', a)
    print('Input:', sess.run(a))
    print('Return type:', b)
    print('Output:', sess.run(b))
    plt.plot(sess.run(b))
    plt.grid()
    plt.show()