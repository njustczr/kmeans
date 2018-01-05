import tensorflow as tf
init = tf.truncated_normal([1,9],stddev=0.1)
value = tf.Variable(init)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

result = tf.reshape(value,[3,3])
result2 = tf.reshape(result,[-1,9])


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(value))
    print("===========================")
    print(sess.run(result))
    print("===========================")
    print(sess.run(result2))
