import tensorflow as tf
import numpy as np

x = np.array(
[ [0,0,1],
  [0,1,1],
  [1,0,1],
  [1,1,1] ]) #4*3  
y = np.array([[0,1,1,0]]).T #4*1

matrixX = tf.placeholder(tf.float32, shape=(4, 3))
matrixY = tf.placeholder(tf.float32, shape=(4, 1))

w1 = tf.Variable(tf.random_normal([3, 4]))
w2 = tf.Variable(tf.random_normal([4, 1]))

l1 = tf.nn.sigmoid(tf.matmul(matrixX, w1))
l2 = tf.nn.sigmoid(tf.matmul(l1, w2))

err2 = tf.multiply(tf.subtract(matrixY, l2), tf.multiply(l2, tf.subtract(1.0, l2))) 
err1 = tf.multiply(tf.matmul(err2, tf.transpose(w2)), tf.multiply(l1, tf.subtract(1.0, l1)))

u2 = tf.assign(w2, tf.add(w2, tf.matmul(tf.transpose(l1), err2)))
u1 = tf.assign(w1, tf.add(w1, tf.matmul(tf.transpose(matrixX), err1)))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#sess.run(l1, feed_dict={matrixX:x})
for i in range(60000):
        sess.run(u2, feed_dict={matrixX:x, matrixY:y})
        sess.run(u1, feed_dict={matrixX:x, matrixY:y})
print(sess.run(l2, feed_dict={matrixX:x, matrixY:y}))