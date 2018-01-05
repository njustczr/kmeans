import tensorflow as tf
import numpy as np

#使用numpy生成假数据
x_data = np.float32(np.random.rand(2,100))
print (x_data)
y_data = np.dot([0.100,0.200],x_data)+0.300

#构造一个线性模型
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1,2],-1,1)) #shape 均匀分布
y = tf.matmul(W,x_data)+b

#最小化方差
loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

#初始化变量
init = tf.initialize_all_variables()

#保存变量
saver = tf.train.Saver()

#启动图
sess = tf.Session()
sess.run(init)

save_path = saver.save(sess,"D:\pycharm\model.ckpt")
print ("Model saved in file: ",save_path)

#拟合平面
for step in range(0,201):
    sess.run(train)
    print (step, sess.run(loss), sess.run(W), sess.run(b))
