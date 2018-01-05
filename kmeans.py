import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as py
import tkinter
import time
import matplotlib.animation as animation

from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets.samples_generator import make_circles

K = 4 #最大类别数
MAX_ITERS = 1000 #最大迭代次数
N = 200 #样本点数目
centers = [[-2,-2],[-2,1.5],[1.5,-2],[2,1.5]] #簇中心
print (centers)
print ("####################")
#生成人工数据
data,features = make_blobs(n_samples=N, centers=centers, n_features=2, cluster_std=0.8, shuffle=False, random_state=42)
#print (data)
#print ("====================")
#print (features)

#plt.scatter(data[:,0],data[:,1],c=features)
#plt.show()

#计算类内平均值函数
def clusterMean(data,id,num):
    total = tf.unsorted_segment_sum(data,id,num)
    count = tf.unsorted_segment_sum(tf.ones_like(data),id,num)
    return total/count

#构建graph
points = tf.Variable(data)
cluster = tf.Variable(tf.zeros([N],dtype=tf.int64))

centers = tf.Variable(tf.slice(points.initialized_value(),[0,0],[K,2])) #将原始数据的前k个点当做初始中心

repCenters = tf.reshape(tf.tile(centers,[N,1]),[N,K,2])
repPoints = tf.reshape(tf.tile(points,[1,K]),[N,K,2])
sumSqure = tf.reduce_sum(tf.square(repCenters-repPoints),reduction_indices=2)#计算距离
bestCenter = tf.argmin(sumSqure,axis=1)
change = tf.reduce_any(tf.not_equal(bestCenter,cluster))
means = clusterMean(points,bestCenter,K)

with tf.control_dependencies([change]):
    update = tf.group(centers.assign(means),cluster.assign(bestCenter))

plt.close()  # clf() # 清图  cla() # 清坐标轴 close() # 关窗口
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.ion()  # interactive mode on

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print (sess.run(centers))
    print ("-----------------")
    print (sess.run(sumSqure))
    print ("-----------------")
    print (sess.run(bestCenter))
    changed = True
    iterNum = 0
    while changed and iterNum < MAX_ITERS:
        iterNum += 1
        # 运行graph
        [changed, _] = sess.run([change, update])
        [centersArr, clusterArr] = sess.run([centers, cluster])
        print (iterNum)

        # 显示图像
        #fig, ax = plt.subplots()
        try:
            ax.scatter(data.transpose()[0], data.transpose()[1], marker='o', s=100, c=clusterArr)
            plt.pause(1)
        except Exception as err:
            print(err)