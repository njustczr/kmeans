from sklearn.cluster import Birch
from sklearn.cluster import KMeans
from sklearn.cluster import spectral_clustering
from sklearn import metrics
from numpy import mat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import _thread

def read_points():
    dataset = []
    with open('D:\data.csv', 'r', encoding='utf8') as file:
        for line in file:
            if (line == '\n'):
                continue
            result = line.strip().split(',')
            #移除 公司代码 日期 会计规则和专家规则的结果列
            result.remove(result[0])
            result.remove(result[0])
            #result1=result.remove(result[0])
            #删除 资产期末余额 和 资产年初余额
            #result.remove(result[0])
            #result.remove(result[0])
            fltline = [float(i) for i in result]
            dataset.append(fltline)
        file.close()
        return dataset

def saveresult(temp):
    with open('D:\Result.csv', 'a') as file:
        file.write(temp)
        file.close()

def Dimensionality_reduction(matrix):
    #最后两列统计结果矩阵
    Y = len(matrix)
    add_matrix = np.zeros([Y,2])
    for y in range(Y):
        add_matrix[y, 0] = matrix[y, 0] - matrix[y, 1]
        add_matrix[y, 1] = matrix[y, 2] - matrix[y, 3]
        #add_matrix[y, 2] = matrix[y, 4] - matrix[y, 5]
    #X_tsne = TSNE(learning_rate=100).fit_transform(matrix)
    #X_pca = PCA(n_components=2).fit_transform(matrix)
    X_tsne = add_matrix
    X_pca = add_matrix
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.title("T-SNE")
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1],marker='.')
    plt.subplot(122)
    plt.title("PCA")
    plt.scatter(X_pca[:, 0], X_pca[:, 1],marker='.')
    print("max value: " ,np.amax(X_pca,axis=0))
    print("min value: " ,np.min(X_pca))
    print("max position: ",np.where(X_pca == np.amax(X_pca,axis=0)))
    print("min position: ",np.where(X_pca == np.min(X_pca)))
    plt.show()

def draw_picture(matrix2list,y_pred):
    pass
#    X_tsne = TSNE(learning_rate=100).fit_transform(matrix2list)
#    X_pca = PCA().fit_transform(matrix2list)

#    plt.close()
#    fig = plt.figure()
#    plt.ion()  # interactive mode on
    #        plt.figure(figsize=(10, 5))
#    plt.subplot(121)
#    plt.title("T-SNE")
#    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_pred)
#    plt.subplot(122)
#    plt.title("PCA")
#    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred)
#    plt.pause(1)


def main():
    fd = open("D:\calculate.txt", 'a')
    csvdata = read_points()
    X = len(csvdata[0])
    Y = len(csvdata)
    New_matrix = np.zeros([Y,X])
    for y in range(Y):
        for x in range(X):
            New_matrix[y, x] = csvdata[y][x]

    temp_mean = New_matrix[:, 0].mean()
    col_std = np.std(New_matrix, axis=0)
    col_mean = np.mean(New_matrix, axis=0)
    col_min = np.min(New_matrix, axis=0)
    col_max = np.max(New_matrix, axis=0)

    #增加利润 营收后的矩阵
    New = np.zeros([Y,5])
    for y in range(Y):
        New[y, 0] = (New_matrix[y, 0] - col_mean[0])/col_std[0];
        New[y, 1] = (New_matrix[y, 0] - New_matrix[y, 1])/New_matrix[y, 1];
        New[y, 2] = (New_matrix[y, 2] - New_matrix[y, 3])/New_matrix[y, 3];
        New[y, 3] = (New_matrix[y, 6] - New_matrix[y, 7])/New_matrix[y, 7];
        New[y, 4] = (New_matrix[y, 8] - New_matrix[y, 9])/New_matrix[y, 9];


    #===========================================
    # 需要对csvdata进行中心化和标准化处理
#    for y in range(Y):
#        for x in range(X):
#            New_matrix[y, x] -= col_mean[x]
#            New_matrix[y, x] /= col_std[x]
    #============================================
    #============================================
    # min max 归一化处理
#    for y in range(Y):
#        for x in range(X):
#            New_matrix[y,x] = (New_matrix[y,x] - col_min[x])/(col_max[x] - col_min[x])
    #============================================
    '''
    scale_matrix = np.zeros([Y,Y])
    for y in range(Y):
        for x in range(Y):
            if x == y:
                scale_matrix[y,x] = New_matrix[y,0]
            else:
                scale_matrix[y,x] = 0
    #for y in range(Y):
        #temp = str(New_matrix[y,0]) + "," + str(New_matrix[y,1]) + "," + str(New_matrix[y,2]) + "," + str(New_matrix[y,3]) + "," + str(New_matrix[y,4]) + "," + str(New_matrix[y,5]) + "\n"
        #saveresult(temp)
    scale_M = mat(scale_matrix)
    right_M = mat(New_matrix)
    cluster_M = scale_M.I*right_M
    #cluster_list = cluster_M.tolist()
    cluster_array = cluster_M.getA()

    compute_array = np.zeros([Y,5])
    for y in range(Y):
        for x in range(X):
            if x == 0:
                pass
            else:
                compute_array[y,x-1] = cluster_array[y,x]

    matrix2list = New_matrix.tolist()
    '''
#    print(matrix2list)
#    print(New_matrix)
    #聚类前降维显示数据
#    Dimensionality_reduction(New_matrix)

    all_vrc = []
    all_silh = []
    sub = []

    for k in range(20):
        # kmeans聚类
        if k==0 or k==1:
            continue
        clf = KMeans(n_clusters=k,init='k-means++')
        #spectral_clustering()
        #y_pred = clf.fit_predict(matrix2list)
        y_pred = clf.fit_predict(New.tolist())
        print("====================")
        print(clf.cluster_centers_)
#        print(clf)
#        print(y_pred)

        sub.append(k)
        #VRC = metrics.calinski_harabaz_score(New_matrix, y_pred)
        VRC = metrics.calinski_harabaz_score(New, y_pred)
        all_vrc.append(VRC)
        silh = metrics.silhouette_score(New, y_pred, metric='euclidean')
        all_silh.append(silh)
        print("k= ",k)
        print("k= "+str(k),file=fd)
        print('VRC方差率：',VRC)
        print("VRC方差率："+str(VRC),file=fd)
        print('轮廓系数：%10.3f' % silh)
        print('轮廓系数：', silh)
        print("轮廓系数："+str(silh),file=fd)

        lines = len(y_pred)
        static_result = np.zeros([1, k])
        first = 0
        second = 0
        third = 0

        for item in y_pred:
           for i in range(0,k):
               if item == i:
                   static_result[0,i] += 1
        '''
        for i , j in enumerate(y_pred):
            if j == 1:
                print(str(i)+","+str(j)+",%.3f,%.3f,%.3f,%.3f,%.3f" % (New[i,0],New[i,1],New[i,2],New[i,3],New[i,4]))
        print("==================================")
        for i , j in enumerate(y_pred):
            if j == 2:
                print(str(i)+","+str(j)+",%.3f,%.3f,%.3f,%.3f,%.3f" % (New[i,0],New[i,1],New[i,2],New[i,3],New[i,4]))
        print("==================================")
        for i , j in enumerate(y_pred):
            if j == 0:
                print(str(i)+","+str(j)+",%.3f,%.3f,%.3f,%.3f,%.3f" % (New[i,0],New[i,1],New[i,2],New[i,3],New[i,4]))
        '''
        for m in range(k):
            for i, j in enumerate(y_pred):
                if j == m:
                    print(str(i) + "," + str(j) + ",%.3f,%.3f,%.3f,%.3f,%.3f" % (New[i, 0], New[i, 1], New[i, 2], New[i, 3], New[i, 4]))
                    print(str(i)+","+str(j)+","+str(New[i,0])+","+str(New[i,1])+","+str(New[i,2])+","+str(New[i,3])+","+str(New[i,4]),file=fd)
            print("==================================")
            print("==================================",file=fd)

        for i in range(0,k):
            print("第"+str(i)+"类数据占比: "+str(static_result[0,i]*100/lines)+"%")
            print("第"+str(i)+"类数据占比: "+str(static_result[0,i]*100/lines)+"%",file=fd)

 #       for item in y_pred:
 #           if item == 0:
 #               first += 1
 #           if item == 1:
 #               second += 1
 #           if item == 2:
 #               third += 1

#        try:
#            print("Start: " + str(k))
#            _thread.start_new_thread(draw_picture,(matrix2list,y_pred,))
#        except:
#            print("Error:unable to start thread")
        #降维显示数据
        #X_tsne = TSNE(learning_rate=100).fit_transform(matrix2list)
        X_tsne = TSNE(learning_rate=100).fit_transform(New.tolist())
        #X_pca = PCA().fit_transform(matrix2list)
        X_pca = PCA(n_components=2).fit_transform(New.tolist())

        plt.close()
        fig = plt.figure()
        plt.ion()  # interactive mode on
#        plt.subplot(121)
#        plt.title("T-SNE")
#        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_pred, marker='.')
#        plt.subplot(122)
        plt.title("PCA")
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, marker='.')
        print("max value: ", np.amax(X_pca, axis=0))
        print("min value: ", np.amin(X_pca, axis=0))
        print("max position: ", np.where(X_pca == np.amax(X_pca, axis=0)))
        print("min position: ", np.where(X_pca == np.amin(X_pca, axis=0)))
        plt.pause(1)



    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.title("VRC")
#    plt.scatter(sub,all_vrc,marker='o')
    plt.plot(all_vrc)
    plt.subplot(122)
    plt.title("silh")
#    plt.scatter(sub,all_silh,marker='x')
    plt.plot(all_silh)
    plt.show()


#    print("第一类数据占比：" + str(((first * 100) / lines)) + "%")
#    print("第二类数据占比：" + str(((second * 100) / lines)) + "%")
#    print("第三类数据占比：" + str(((third * 100) / lines)) + "%")
#    temp = "第一类数据占总数的:" + str(((first * 100) / lines)) + "%\n" + "第二类数据占总数的:" + str(((second * 100) / lines)) + "%\n" + "第三类数据占总数的:" + str(((third * 100) / lines)) + "%\n"
#    saveresult(temp)

#    X_tsne = TSNE(learning_rate=100).fit_transform(matrix2list)
#    X_pca = PCA().fit_transform(matrix2list)
#    plt.figure(figsize=(10, 5))
#    plt.subplot(121)
#    plt.title("T-SNE")
#    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_pred)
#    plt.subplot(122)
#    plt.title("PCA")
#    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred)
#    plt.show()

if __name__ == "__main__":
    main()
    pass




"""
csvdata = read_points()
X = len(csvdata[0])
Y = len(csvdata)
New_matrix = np.zeros([Y,X])
for y in range(Y):
    for x in range(X):
        New_matrix[y,x] = csvdata[y][x]

temp_mean = New_matrix[:,0].mean()
col_std = np.std(New_matrix,axis=0)
col_mean = np.mean(New_matrix,axis=0)
#需要对csvdata进行中心化和标准化处理
for y in range(Y):
    for x in range(X):
        New_matrix[y,x]-=col_mean[x]
        New_matrix[y,x]/=col_std[x]

matrix2list = New_matrix.tolist()

X = [[0.0888, 0.5885],
     [0.1399, 0.8291],
     [0.0747, 0.4974],
     [0.0983, 0.5772],
     [0.1276, 0.5703],
     [0.1671, 0.5835],
     [0.1906, 0.5276],
     [0.1061, 0.5523],
     [0.2446, 0.4007],
     [0.1670, 0.4770],
     [0.2485, 0.4313],
     [0.1227, 0.4909],
     [0.1240, 0.5668],
     [0.1461, 0.5113],
     [0.2315, 0.3788],
     [0.0494, 0.5590],
     [0.1107, 0.4799],
     [0.2521, 0.5735],
     [0.1007, 0.6318],
     [0.1067, 0.4326],
     [0.1956, 0.4280]
    ]
print(X)
#kmeans聚类
clf = KMeans(n_clusters=3)
y_pred = clf.fit_predict(matrix2list)
print(clf)
print(y_pred)

lines = len(y_pred)
first = 0
second = 0
third = 0
for item in y_pred:
    if item == 0:
        first += 1
    if item == 1:
        second += 1
    if item == 2:
        third += 1

print("第一类数据占比：" + str(((first*100)/lines)) + "%")
print("第二类数据占比：" + str(((second*100)/lines)) + "%")
print("第三类数据占比：" + str(((third*100)/lines)) + "%")
"""



"""
import numpy as np
import matplotlib.pyplot as plt
x = [n[0] for n in X]
print(x)
y = [n[1] for n in X]
print(y)

#可视化操作
plt.scatter(x,y,c=y_pred,marker='x')
plt.title("kmeans basketball data")
plt.xlabel("assists_per_minute")
plt.ylabel("points_per_minute")
plt.legend(["Rank"])
plt.show()
"""