from sklearn.cluster import Birch
from sklearn.cluster import KMeans
from sklearn import metrics
from numpy import mat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import Canopy as ca

def read_points():
    dataset = []
    with open('D:\debt1.csv', 'r', encoding='utf8') as file:
        for line in file:
            if (line == '\n'):
                continue
            result = line.strip().split(',')
            #移除 公司代码 日期 会计规则和专家规则的结果列
            result.remove(result[0])
            result.remove(result[0])
            result1=result.remove(result[0])
            fltline = [float(i) for i in result]
            dataset.append(fltline)
        file.close()
        return dataset

def saveresult(temp):
    with open('D:\Result.csv', 'a') as file:
        file.write(temp)
        file.close()

def Dimensionality_reduction(matrix):
    X_tsne = TSNE(learning_rate=100).fit_transform(matrix)
    X_pca = PCA(n_components=2).fit_transform(matrix)
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

def showCanopy(canopies, dataset, t1, t2):
    fig = plt.figure()
    sc = fig.add_subplot(111)
    colors = ['brown', 'green', 'blue', 'y', 'r', 'tan', 'dodgerblue', 'deeppink', 'orangered', 'peru', 'blue', 'y', 'r',
              'gold', 'dimgray', 'darkorange', 'peru', 'blue', 'y', 'r', 'cyan', 'tan', 'orchid', 'peru', 'blue', 'y', 'r', 'sienna']
    markers = ['*', 'h', 'H', '+', 'o', '1', '2', '3', ',', 'v', 'H', '+', '1', '2', '^',
               '<', '>', '.', '4', 'H', '+', '1', '2', 's', 'p', 'x', 'D', 'd', '|', '_']
    for i in range(len(canopies)):
        canopy = canopies[i]
        center = canopy[0]
        components = canopy[1]
        sc.plot(center[0], center[1], marker=markers[i],
                color=colors[i], markersize=10)
        t1_circle = plt.Circle(
            xy=(center[0], center[1]), radius=t1, color='dodgerblue', fill=False)
        t2_circle = plt.Circle(
            xy=(center[0], center[1]), radius=t2, color='skyblue', alpha=0.2)
        sc.add_artist(t1_circle)
        sc.add_artist(t2_circle)
        for component in components:
            sc.plot(component[0], component[1],
                    marker=markers[i], color=colors[i], markersize=1.5)
    maxvalue = np.amax(dataset)
    minvalue = np.amin(dataset)
    plt.xlim(minvalue - t1, maxvalue + t1)
    plt.ylim(minvalue - t1, maxvalue + t1)
    plt.show()

def main():
    csvdata = read_points()
    X = len(csvdata[0])
    Y = len(csvdata)
    New_matrix = np.zeros([Y,X])
    for y in range(Y):
        for x in range(X):
            New_matrix[y, x] = csvdata[y][x]



    #最后三列统计结果矩阵
    add_matrix = np.zeros([Y,3])
    for y in range(Y):
        add_matrix[y, 0] = New_matrix[y, 0] - New_matrix[y, 1]
        add_matrix[y, 1] = New_matrix[y, 2] - New_matrix[y, 3]
        add_matrix[y, 2] = New_matrix[y, 4] - New_matrix[y, 5]

    temp_mean = add_matrix[:, 0].mean()
    temp_mean1 = add_matrix[:, 1].mean()
    temp_mean2 = add_matrix[:, 2].mean()
    col_std = np.std(add_matrix, axis=0)
    col_mean = np.mean(add_matrix, axis=0)
    # 需要对csvdata进行中心化和标准化处理
    for y in range(Y):
        for x in range(0,3):
            add_matrix[y, x] -= col_mean[x]
            add_matrix[y, x] /= col_std[x]

#    for y in range(Y):
#        temp = str(add_matrix[y,0]) + "," + str(add_matrix[y,1]) + "," + str(add_matrix[y,2]) + "\n"
#        saveresult(temp)



    predict_matrix = np.array([(7898765467.24,3235676823.00,3957177004.54,3444000321.55,5432112345.77,2900000089.12),
                               (133241575988.56, 39872238928.11, 14551119352.78, 3164290276.21, 3444305407.86,1015886389.47),
                               (93805217949.67,34975605193.08,2326015727.05,1922978314.70,2273603448.91,4777927001.24)])
    predict_subtract = np.zeros([3,3])
    for i in range(3):
        predict_subtract[i, 0] = predict_matrix[i, 0] - predict_matrix[i, 1]
        predict_subtract[i, 1] = predict_matrix[i, 2] - predict_matrix[i, 3]
        predict_subtract[i, 2] = predict_matrix[i, 4] - predict_matrix[i, 5]
    for i in range(3):
        for x in range(0,3):
            predict_subtract[i, x] -= col_mean[x]
            predict_subtract[i, x] /= col_std[x]


    matrix2list = add_matrix.tolist()
    print(matrix2list)
    print(add_matrix)
#    聚类前降维显示
#    Dimensionality_reduction(matrix2list)

    X_pca = PCA(n_components=2).fit_transform(add_matrix)
    t1 = 20
    t2 = 15
    gc = ca.Canopy(X_pca)
    gc.setThreshold(t1, t2)
    canopies = gc.clustering()
#    showCanopy(canopies,X_pca,t1,t2)

    all_vrc = []
    all_silh = []
    sub = []
    for k in range(20):
        # kmeans聚类
        if k==0 or k==1:
            continue
        clf = KMeans(n_clusters=k,init='k-means++')
        y_pred = clf.fit_predict(matrix2list)
        add_pred = clf.predict(predict_subtract.tolist())
#        print(clf)
#        print(y_pred)

        sub.append(k)
        VRC = metrics.calinski_harabaz_score(add_matrix, y_pred)
        all_vrc.append(VRC)
        silh = metrics.silhouette_score(add_matrix, y_pred, metric='euclidean')
        all_silh.append(silh)
        print("k= ",k)
        print('VRC方差率：',VRC)
     #   print('轮廓系数：%10.3f' % silh)
        print('轮廓系数：', silh)

        lines = len(y_pred)
        static_result = np.zeros([1, k])
        first = 0
        second = 0
        third = 0

        for item in y_pred:
           for i in range(0,k):
               if item == i:
                   static_result[0,i] += 1

        for i , j in enumerate(y_pred):
            if j == 1:
                print(i,j)

        for i in range(0,k):
            print("第"+str(i)+"类数据占比: "+str(static_result[0,i]*100/lines)+"%")

# 降维显示数据
        X_tsne = TSNE(learning_rate=100).fit_transform(matrix2list)
        X_pca = PCA().fit_transform(matrix2list)
        #单点降维
        signal_pca = PCA().fit_transform(predict_subtract.tolist())
        plt.close()
        fig = plt.figure()
        plt.ion()  # interactive mode on
        plt.subplot(121)
        plt.title("T-SNE")
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_pred)
        plt.subplot(122)
        plt.title("PCA")
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred)
        plt.scatter(signal_pca[:,0],signal_pca[:,1],c='r')
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


    print("第一类数据占比：" + str(((first * 100) / lines)) + "%")
    print("第二类数据占比：" + str(((second * 100) / lines)) + "%")
    print("第三类数据占比：" + str(((third * 100) / lines)) + "%")
    temp = "第一类数据占总数的:" + str(((first * 100) / lines)) + "%\n" + "第二类数据占总数的:" + str(((second * 100) / lines)) + "%\n" + "第三类数据占总数的:" + str(((third * 100) / lines)) + "%\n"
    saveresult(temp)

    X_tsne = TSNE(learning_rate=100).fit_transform(matrix2list)
    X_pca = PCA().fit_transform(matrix2list)
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.title("T-SNE")
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_pred)
    plt.subplot(122)
    plt.title("PCA")
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred)
    plt.show()

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