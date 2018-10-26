import matplotlib.pyplot as plt
import numpy as np
import random

def siglog(inX):
    return np.exp(inX)/(1+np.exp(inX))


from sklearn.datasets import load_iris
data=load_iris()      #导入数据库到data中
type(data)
a = np.mat(data.data)   #将数据库中的数据即x写成矩阵形式，如果不这样好像是不能进行操作
b = np.mat(data.target) #将数据库中的目标即y写成矩阵形式
b = b.transpose()   #将b矩阵进行转置成150*1
b[100:] = 1 #因为这个数据库中由三类数据，但是我先按照两类来进行分类，所以将数据的目标中的2改成1
column = np.ones(150)
a = np.column_stack((a,column)) #构建了一个全为1的矩阵加入到a矩阵的最后一列，当作常数项即x0 = 1
a = np.delete(a,[2,3],axis = 1) #因为这数据库的数据有4个特征而我们为了表示成二维图像，所以将后面的2个特征进行了省略

alpha = 0.05    #设定学习步长
maxCycle = 500  #设定总循环数
m,n = np.shape(a)   #得到矩阵a的m*n
weights = np.ones((n,1))    #首先设定权重都为1，后面就是循环来改善weights的值
for k in range(maxCycle):
    h = siglog(a*weights)   #将Wi*Xi的值带入到sigmod函数中，将值变成（0,1）的范围
    error = (b - h) #b为设定的值
    weights = weights + alpha*(a.transpose())*error #这是用梯度上升法进行求解weights，
                                                    # 可以参照上文中用梯度下降得出的最后公式。
                                                    # 梯度上升和梯度下降就是符号反一下


x_index=0
y_index=1
colors=['blue','red','green']
for label,color in zip(range(len(data.target_names)),colors):
    plt.scatter(data.data[data.target==label,x_index],
                data.data[data.target==label,y_index],
                label=data.target_names[label],
                c=color)    #将数据以二维的方式进行绘图表示
# fig = plt.figure()
# f = eyes(35)
# ax = fig.add_subplot(111)
x = np.arange(4.5,8.0,0.1)  #设定x的范围
y = (-weights[-1]-weights[0]*x)/weights[1]  #因为WO*X0+W1*X1+W2*X2 = 0 所以X2 = (-W0+W1*X1)/W2
                                            # （因为能取的类别就0和1，0是分界线。所以将Wi*XI的值设为0）
y = y.transpose()   #这里y一定要转置，不然不能绘制出直线，会报错。这是与机器学习实践不同的地方
plt.plot(x,y)
plt.xlabel(data.feature_names[x_index])
plt.ylabel(data.feature_names[y_index])
plt.legend(loc='upper left')
plt.show()




